from typing import Any, Dict, Optional

import functorch
import torch
import torch.nn as nn

from .problems import InitialValueProblem
from .single_step_methods import SingleStepMethod
from .solution import Solution
from .status_codes import Status
from .step_size_controllers import StepSizeController
from .terms import ODETerm
from .typing import *


class Adjoint:
    def solve(
        self,
        problem: InitialValueProblem,
        term: Optional[ODETerm] = None,
        dt0: Optional[TimeTensor] = None,
        args: Any = None,
    ) -> Solution:
        raise NotImplementedError()


class AutoDiffAdjoint(nn.Module):
    def __init__(
        self,
        step_method: SingleStepMethod,
        step_size_controller: StepSizeController,
        *,
        max_steps: Optional[int] = None,
        backprop_through_step_size_control: bool = True,
    ):
        super().__init__()

        self.step_method = step_method
        self.step_size_controller = step_size_controller
        self.max_steps = max_steps
        self.backprop_through_step_size_control = backprop_through_step_size_control

    @torch.jit.export
    def solve(
        self,
        problem: InitialValueProblem,
        term: Optional[ODETerm] = None,
        dt0: Optional[TimeTensor] = None,
        args: Any = None,
    ) -> Solution:
        step_method, step_size_controller = self.step_method, self.step_size_controller
        device, batch_size = problem.device, problem.batch_size
        t_start, t_end, t_eval = problem.t_start, problem.t_end, problem.t_eval
        time_direction = problem.time_direction.to(dtype=t_start.dtype)

        ###############################
        # Initialize the solver state #
        ###############################

        t = t_start
        y = problem.y0
        stats_n_steps = y.new_zeros(batch_size, dtype=torch.long)
        stats_n_accepted = y.new_zeros(batch_size, dtype=torch.long)
        stats: Dict[str, Any] = {}

        # TorchScript is not smart enough yet to figure out that we only access these
        # variables when they have been defined, so we have to always define them and
        # intialize them to any valid tensor.
        y_eval: torch.Tensor = y
        not_yet_evaluated: torch.Tensor = y
        minus_t_eval_normalized: torch.Tensor = y

        if t_eval is not None:
            y_eval = y.new_empty(
                (batch_size, problem.n_evaluation_points, problem.n_features)
            )

            # Keep track of which evaluation points have not yet been handled
            not_yet_evaluated = torch.ones_like(t_eval, dtype=torch.bool)

        # Normalize the time direction of the evaluation and end times for faster
        # comparisons
        minus_t_end_normalized = -time_direction * t_end
        if t_eval is not None:
            minus_t_eval_normalized = -time_direction[:, None] * t_eval

        # Keep track of which solves are still running
        running = y.new_ones(batch_size, dtype=torch.bool)

        # Initialize additional statistics to track for the integration term
        term_ = term
        if torch.jit.is_scripting() or term is None:
            assert term is None, "The integration term is fixed for JIT compilation"
            term_ = self.step_method.term
        assert term_ is not None
        term_.init(problem, stats)

        # Compute an initial step size
        convergence_order = step_method.convergence_order()
        dt, controller_state, f0 = step_size_controller.init(
            term, problem, convergence_order, dt0, stats=stats, args=args
        )
        method_state = step_method.init(term, problem, f0, stats=stats, args=args)

        # TorchScript does not support set_grad_enabled, so we detach manually
        if not self.backprop_through_step_size_control:
            dt = dt.detach()

        ##############################################
        # Take care of evaluation exactly at t_start #
        ##############################################

        # We copy the initial state into the evaluation if the first evaluation point
        # happens to be exactly `t_start`. This is required so that we can later assume
        # that rejection of the step (and therefore also no change in `t`) means that we
        # also did not pass any evaluation points.
        if t_eval is not None:
            eval_at_start = t_eval[:, 0] == t_start
            y_eval[eval_at_start, 0] = y[eval_at_start]
            not_yet_evaluated[eval_at_start, 0] = False

        ####################################
        # Solve the initial value problems #
        ####################################

        # Iterate the single step method until all ODEs have been solved up to their end
        # point or any of them failed
        max_steps = self.max_steps
        while True:
            step_out = step_method.step(
                term, running, y, t, dt, method_state, stats=stats, args=args
            )
            step_result, interp_data, method_state_next, method_status = step_out
            controller_out = step_size_controller.adapt_step_size(
                t, dt, y, step_result, controller_state, stats
            )
            accept, dt_next, controller_state_next, controller_status = controller_out

            # TorchScript does not support set_grad_enabled, so we detach manually
            if not self.backprop_through_step_size_control:
                dt_next = dt_next.detach()

            # Update the solver state where the step was accepted
            to_update = accept & running
            t = torch.where(to_update, t + dt, t)
            y = torch.where(to_update[:, None], step_result.y, y)
            method_state = step_method.merge_states(
                to_update, method_state_next, method_state
            )

            #####################
            # Update statistics #
            #####################

            stats_n_steps.add_(running)
            stats_n_accepted.add_(to_update)

            ##################################
            # Update solver state and status #
            ##################################

            # Stop a solve if `t` has passed its endpoint in the direction of time
            running = torch.addcmul(minus_t_end_normalized, time_direction, t) < 0.0

            status = method_status
            if status is None:
                status = controller_status
            elif controller_status is not None:
                status = torch.maximum(status, controller_status)
            if max_steps is not None:
                status = torch.where(
                    stats_n_steps >= max_steps,
                    Status.REACHED_MAX_STEPS.value,
                    status if status is not None else Status.SUCCESS.value,
                )

            # We evaluate the termination condition here already and initiate a
            # non-blocking transfer to the CPU to increase the chance that we won't have
            # to wait for the result when we actually check the termination condition
            continue_iterating = torch.any(running)
            if status is not None:
                continue_iterating = continue_iterating & torch.all(
                    status == Status.SUCCESS.value
                )
            continue_iterating = continue_iterating.to("cpu", non_blocking=True)

            # There is a bug as of pytorch 1.12.1 where non-blocking transfer from
            # device to host can sometimes gives the wrong result, so we place this
            # event after the transfer to ensure that the transfer has actually happened
            # by the time we evaluate the result.
            if device.type == "cuda":
                continue_iterating_done = torch.cuda.Event()
                continue_iterating_done.record(torch.cuda.current_stream(device))
            else:
                continue_iterating_done = None

            #########################
            # Evaluate the solution #
            #########################

            # Evaluate the solution at all evaluation points that have been passed in
            # this step.
            #
            # We always build the interpolation and evaluate it, even if no evaluation
            # points have actually been passed, because this avoids a CPU-GPU
            # synchronization and for time series models we expect that most steps will
            # pass at least one evaluation point across the whole batch (usually more).
            if t_eval is not None:
                to_be_evaluated = (
                    torch.addcmul(
                        minus_t_eval_normalized,
                        time_direction[:, None],
                        t[:, None],
                    )
                    > 0.0
                ) & not_yet_evaluated
                if to_be_evaluated.any():
                    interpolation = step_method.build_interpolation(interp_data)
                    nonzero = to_be_evaluated.nonzero()
                    sample_idx, eval_t_idx = nonzero[:, 0], nonzero[:, 1]
                    y_eval[sample_idx, eval_t_idx] = interpolation.evaluate(
                        t_eval[sample_idx, eval_t_idx], sample_idx
                    )

                    not_yet_evaluated = torch.logical_xor(
                        to_be_evaluated, not_yet_evaluated
                    )

            ########################
            # Update the step size #
            ########################

            # We update the step size and controller state only for solves which will
            # still be running in the next iteration. Otherwise, a finished instance
            # with an adaptive step size controller could reach infinite step size if
            # its final error was small and another instance is running for many steps.
            # This would then cancel the solve even though the "problematic" instance is
            # not even supposed to be running anymore.

            dt = torch.where(running, dt_next, dt)
            controller_state = step_size_controller.merge_states(
                running, controller_state_next, controller_state
            )

            if continue_iterating_done is not None:
                continue_iterating_done.synchronize()
            if continue_iterating:
                continue

            ##################################################
            # Finalize the solver and construct the solution #
            ##################################################

            # Ensure that the user always gets a status tensor
            if status is None:
                status = torch.tensor(
                    Status.SUCCESS.value, dtype=torch.long, device=device
                ).expand(batch_size)

            # Put the step statistics into the stats dict in the end, so that
            # we don't have to type-assert all the time in torchscript
            stats["n_steps"] = stats_n_steps
            stats["n_accepted"] = stats_n_accepted

            # The finalization scope is in the scope of the while loop so that the
            # `t_eval is None` case can access the `interp_data` in TorchScript.
            # Declaring `interp_data` outside of the loop does not work because its type
            # depends on the step method.

            if t_eval is not None:
                # Report the number of evaluation steps that have been initialized with
                # actual data at termination. Depending on the termination condition,
                # the data might be NaN or inf but it will not be uninitialized memory.
                #
                # As of torch 1.12.1, searchsorted is not implemented for bool tensors,
                # so we convert to int first.
                stats["n_initialized"] = torch.searchsorted(
                    not_yet_evaluated.int(),
                    torch.ones((batch_size, 1), dtype=torch.int, device=device),
                ).squeeze(dim=1)

                return Solution(ts=t_eval, ys=y_eval, stats=stats, status=status)
            else:
                # Evaluate the solution only at the end point, e.g. in continuous
                # normalizing flows
                interpolation = step_method.build_interpolation(interp_data)
                y_end = interpolation.evaluate(
                    t_end, torch.arange(batch_size, device=device)
                )

                stats["n_initialized"] = torch.ones(
                    batch_size, dtype=torch.long, device=device
                )

                return Solution(
                    ts=t_end[:, None], ys=y_end[:, None], stats=stats, status=status
                )

        assert False, "unreachable"

    def __repr__(self):
        return (
            f"AutoDiffAdjoint(step_method={self.step_method}, "
            f"step_size_controller={self.step_size_controller}, "
            f"max_steps={self.max_steps}, "
            f"backprop_through_step_size_control={self.backprop_through_step_size_control})"
        )


def flatten_tensors(*tensors):
    return [t.shape[1:] for t in tensors], torch.cat(
        [t.reshape((t.shape[0], -1)) for t in tensors], dim=1
    )


def unflatten_tensors(shapes, tensors):
    def prod(nums):
        p = 1
        for n in nums:
            p = p * n
        return p

    return [
        tensor.reshape((-1, *shape))
        for shape, tensor in zip(
            shapes, torch.split(tensors, [prod(shape) for shape in shapes], dim=1)
        )
    ]


class BacksolveFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        solver,
        aug_solver,
        term,
        aug_term,
        y0,
        t_start,
        t_end,
        t_eval,
        args,
        *adjoint_params,
    ):
        ctx.aug_solver = aug_solver
        ctx.aug_term = aug_term
        ctx.args = args

        problem = InitialValueProblem(y0, t_start, t_end, t_eval)
        with torch.no_grad():
            solution = solver.solve(problem, dt0=None, args=args, term=term)

        ctx.stats = solution.stats
        ctx.save_for_backward(t_start, t_end, t_eval, solution.ys, *adjoint_params)

        return solution.ts, solution.ys, solution.stats, solution.status

    @staticmethod
    def backward(ctx, grad_ts, grad_ys, grad_stats, grad_status):
        aug_solver = ctx.aug_solver
        aug_term = ctx.aug_term
        args = ctx.args
        t_start, t_end, t_eval, ys, *adjoint_params = ctx.saved_tensors

        stats = ctx.stats
        stats.setdefault("backsolve", [])

        batch_size, n_eval, n_features = ys.shape
        shapes, aug_state = flatten_tensors(
            ys.new_zeros((batch_size, 1)),
            ys[:, -1],
            grad_ys[:, -1],
            *[torch.zeros_like(p).expand(batch_size, *p.shape) for p in adjoint_params],
        )
        with torch.no_grad():
            if t_eval is None:
                problem = InitialValueProblem(
                    aug_state, t_start=t_end, t_end=t_start, t_eval=None
                )
                solution = aug_solver.solve(
                    problem, dt0=None, args=(shapes, args), term=aug_term
                )
                aug_state = solution.ys[:, -1]

                stats["backsolve"].append(solution.stats)
            else:
                for i in range(n_eval - 1, 0, -1):
                    problem = InitialValueProblem(
                        aug_state,
                        t_start=t_eval[:, i],
                        t_end=t_eval[:, i - 1],
                        t_eval=None,
                    )
                    solution = aug_solver.solve(
                        problem, dt0=None, args=(shapes, args), term=aug_term
                    )
                    aug_state = solution.ys[:, -1]
                    aug_state[:, 1 : 1 + n_features] = ys[:, i - 1]
                    aug_state[:, 1 + n_features : 1 + 2 * n_features] += grad_ys[
                        :, i - 1
                    ]

                    stats["backsolve"].append(solution.stats)

        _, _, grad_y0, *grad_params = unflatten_tensors(shapes, aug_state)
        # Accumulate gradients over samples
        grad_params = [p.sum(dim=0) for p in grad_params]

        return None, None, None, None, grad_y0, None, None, None, None, *grad_params


class AugmentedDynamicsTerm(nn.Module):
    def __init__(
        self, term: ODETerm, vmap_args_dims=None, vmap_randomness: str = "error"
    ):
        super().__init__()

        self.term = term
        self.vmap_args_dims = vmap_args_dims
        self.vmap_randomness = vmap_randomness

        self.func, self.params, self.buffers = functorch.make_functional_with_buffers(
            self.term.f
        )

        def vjp_single_sample(t_i, y_i, adj_y_i, arg_i):
            def wrapper(params, t_, y_):
                if self.term.with_args:
                    return self.func(params, self.buffers, t_, y_, arg_i)
                else:
                    return self.func(params, self.buffers, t_, y_)

            dy, vjp = functorch.vjp(wrapper, self.params, t_i, y_i)
            vjp_params, vjp_t, vjp_y = vjp(-adj_y_i)
            return dy, vjp_t, vjp_y, vjp_params

        self.vjp_vf = functorch.vmap(
            vjp_single_sample,
            in_dims=(0, 0, 0, self.vmap_args_dims),
            randomness=self.vmap_randomness,
        )

    def init(self, problem: InitialValueProblem, stats: Dict[str, Any]):
        return self.term.init(problem, stats)

    def vf(
        self, t: TimeTensor, y: DataTensor, stats: Dict[str, Any], args: Any
    ) -> DataTensor:
        shapes, args = args

        adj_t, y, adj_y, *_ = unflatten_tensors(shapes, y)
        dy, vjp_t, vjp_y, vjp_params = self.vjp_vf(t, y, adj_y, args)

        return flatten_tensors(vjp_t[:, None], dy, vjp_y, *vjp_params)[1]


class BacksolveAdjoint(nn.Module):
    def __init__(
        self,
        term,
        step_method,
        step_size_controller,
        vmap_args_dims=None,
        vmap_randomness="error",
    ):
        super().__init__()

        self.term = term
        self.augmented_term = AugmentedDynamicsTerm(
            self.term, vmap_args_dims=vmap_args_dims, vmap_randomness=vmap_randomness
        )
        self.forward_adjoint = AutoDiffAdjoint(step_method, step_size_controller)
        self.backward_adjoint = AutoDiffAdjoint(step_method, step_size_controller)

    def solve(
        self,
        problem: InitialValueProblem,
        term: Optional[ODETerm] = None,
        dt0: Optional[TimeTensor] = None,
        args: Any = None,
    ) -> Solution:
        ts, ys, stats, status = BacksolveFunction.apply(
            self.forward_adjoint,
            self.backward_adjoint,
            self.term,
            self.augmented_term,
            problem.y0,
            problem.t_start,
            problem.t_end,
            problem.t_eval,
            args,
            *list(self.term.parameters()),
        )
        return Solution(ts, ys, stats, status)

    def __repr__(self):
        return (
            f"BacksolveAdjoint(term={self.term!r}, "
            f"augmented_term={self.augmented_term!r}, "
            f"forward_adjoint={self.forward_adjoint!r}, "
            f"backward_adjoint={self.backward_adjoint!r})"
        )


class UnwrappingODETerm(nn.Module):
    def __init__(self, term: ODETerm):
        super().__init__()

        self.term = term

    def init(self, problem: InitialValueProblem, stats: Dict[str, Any]):
        return self.term.init(problem, stats)

    def vf(
        self, t: TimeTensor, y: DataTensor, stats: Dict[str, Any], args: Any
    ) -> DataTensor:
        """Evaluate the vector field."""
        batch_size, t_intercept, t_slope, term_args = args

        # Map from the integration interval of the first instance onto the other
        # integration intervals
        t = torch.addcmul(t_intercept, t_slope, t)

        dy = self.term.vf(t, y[0].reshape((batch_size, -1)), stats, term_args)

        # Correct the y-derivative for the distortion through the linear mapping
        # according to the substitution rule
        dy = dy * t_slope[:, None]

        return dy.flatten()[None]


class JointAugmentedDynamicsTerm(nn.Module):
    def __init__(self, term: UnwrappingODETerm):
        super().__init__()

        self.term = term
        self.parameters = list(self.term.parameters())

    def init(self, problem: InitialValueProblem, stats: Dict[str, Any]):
        return self.term.init(problem, stats)

    def vf(
        self, t: TimeTensor, y: DataTensor, stats: Dict[str, Any], args: Any
    ) -> DataTensor:
        shapes, term_args = args

        adj_t, y, adj_y, *_ = unflatten_tensors(shapes, y)

        with torch.enable_grad():
            t_ = t.detach().requires_grad_()
            y_ = y.detach().requires_grad_()
            dy = self.term.vf(t_, y_, stats, term_args)
            vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                dy,
                [t_, y_] + self.parameters,
                -adj_y,
                allow_unused=True,
                retain_graph=True,
            )

        if vjp_t is None:
            vjp_t = torch.zeros_like(t)
        if vjp_y is None:
            vjp_y = torch.zeros_like(y)
        vjp_params = [
            vp if vp is not None else torch.zeros_like(p)
            for p, vp in zip(self.parameters, vjp_params)
        ]

        dy_joint = dy.flatten()[None]
        vjp_params_joint = [p[None] for p in vjp_params]
        return flatten_tensors(vjp_t[:, None], dy_joint, vjp_y, *vjp_params_joint)[1]


class JointBacksolveAdjoint(nn.Module):
    def __init__(self, term, step_method, step_size_controller):
        super().__init__()

        self.term = UnwrappingODETerm(term)
        self.augmented_term = JointAugmentedDynamicsTerm(self.term)
        self.forward_loop = AutoDiffAdjoint(step_method, step_size_controller)
        self.backward_loop = AutoDiffAdjoint(step_method, step_size_controller)

    def solve(
        self,
        problem: InitialValueProblem,
        term: Optional[ODETerm] = None,
        dt0: Optional[TimeTensor] = None,
        args: Any = None,
    ) -> Solution:
        y0 = problem.y0.flatten()[None]
        t_start = problem.t_start[:1]
        t_end = problem.t_end[:1]
        t_eval = problem.t_eval
        if t_eval is not None:
            eval_steps = torch.diff(t_eval, dim=1)
            normalized_steps = eval_steps / torch.maximum(
                eval_steps[:, :1], t_eval.new_tensor(1e-8)
            )
            assert (normalized_steps - normalized_steps[0]).abs().max() < 1e-8, (
                "JointBacksolveAdjoint can only be applied if all instances in "
                "the batch are evaluated at the same points in time"
            )
            t_eval = t_eval[:1]

        # Compute linear maps from the integration interval (and evaluation points) of
        # any instance onto the integration interval of the first instance in the batch
        all_slopes = problem.t_end - problem.t_start
        t_slope = all_slopes / all_slopes[0]
        t_intercept = problem.t_start - t_slope * problem.t_start[0]

        _, ys, stats, status = BacksolveFunction.apply(
            self.forward_loop,
            self.backward_loop,
            self.term,
            self.augmented_term,
            y0,
            t_start,
            t_end,
            t_eval,
            (problem.batch_size, t_intercept, t_slope, args),
            *list(self.term.parameters()),
        )

        ys = ys[0].unflatten(dim=1, sizes=(problem.batch_size, -1)).transpose(1, 0)
        if problem.t_eval is None:
            ts = problem.t_end
        else:
            ts = problem.t_eval
        return Solution(ts, ys, stats, status)

    def __repr__(self):
        return (
            f"JointBacksolveAdjoint(term={self.term!r}, "
            f"augmented_term={self.augmented_term!r}, "
            f"forward_loop={self.forward_loop!r}, "
            f"backward_loop={self.backward_loop!r})"
        )
