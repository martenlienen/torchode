from math import prod
from typing import Any

import torch
import torch.nn as nn
from torch.utils._pytree import (
    _broadcast_to_and_flatten,
    tree_flatten,
    tree_leaves,
    tree_unflatten,
)

from . import status_codes
from .problems import InitialValueProblem
from .single_step_methods import SingleStepMethod
from .solution import Solution
from .step_size_controllers import StepSizeController
from .terms import ODETerm
from .typing import *


class Adjoint:
    def solve(
        self,
        problem: InitialValueProblem,
        term: ODETerm | None = None,
        dt0: TimeTensor | None = None,
        args: Any = None,
    ) -> Solution:
        raise NotImplementedError()


class AutoDiffAdjoint(nn.Module):
    def __init__(
        self,
        step_method: SingleStepMethod,
        step_size_controller: StepSizeController,
        *,
        max_steps: int | None = None,
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
        term: ODETerm | None = None,
        dt0: TimeTensor | None = None,
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
        stats: dict[str, Any] = {}

        # Compute the boundaries in time to ensure that we never step outside of them
        t_min = torch.minimum(t_start, t_end)
        t_max = torch.maximum(t_end, t_start)

        # TorchScript is not smart enough yet to figure out that we only access these
        # variables when they have been defined, so we have to always define them and
        # initialize them to any valid tensor.
        y_eval: torch.Tensor = y
        not_yet_evaluated: torch.Tensor = y
        minus_t_eval_normalized: torch.Tensor = y

        if t_eval is not None:
            # Initialize to NaN to make it easy to detect usage of uninitialized
            # results
            y_eval = y.new_full(
                (batch_size, problem.n_evaluation_points, problem.n_features),
                float("nan"),
            )

            # Keep track of which evaluation points have not yet been handled
            not_yet_evaluated = torch.ones_like(t_eval, dtype=torch.bool)

        # Normalize the time direction of the evaluation and end times for faster
        # comparisons
        minus_t_end_normalized = -time_direction * t_end
        if t_eval is not None:
            minus_t_eval_normalized = -time_direction[:, None] * t_eval

        # Keep track of which solves are still running
        running = t_start != t_end
        status = torch.full(
            (batch_size,), status_codes.SUCCESS, dtype=torch.long, device=device
        )

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

        # Ensure that the initial dt does not step outside of the time domain
        dt = torch.clamp(dt, t_min - t, t_max - t)

        # TorchScript does not support set_grad_enabled, so we detach manually
        if not self.backprop_through_step_size_control:
            dt = dt.detach()

        ##############################################
        # Take care of evaluation exactly at t_start #
        ##############################################

        # We copy the initial state into all evaluation points that are exactly
        # `t_start`. This is required so that we can later assume that rejection
        # of the step (and therefore also no change in `t`) means that we also
        # did not pass any evaluation points.
        if t_eval is not None:
            eval_at_start = t_eval == t_start[:, None]
            y_eval = torch.where(eval_at_start[..., None], y[:, None], y_eval)
            not_yet_evaluated = not_yet_evaluated & ~eval_at_start

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

            # A solve is unfinished while `t` has not passed its endpoint in the
            # direction of time
            unfinished = running & (
                torch.addcmul(minus_t_end_normalized, time_direction, t) < 0.0
            )

            if method_status is not None:
                # If the step method failed, the solve failed
                status = torch.where(running, method_status, status)
            if controller_status is not None:
                # If the controller failed, but the solve has finished, we
                # ignore the error, because it would only affect the next dt
                status = torch.where(
                    unfinished & (status == status_codes.SUCCESS),
                    controller_status,
                    status,
                )
            if max_steps is not None:
                status = torch.where(
                    unfinished
                    & (status == status_codes.SUCCESS)
                    & (stats_n_steps >= max_steps),
                    status_codes.REACHED_MAX_STEPS,
                    status,
                )
            # Successful solves that are not finished yet keep running
            running = unfinished & (status == status_codes.SUCCESS)

            # We evaluate the termination condition here already and initiate a
            # non-blocking transfer to the CPU to increase the chance that we won't have
            # to wait for the result when we actually check the termination condition
            continue_iterating = torch.any(running).to("cpu", non_blocking=True)

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
            # This causes a blocking CPU-GPU sync point at to_be_evaluated.any
            # when t_eval is not None, but deferring this sync doesn't seem to
            # yield a speedup. A sync is necessary at each evaluation point anyway
            # since nonzero produces a variable-shape result.
            # See https://github.com/martenlienen/torchode/issues/46
            if t_eval is not None:
                to_be_evaluated = (
                    torch.addcmul(
                        minus_t_eval_normalized,
                        time_direction[:, None],
                        t[:, None],
                    )
                    >= 0.0
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

            # Ensure that we do not step outside of the time domain, even for instances
            # that are not running anymore
            dt = torch.clamp(dt, t_min - t, t_max - t)

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
                # actual data at termination.
                #
                # As of torch 1.12.1, searchsorted is not implemented for bool tensors,
                # so we convert to int first.
                stats["n_initialized"] = torch.searchsorted(
                    not_yet_evaluated.int(),
                    torch.ones((batch_size, 1), dtype=torch.int, device=device),
                ).squeeze(dim=1)

                return Solution(ts=t_eval, ys=y_eval, stats=stats, status=status)
            else:
                # Evaluate successful solutions only at the end point, e.g. in
                # continuous normalizing flows
                interpolation = step_method.build_interpolation(interp_data)
                success = status == status_codes.SUCCESS
                y_end = interpolation.evaluate(
                    t_end, torch.arange(batch_size, device=device)
                )
                y_end = torch.where(success[:, None], y_end, float("nan"))

                stats["n_initialized"] = success.long()

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
    return [
        tensor.reshape((-1, *shape))
        for shape, tensor in zip(
            shapes, torch.split(tensors, [prod(shape) for shape in shapes], dim=1)
        )
    ]


def ensure_args_do_not_require_grad(args):
    if torch.is_grad_enabled() and any(
        isinstance(leaf, torch.Tensor) and leaf.requires_grad
        for leaf in tree_leaves(args)
    ):
        raise NotImplementedError(
            "Backsolve adjoints cannot compute gradients with respect to tensors in "
            "args. Detach them or use AutoDiffAdjoint."
        )


def expand_stats(stats: dict[str, Any], batch_size: int):
    """Expand the stats of a joint solve in place to every instance in the batch."""

    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            stats[key] = value.expand(batch_size, *value.shape[1:])


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
        dt0,
        backward_dt0,
        args,
        *adjoint_params,
    ):
        ctx.term = term
        ctx.aug_solver = aug_solver
        ctx.aug_term = aug_term
        ctx.args = args

        problem = InitialValueProblem(y0, t_start, t_end, t_eval)
        with torch.no_grad():
            solution = solver.solve(problem, dt0=dt0, args=args, term=term)

        ctx.stats = solution.stats
        ctx.save_for_backward(
            y0, t_start, t_end, t_eval, backward_dt0, solution.ys, *adjoint_params
        )

        return solution.ts, solution.ys, solution.stats, solution.status

    @staticmethod
    def backward(ctx, grad_ts, grad_ys, grad_stats, grad_status):
        aug_solver = ctx.aug_solver
        aug_term = ctx.aug_term
        args = ctx.args
        y0, t_start, t_end, t_eval, dt0, ys, *adjoint_params = ctx.saved_tensors

        stats = ctx.stats
        stats.setdefault("backsolve", [])

        ##########################
        # y and params gradients #
        ##########################

        batch_size, n_eval, n_features = ys.shape
        shapes, aug_state = flatten_tensors(
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
                    problem, dt0=dt0, args=(shapes, args), term=aug_term
                )
                aug_state = solution.ys[:, -1]

                stats["backsolve"].append({"status": solution.status, **solution.stats})
            else:
                for i in range(n_eval - 1, 0, -1):
                    problem = InitialValueProblem(
                        aug_state,
                        t_start=t_eval[:, i],
                        t_end=t_eval[:, i - 1],
                        t_eval=None,
                    )
                    solution = aug_solver.solve(
                        problem, dt0=dt0, args=(shapes, args), term=aug_term
                    )
                    aug_state = solution.ys[:, -1]
                    aug_state[:, :n_features] = ys[:, i - 1]
                    aug_state[:, n_features : 2 * n_features] += grad_ys[:, i - 1]

                    stats["backsolve"].append(
                        {"status": solution.status, **solution.stats}
                    )

                # Evaluation points need not include the initial time. Propagate
                # the first observation's cotangent through the remaining interval.
                needs_initial_solve = t_eval[:, 0] != t_start
                if needs_initial_solve.any():
                    problem = InitialValueProblem(
                        aug_state,
                        t_start=t_eval[:, 0],
                        t_end=t_start,
                        t_eval=None,
                    )
                    solution = aug_solver.solve(
                        problem, dt0=dt0, args=(shapes, args), term=aug_term
                    )
                    # Keep aligned members unchanged
                    aug_state = torch.where(
                        needs_initial_solve[:, None], solution.ys[:, -1], aug_state
                    )

                    stats["backsolve"].append(
                        {"status": solution.status, **solution.stats}
                    )

        _, grad_y0, *grad_params = unflatten_tensors(shapes, aug_state)
        # Accumulate gradients over samples
        grad_params = [p.sum(dim=0) for p in grad_params]

        ##################
        # Time gradients #
        ##################

        grad_t_start, grad_t_end, grad_t_eval = None, None, None
        if any(ctx.needs_input_grad[5:8]):
            term = ctx.term
            vf_stats: dict[str, Any] = {}
            term.init(InitialValueProblem(y0, t_start, t_end, t_eval), vf_stats)

            def project_vf(t, y, cotangent):
                dy = term.vf(t, y, vf_stats, args)
                return (cotangent * dy).sum(dim=-1).to(dtype=t.dtype)

            # Moving an evaluation point moves the solution along f, and moving
            # t_start acts like perturbing y0 by -f(t_start, y0)
            grad_t_start = -project_vf(t_start, y0, grad_y0)
            eval_times = t_end[:, None] if t_eval is None else t_eval
            grad_eval_times = grad_ts + torch.stack(
                [
                    project_vf(eval_times[:, i], ys[:, i], grad_ys[:, i])
                    for i in range(n_eval)
                ],
                dim=1,
            )
            if t_eval is None:
                grad_t_end = grad_eval_times[:, 0]
            else:
                grad_t_eval = grad_eval_times

        return (
            None,
            None,
            None,
            None,
            grad_y0,
            grad_t_start,
            grad_t_end,
            grad_t_eval,
            None,
            None,
            None,
            *grad_params,
        )


def add_batch_dim(args, in_dims):
    """Restore the dimensions that `vmap` with `in_dims` removed from `args`."""

    leaves, spec = tree_flatten(args)
    dims = _broadcast_to_and_flatten(in_dims, spec)
    leaves = [x if d is None else x.unsqueeze(d) for x, d in zip(leaves, dims)]
    return tree_unflatten(leaves, spec)


class AugmentedDynamicsTerm(nn.Module):
    def __init__(
        self, term: ODETerm, vmap_args_dims=None, vmap_randomness: str = "error"
    ):
        super().__init__()

        self.term = term
        self.vmap_args_dims = vmap_args_dims
        self.vmap_randomness = vmap_randomness

        def vjp_single_sample(t_i, y_i, adj_y_i, arg_i):
            # Look up parameters and buffers on every call, because they can be
            # reassigned or converted after construction, e.g. by `.to()`
            f = self.term.f
            params = dict(f.named_parameters())
            trainable = {k: p for k, p in params.items() if p.requires_grad}
            # Pass frozen parameters with buffers to avoid computing their
            # gradients
            constants = {k: p for k, p in params.items() if not p.requires_grad}
            constants.update(f.named_buffers())

            def wrapper(trainable_values, y_):
                # vmap removes the batch dimension, but f expects batched inputs
                # as in the forward pass, so pass it a batch of one instance
                f_args = (t_i[None], y_[None])
                if self.term.with_args:
                    f_args = f_args + (add_batch_dim(arg_i, self.vmap_args_dims),)
                trainable_dict = dict(zip(trainable.keys(), trainable_values))
                dy = torch.func.functional_call(f, (trainable_dict, constants), f_args)
                return dy[0]

            dy, vjp = torch.func.vjp(wrapper, tuple(trainable.values()), y_i)
            vjp_trainable, vjp_y = vjp(-adj_y_i)
            vjp_trainable = dict(zip(trainable.keys(), vjp_trainable))
            vjp_params = [
                vjp_trainable[k] if k in vjp_trainable else torch.zeros_like(p)
                for k, p in params.items()
            ]
            return dy, vjp_y, vjp_params

        self.vjp_vf = torch.func.vmap(
            vjp_single_sample,
            in_dims=(0, 0, 0, self.vmap_args_dims),
            randomness=self.vmap_randomness,
        )

    def init(self, problem: InitialValueProblem, stats: dict[str, Any]):
        return self.term.init(problem, stats)

    def vf(
        self, t: TimeTensor, y: DataTensor, stats: dict[str, Any], args: Any
    ) -> DataTensor:
        shapes, args = args

        y, adj_y, *_ = unflatten_tensors(shapes, y)
        dy, vjp_y, vjp_params = self.vjp_vf(t, y, adj_y, args)

        # Count calls since we bypass ODETerm.vf with torch.func.functional_call
        if self.term.with_stats:
            stats["n_f_evals"].add_(1)

        return flatten_tensors(dy, vjp_y, *vjp_params)[1]


class BacksolveAdjoint(nn.Module):
    """Compute gradients by solving the adjoint equations backwards in time.

    The dynamics `f` of `term` must be an `nn.Module`, because the backward pass
    differentiates the parameters in `term.parameters()`.
    """

    def __init__(
        self,
        term: ODETerm,
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
        term: ODETerm | None = None,
        dt0: TimeTensor | None = None,
        args: Any = None,
        backward_dt0: TimeTensor | None = None,
    ) -> Solution:
        ensure_args_do_not_require_grad(args)
        if term is None or term is self.term:
            term, augmented_term = self.term, self.augmented_term
        else:
            augmented_term = AugmentedDynamicsTerm(
                term,
                vmap_args_dims=self.augmented_term.vmap_args_dims,
                vmap_randomness=self.augmented_term.vmap_randomness,
            )
        if backward_dt0 is None and dt0 is not None:
            backward_dt0 = -dt0
        ts, ys, stats, status = BacksolveFunction.apply(
            self.forward_adjoint,
            self.backward_adjoint,
            term,
            augmented_term,
            problem.y0,
            problem.t_start,
            problem.t_end,
            problem.t_eval,
            dt0,
            backward_dt0,
            args,
            *list(term.parameters()),
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

    def init(self, problem: InitialValueProblem, stats: dict[str, Any]):
        return self.term.init(problem, stats)

    def vf(
        self, t: TimeTensor, y: DataTensor, stats: dict[str, Any], args: Any
    ) -> DataTensor:
        """Evaluate the vector field."""
        batch_size, term_args = args

        t = t.expand(batch_size)
        dy = self.term.vf(t, y[0].reshape((batch_size, -1)), stats, term_args)
        return dy.flatten()[None]


class JointAugmentedDynamicsTerm(nn.Module):
    def __init__(self, term: UnwrappingODETerm):
        super().__init__()

        self.term = term

    def init(self, problem: InitialValueProblem, stats: dict[str, Any]):
        return self.term.init(problem, stats)

    def vf(
        self, t: TimeTensor, y: DataTensor, stats: dict[str, Any], args: Any
    ) -> DataTensor:
        shapes, term_args = args

        y, adj_y, *_ = unflatten_tensors(shapes, y)

        # Look up parameters and buffers on every call, because they can be
        # reassigned or converted after construction, e.g. by `.to()`
        params = list(self.term.parameters())
        trainable = [p for p in params if p.requires_grad]
        with torch.enable_grad():
            y_ = y.detach().requires_grad_()
            dy = self.term.vf(t, y_, stats, term_args)
            vjp_y, *vjp_trainable = torch.autograd.grad(
                dy,
                [y_] + trainable,
                -adj_y,
                allow_unused=True,
                retain_graph=True,
            )

        if vjp_y is None:
            vjp_y = torch.zeros_like(y)
        vjp_trainable = iter(vjp_trainable)
        vjp_params = []
        for p in params:
            vp = next(vjp_trainable) if p.requires_grad else None
            vjp_params.append(torch.zeros_like(p) if vp is None else vp)

        dy_joint = dy.flatten()[None]
        vjp_params_joint = [p[None] for p in vjp_params]
        return flatten_tensors(dy_joint, vjp_y, *vjp_params_joint)[1]


class ExpandingStatsSolver:
    """Broadcast the stats and status of a joint solve to all instances of the batch"""

    def __init__(self, solver, batch_size: int):
        self.solver = solver
        self.batch_size = batch_size

    def solve(self, *args, **kwargs) -> Solution:
        solution = self.solver.solve(*args, **kwargs)
        expand_stats(solution.stats, self.batch_size)
        status = solution.status.expand(self.batch_size)
        return Solution(solution.ts, solution.ys, solution.stats, status)


class JointBacksolveAdjoint(nn.Module):
    """Compute gradients by solving the adjoint equations of the whole batch
    jointly.

    The dynamics `f` of `term` must be an `nn.Module`, because the backward pass
    differentiates the parameters in `term.parameters()`.
    """

    def __init__(self, term: ODETerm, step_method, step_size_controller):
        super().__init__()

        self.term = UnwrappingODETerm(term)
        self.augmented_term = JointAugmentedDynamicsTerm(self.term)
        self.forward_loop = AutoDiffAdjoint(step_method, step_size_controller)
        self.backward_loop = AutoDiffAdjoint(step_method, step_size_controller)

    def solve(
        self,
        problem: InitialValueProblem,
        term: ODETerm | None = None,
        dt0: TimeTensor | None = None,
        args: Any = None,
        backward_dt0: TimeTensor | None = None,
    ) -> Solution:
        ensure_args_do_not_require_grad(args)
        times = [problem.t_start[:, None], problem.t_end[:, None]]
        if problem.t_eval is not None:
            times.append(problem.t_eval)
        times = torch.cat(times, dim=1)
        if not (times == times[:1]).all():
            raise ValueError(
                "JointBacksolveAdjoint requires all instances in the batch to share "
                "t_start, t_end and t_eval. Use BacksolveAdjoint otherwise."
            )

        if term is None or term is self.term.term:
            term, augmented_term = self.term, self.augmented_term
        else:
            term = UnwrappingODETerm(term)
            augmented_term = JointAugmentedDynamicsTerm(term)

        y0 = problem.y0.flatten()[None]
        t_start = problem.t_start[:1]
        t_end = problem.t_end[:1]
        t_eval = None if problem.t_eval is None else problem.t_eval[:1]
        if dt0 is not None:
            dt0 = dt0[:1]
        if backward_dt0 is not None:
            backward_dt0 = backward_dt0[:1]

        if backward_dt0 is None and dt0 is not None:
            backward_dt0 = -dt0
        _, ys, stats, status = BacksolveFunction.apply(
            self.forward_loop,
            ExpandingStatsSolver(self.backward_loop, problem.batch_size),
            term,
            augmented_term,
            y0,
            t_start,
            t_end,
            t_eval,
            dt0,
            backward_dt0,
            (problem.batch_size, args),
            *list(term.parameters()),
        )

        batch_size = problem.batch_size
        ys = ys[0].unflatten(dim=1, sizes=(batch_size, -1)).transpose(1, 0)
        ts = problem.t_end[:, None] if problem.t_eval is None else problem.t_eval

        # Broadcast status and stats from the joint solve to all instances
        status = status.expand(batch_size)
        expand_stats(stats, batch_size)
        return Solution(ts, ys, stats, status)

    def __repr__(self):
        return (
            f"JointBacksolveAdjoint(term={self.term!r}, "
            f"augmented_term={self.augmented_term!r}, "
            f"forward_loop={self.forward_loop!r}, "
            f"backward_loop={self.backward_loop!r})"
        )
