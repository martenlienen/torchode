# Solving ODEs

torchode solves a batch of initial value problems at once. Each instance in the batch has its own initial condition, time span and evaluation points, and the solver adapts the step size for each instance separately.

```python
import torch
import torchode as to

def f(t, y):
    return -0.5 * y

y0 = torch.tensor([[1.2], [5.0]])
t_eval = torch.stack((torch.linspace(0, 5, 10), torch.linspace(3, 4, 10)))

term = to.ODETerm(f)
step_method = to.Dopri5(term=term)
step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
solver = to.AutoDiffAdjoint(step_method, step_size_controller)

problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
sol = solver.solve(problem)
```

The components are:

- `to.ODETerm` wraps your dynamics `f(t, y)` and tracks function evaluations. It can also pass [extra arguments](extra-args.md) to `f`.
- `to.Dopri5` is the step method, i.e. the Runge-Kutta scheme that takes a single step. See [Step Methods](step-methods.md) for other available methods.
- `to.IntegralController` chooses the step size for each instance based on the estimated error. See [Step Size Controllers](step-size-controllers.md) for the alternatives.
- `to.AutoDiffAdjoint` runs the solver loop and computes gradients by backpropagating through it.
- `to.InitialValueProblem` holds the batch of problems. `y0` has to be 2-dimensional with shape `(batch, features)`, see [Images, Graphs and N-d Tensors](nd-data.md) for other shapes. `t_start` and `t_end` default to the first and last evaluation points. If you pass `t_start` and `t_end` without `t_eval`, the solution is only evaluated at `t_end`, which makes the solver loop faster.

## The solution

`solver.solve` returns a `to.Solution` with four attributes.

### `sol.ts`

The times at which the solution was evaluated, with shape `(batch, n_eval)`. This is `t_eval` if you gave evaluation points and `t_end[:, None]` otherwise. It has the dtype of the time domain, which can differ from the dtype of `y0`.

### `sol.ys`

The solution at the times in `sol.ts`, with shape `(batch, n_eval, features)` and the dtype of `y0`. So `sol.ys[i, j]` is the state of instance `i` at time `sol.ts[i, j]`. The solver computes it by interpolating within each step, so evaluation points and step size are independent.

If a solve fails, e.g. due to reaching the maximum number of steps, `sol.ys` holds NaN for all evaluation points that the solve did not reach. It is possible for a solution to succeed for some systems and fail for other, for example with variable step sizes, see `sol.status` below.

`sol.ys` is differentiable with respect to `y0`, the times in the problem and the parameters of your dynamics. If backpropagating through the solver needs too much memory, see [Gradients & Backpropagation](backprop.md) for adjoints that solve backwards in time instead.

### `sol.status`

One status code per instance with shape `(batch,)`. `0` (`to.Status.SUCCESS`) means that the solve succeeded and any other value means that it failed. The following status codes are possible:

- `SUCCESS`: The solve reached `t_end`.
- `REACHED_DT_MIN`: The step size controller would have needed a step smaller than its `dt_min`.
- `REACHED_MAX_STEPS`: The solve took `max_steps` steps without reaching `t_end`. You can set `max_steps` on the adjoint.
- `INFINITE_NORM`: The estimated error was infinite or NaN, usually because the dynamics returned infinite or NaN values.
- `GENERAL_ERROR`: A custom step method or step size controller reported an error. The built-in components do not use this code.

For example, if we limit the solver from above to 3 steps, the first instance fails after 6 of its 10 evaluation points:

```python
limited_solver = to.AutoDiffAdjoint(step_method, step_size_controller, max_steps=3)
limited_sol = limited_solver.solve(problem)
print(limited_sol.status)
# => tensor([3, 0])
print(limited_sol.ys[:, :, 0])
# => tensor([[1.2000, 0.9089, 0.6885, 0.5213, 0.3946, 0.2992,    nan,    nan,    nan,    nan],
# =>         [5.0000, 4.7298, 4.4742, 4.2324, 4.0036, 3.7873, 3.5826, 3.3890, 3.2059, 3.0327]])
```

torchode does not raise an error for failed solves, so check `sol.status` if your problems can fail.

Gradients are only guaranteed if all solves in the batch succeeded, see [Failed solves](backprop.md#failed-solves).

### `sol.stats`

A dictionary with statistics about the solve, such as the number of steps and function evaluations of each instance. See [Solver Statistics](stats.md) for all entries. For the successful solve from the first example:

```python
print(sol.stats)
# => {'n_f_evals': tensor([26, 26]), 'n_steps': tensor([4, 2]),
# =>  'n_accepted': tensor([4, 2]), 'n_initialized': tensor([10, 10])}
```
