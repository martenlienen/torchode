# Solver Statistics

torchode captures statistics about each solve in `sol.stats`. Each entry is a tensor with one value per instance in the batch.

```python
sol = solver.solve(problem)
print(sol.stats)
# => {'n_f_evals': tensor([26, 26]), 'n_steps': tensor([4, 2]),
# =>  'n_accepted': tensor([4, 2]), 'n_initialized': tensor([10, 10])}
```

## Entries

The built-in components collect the following statistics:

- `n_f_evals`: Number of evaluations of the dynamics (also known as NFE). The dynamics are always evaluated for the whole batch at once, so this number is the same for all instances regardless of how many steps they actually take. You can disable collecting this number by passing `with_stats=False` to your `ODETerm`.
- `n_steps`: Number of steps that the solver attempted for each instance, including rejected steps.
- `n_accepted`: Number of accepted steps for each instance. `n_steps - n_accepted` is the number of rejected steps.
- `n_initialized`: Number of evaluation points at the start of `t_eval` that hold solution values. For a successful solve, this is the number of evaluation points. If a solve fails (`status != to.Status.SUCCESS`), only the first `n_initialized` entries of its `sol.ys` are valid and the rest is NaN. Without `t_eval`, the solution is only evaluated at `t_end` and this entry is 1 for a successful solve and 0 for a failed one.

## Custom statistics

All solver components receive the `stats` dictionary and can record their own entries. See [Logging Solver Information](extra-stats.ipynb) for an example that tracks every step.

## Adjoint gradient computation statistics

`to.BacksolveAdjoint` and `to.JointBacksolveAdjoint` solve the adjoint equations backwards in time when you call `backward()`. The backward pass adds its statistics in-place to the `sol.stats` dictionary of the forward solve as `sol.stats["backsolve"]`. It holds a list with the statistics of each backward solve and the same keys as above. The backward pass solves one segment between each pair of consecutive evaluation points, starting from the last one, and one more from the first evaluation point back to `t_start` if they differ. Without `t_eval`, there is a single backward solve from `t_end` to `t_start`.

For example, with three evaluation points that start at `t_start`, there are two backward solves:

```python
solver = to.BacksolveAdjoint(
    term, to.Dopri5(), to.IntegralController(atol=1e-6, rtol=1e-3)
)
sol = solver.solve(problem)
sol.ys.sum().backward()
print(len(sol.stats["backsolve"]))
# => 2
print(sol.stats["backsolve"][0])
# => {'status': tensor([0, 0]), 'n_f_evals': tensor([32, 32]), 'n_steps': tensor([4, 5]),
# =>  'n_accepted': tensor([4, 5]), 'n_initialized': tensor([1, 1])}
```

In addition to the keys above, each entry in `sol.stats["backsolve"]` has a `"status"` key with one status code per instance. If a backward solve fails, the gradients of that instance with respect to `y0`, `t_start` and the parameters become NaN. Since the gradients of the parameters sum over all instances, they become NaN for the whole batch. The backward solves of the remaining segments then start from a NaN state and fail with `INFINITE_NORM`, so the first failed entry in the list shows the actual cause.

The backward `n_f_evals` count the evaluations of the augmented dynamics, which evaluate your model and its vector-Jacobian product together. They do not include the evaluations for gradients with respect to `t_start`, `t_end` and `t_eval`.

`to.JointBacksolveAdjoint` solves the whole batch as one joint system, so all instances share the same values in both the forward and the backward statistics.
