# Gradients & Backpropagation

There are two ways to compute gradients of the dynamics of an ODE, so the neural network
in the case of neural ODEs, with respect to the solution of the ODE. The first is to
backpropagate straight through the solver. After all, an ODE solver is just a series of
simple operations that define a dynamic computation graph that can be backpropagated
through with pytorch's autograd. This is implemented in `to.AutoDiffAdjoint`, so called
because it uses the autodiff/autograd mechanism. In general, this is the preferred method
as long as you have enough memory, because it is fast and gives accurate gradients.

If you run out of memory, you can compute gradients by solving the so called adjoint
equations, which basically solve the ODE backwards and track gradients along the way. This
is implemented in `to.BacksolveAdjoint`. Solving the adjoint equations requires the
computation of gradients of the model at different steps in time, which
`to.BacksolveAdjoint` implements with `torch.func`. It evaluates your model for each
instance separately with `torch.func.vmap`, passing a batch of size one, so your model
must not depend on the batch size or mix information between instances. If your extra
`args` hold per-instance tensors, pass `vmap_args_dims` to tell `to.BacksolveAdjoint`
which of their dimensions is the batch dimension. If your model is not compatible and
you get errors because of this, you can fall back to `to.JointBacksolveAdjoint`. This
computes the model gradients with pytorch's usual autograd and should always work but
comes with two caveats. However, to make this work, `to.JointBacksolveAdjoint` needs to
solve the `n` independent adjoint equations jointly as one joint system that is jointly
discretized. This breaks with torchode's approach of solving each ODE completely
independently, because the joint discretization introduces a subtle coupling between the
solutions of your batch of ODEs. Therefore, `to.JointBacksolveAdjoint` should be your
backpropagation choice of last resort. Furthermore, it is only applicable if all ODEs in
your batch share the same `t_start`, `t_end` and evaluation points.

Both backsolve adjoints need the dynamics `f` of your `to.ODETerm` to be an `nn.Module`,
because they look up the parameters to differentiate via `term.parameters()`.
`to.AutoDiffAdjoint` does not have this restriction.

## Failed solves

Computing gradients is only guaranteed to work for batches where all solves succeeded.
You might get gradients in other cases, e.g. by only backpropagating through valid `ys`
with `to.AutoDiffAdjoint`, but it is not guaranteed. Therefore, the safe pattern is to
skip the update for batches with failed solves:

```python
sol = solver.solve(problem)
if (sol.status == to.Status.SUCCESS.value).all():
    loss_fn(sol.ys).backward()
    optimizer.step()
```
