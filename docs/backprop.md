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
`to.BacksolveAdjoint` implements with `torch.func`. If your model is not compatible and
you get errors because of this, you can fall back to `to.JointBacksolveAdjoint`. This
computes the model gradients with pytorch's usual autograd and should always work but
comes with two caveats. However, to make this work, `to.JointBacksolveAdjoint` needs to
solve the `n` independent adjoint equations jointly as one joint system that is jointly
discretized. This breaks with torchode's approach of solving each ODE completely
independently, because the joint discretization introduces a subtle coupling between the
solutions of your batch of ODEs. Therefore, `to.JointBacksolveAdjoint` should be your
backpropagation choice of last resort. Furthermore, it is only applicable if all ODEs in
your batch have the same evaluation points.
