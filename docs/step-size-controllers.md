# Step Size Controllers

Solving an ODE means following a gradient field through time by integrating the right hand side of the differential equation. A crucial aspect here is the step size, i.e. how far forward in time the solver goes at each step. A small step size promises small integration error but also requires many steps to cover the complete integration range and therefore increases the runtime. In the end, the step size has to be chosen to achieve a balance between the competing concerns of error and runtime.

## Adaptive Step Size

The default approach in `torchode` is automatic step size control. Here a controller (derived from the [PID controller](https://www.wikiwand.com/en/PID_controller)) regulates the step size automatically based on running error estimates to stay within certain (local) error bounds. Your first choice should be the `IntegralController`. It works well in most situations to keep the error within the absolute (`atol`) and relative (`rtol`) tolerances.

```python
import torchode as to

def f(t, y):
    return -0.5 * y

term = to.ODETerm(f)
step_method = to.Tsit5(term=term)
step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
solver = to.AutoDiffAdjoint(step_method, step_size_controller)
problem = to.InitialValueProblem(...)
sol = solver.solve(problem)
```

If you work with stiff dynamics, the more general `PIDController` might be useful and save a few solver steps as we have explored in [our paper](https://arxiv.org/abs/2210.12375). To use it, just drop it in as a replacement for the `IntegralController`.

```python
step_size_controller = to.PIDController(atol=1e-6, rtol=1e-3, pcoeff=0.2, icoeff=0.5, dcoeff=0.0, term=term)
```

## Fixed Step Size

If you know a good step size or want to ensure constant progress of the solver at the cost of error control, you can also fix a step size yourself with the `FixedStepController`. The important difference is that you now have to provide an initial step size to `solver.solve` which will also be used for all further steps.

```python
step_size_controller = to.FixedStepController()
solver = to.AutoDiffAdjoint(step_method, step_size_controller)
dt0 = torch.full((batch_size,), 0.01)
sol = solver.solve(problem, dt0=dt0)
```
