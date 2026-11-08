# Step Methods

The step method takes a single step of the solver from `t` to `t + dt`. It also estimates the error of the step, which an adaptive [step size controller](step-size-controllers.md) uses to accept or reject the step and choose the next step size, and it interpolates the solution within the step to evaluate it at the points in `t_eval`. All methods are explicit Runge-Kutta methods.

| Method      | Name in `solve_ivp` | Order | Evaluations of `f` per step | Error estimate |
|-------------|---------------------|-------|-----------------------------|----------------|
| `to.Tsit5`  | `"tsit5"`           | 5     | 6                           | yes            |
| `to.Dopri5` | `"dopri5"`          | 5     | 6                           | yes            |
| `to.Heun`   | `"heun"`            | 2     | 2                           | yes            |
| `to.Euler`  | `"euler"`           | 1     | 1                           | no             |

- `to.Tsit5`: The 5th order method by [Tsitouras](https://doi.org/10.1016/j.camwa.2011.06.002) and the default of `solve_ivp`. The Julia package DifferentialEquations.jl [recommends it](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/) for most non-stiff problems.
- `to.Dopri5`: The 5th order Dormand-Prince method, which is the default of `torchdiffeq` and the `RK45` method in SciPy.
- `to.Heun`: A 2nd order method that is cheaper per step but needs many more steps for accurate solutions. It can be a good choice if you only need low accuracy.
- `to.Euler`: The simplest method with 1st order. Since it does not estimate its error, adaptive controllers keep the initial step size, so use it with `to.FixedStepController` and an explicit `dt0`.
