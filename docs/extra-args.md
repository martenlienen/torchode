# Passing Extra Arguments Along

Sometimes you might want to configure some additional parameters that are specific to the instances in your batch. For example, you could be solving a bunch of decay problems where each instance has its own decay parameter. For this case, you can instantiate the `to.ODETerm` with `with_args=True` to make it pass an arbitrary object as a third argument to your dynamics function `f`.

```python
import matplotlib.pyplot as pp
import torch
import torchode as to

def f(t, y, decay_rate):
    return -decay_rate[:, None] * y

term = to.ODETerm(f, with_args=True)
step_method = to.Tsit5(term=term)
step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
solver = to.AutoDiffAdjoint(step_method, step_size_controller)

y0 = torch.tensor([[1.2], [5.0]])
n_steps = 10
t_eval = torch.stack((torch.linspace(0, 5, n_steps), torch.linspace(3, 4, n_steps)))
decay_rate = torch.tensor([0.1, 5.0])

problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
sol = solver.solve(problem, args=decay_rate)

pp.plot(sol.ts[0], sol.ys[0])
pp.plot(sol.ts[1], sol.ys[1])
pp.show()
```

Note that torchode is completely oblivious to the contents of your `args` object. So you can pass tensors, tuples, custom classes or anything else to your dynamics. However, you have to take care of correct broadcasting yourself as in the example above.
