# A Parallel ODE Solver for PyTorch

![pytest](https://github.com/martenlienen/torchode/actions/workflows/python-package.yml/badge.svg)

torchode is a suite of single-step ODE solvers such as `dopri5` or `tsit5` that are
compatible with PyTorch's JIT compiler and parallelized across a batch. JIT compilation
often gives a performance boost, especially for code with many small operations such as an
ODE solver, while batch-parallelization means that the solver can take a step of `0.1` for
one sample and `0.33` for another, depending on each sample's difficulty. This can avoid
performance traps for models of varying stiffness and ensures that the model's predictions
are independent from the compisition of the batch. See the
[paper](https://openreview.net/forum?id=uiKVKTiUYB0) for details.

- [*Documentation*](https://torchode.readthedocs.org)

If you get stuck at some point, you think the library should have an example on _x_ or you
want to suggest some other type of improvement, please open an [issue on
github](https://github.com/martenlienen/torchode/issues/new).

## Installation

You can get the latest released version from PyPI with

```sh
pip install torchode
```

To install a development version, clone the repository and install in editable mode:

```sh
git clone https://github.com/martenlienen/torchode
cd torchode
pip install -e .
```

## Usage

```python
import matplotlib.pyplot as pp
import torch
import torchode as to

def f(t, y):
    return -0.5 * y

y0 = torch.tensor([[1.2], [5.0]])
n_steps = 10
t_eval = torch.stack((torch.linspace(0, 5, n_steps), torch.linspace(3, 4, n_steps)))

term = to.ODETerm(f)
step_method = to.Dopri5(term=term)
step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
solver = to.AutoDiffAdjoint(step_method, step_size_controller)
jit_solver = torch.jit.script(solver)

sol = jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))
print(sol.stats)
# => {'n_f_evals': tensor([26, 26]), 'n_steps': tensor([4, 2]),
# =>  'n_accepted': tensor([4, 2]), 'n_initialized': tensor([10, 10])}

pp.plot(sol.ts[0], sol.ys[0])
pp.plot(sol.ts[1], sol.ys[1])
```

## Citation

If you build upon this work, please cite the following paper.

```
@inproceedings{lienen2022torchode,
  title = {torchode: A Parallel {ODE} Solver for PyTorch},
  author = {Marten Lienen and Stephan G{\"u}nnemann},
  booktitle = {The Symbiosis of Deep Learning and Differential Equations II, NeurIPS},
  year = {2022},
  url = {https://openreview.net/forum?id=uiKVKTiUYB0}
}
```
