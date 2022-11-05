# Comparison to other solvers

The main points that differentiate torchode from other PyTorch ODE solvers are of course
[JIT compatibility](./jit.ipynb) and batch parallelization. JIT compilation can speed up
your model and the fact that torchode is fully compatible means that you can compile your
complete model, even if it uses an ODE solver internally, for example for continuously
evolving latent representations. Compiled models can also participate in other advanced
PyTorch mechanisms such as [ONNX export](https://pytorch.org/docs/stable/onnx.html) or
[optimization with
nvFuser](https://pytorch.org/blog/introducing-nvfuser-a-deep-learning-compiler-for-pytorch/).

Keeping track of each sample's solver state in batched training separately means that
torchode can step every sample forward in time and accept or reject their steps
independently. Therefore, the error that your model makes on one sample does not influence
the solver on another sample and sample `x` will always take the same amount of steps and
produce the same error and gradient regardless of it it was batched with sample `y` or
sample `z`. In contrast, torchdiffeq and TorchDyn only track a single solver state and the
established batching method is to treat all samples `n` as one large ODE system instead of
an independent set of `n` ODEs. For well behaved models and ODEs, the effect is negligible
but can become important if you work with stiff equations, for example, or place high
importance on eliminating such unintended interactions between samples.

Additional points of differentiation are extensibility and solver statistics. In contrast
to the other PyTorch solvers, torchode not only gives you the number of function
evaluations but also the number of accepted and rejected steps which makes it easier to
understand the performance of neural ODE models. torchode is also easily extensible in all
aspects and your custom stepping methods etc. can use the same stats tracking system to
collect custom statistics.

## [torchdiffeq](https://github.com/rtqichen/torchdiffeq)

torchdiffeq solves ODEs and some extensions such as Jump-ODEs and does a great job at
that. It is easy to use and well implemented. Its drawback is that it is almost impossible
to extend without resorting to really ugly python hacks. If you would like to know how
many steps were rejected, implement your own step size controller or add another stepping
method such as `tsit5`, your only option is to fork the library. In torchode, each part is
an exchangeable component and it is trivial to plug in your own stepping method or wrap an
existing step size controller to track additional statistics.

## [TorchDyn](https://github.com/DiffEqML/torchdyn)

TorchDyn comes with everything but the kitchen sink when it comes to implicit models and
numerical methods. They have ODEs, SDEs, equilibrium models, hyper solvers, multiple
shooting methods, predefined datasets, tutorials and probably more. However, this gives
their library a huge surface area that is difficult to polish in all places at once. When
I checked out their library, it was at times difficult for me to tell which parts are
stable and which are ongoing research. In comparison, torchode focuses on one thing only,
ODE solving.

## [diffrax](https://github.com/patrick-kidger/diffrax)

diffrax was a major inspiration for torchode and you will find references to it throughout
the code. Besides the obvious difference that diffrax is a JAX library and torchode
targets PyTorch, they mostly differ in scope. torchode focuses on just ODE solving to keep
the code small, maintainable and easily understandable. In contrast, diffrax combines
ODEs, SDEs and CDEs in one common framework (their code is great nonetheless).

## Did I misrepresent a library?

If you feel that I misrepresent one or more of the other ODE solvers in this comparison or
my description is outdated, please message me so that we can get this page more accurate,
for example via [github issues](https://github.com/martenlienen/torchode/issues), email or
twitter.
