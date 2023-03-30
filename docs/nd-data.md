# Images, Graphs and N-d Tensors

torchode is written with a sharp focus on its target problem: being a high-performance ODE
solver for PyTorch. To keep the code base as simple as possible, it neglects some niceties
such as arbitrary data dimensions. torchode expects your data to be exactly 2-dimensional,
`batch x features`. If your samples are not 1-dimensional vectors, but for example images,
you have to manage the shapes yourself. This means that you have to flatten the inputs
when you hand them over to torchode and reshape them, when you pass them to a
convolutional layer, let's say.

```python
# Example with a batch of RGB images
b, c, w, h = 8, 3, 72, 72
y0 = torch.randn(b, w, h)

# Setup solver etc.
# ...

def f(t, y, args)
    c, w, h = args

    # Reshape 2D to B x C x W x H tensor and apply convolutional neural network
    y = some_cnn(y.reshape((-1, c, w, h)))

    # Flatten back to 2D when we return control back to torchode
    return y.flatten(start_dim=1)

term = to.ODETerm(f, with_args=True)
problem = to.InitialValueProblem(y0=y0.flatten(start_dim=1), t_eval=..)
sol = solver.solve(problem, args=(c, w, h))
```

If you are willing to contribute a small utility that does this shape flattening and
reshaping automatically, we would be happy about your contribution!
