Physical Loss Terms
=======================

The supervised setting of the previous sections can quickly 
yield approximate solutions with a fairly simple training process, but what's
quite sad to see here is that we only use physical models and numerics
as an "external" tool to produce a big pile of data 😢.

```{figure} resources/physloss-overview.jpg
---
height: 220px
name: physloss-overview
---
Physical losses typically combine a supervised loss with a combination of derivatives from the neural network.
```

## Using physical models

We can improve this setting by trying to bring the model equations (or parts thereof)
into the training process. E.g., given a PDE for $\mathbf{u}(\mathbf{x},t)$ with a time evolution, 
we can typically express it in terms of a function $\mathcal F$ of the derivatives 
of $\mathbf{u}$ via  
$
  \mathbf{u}_t = \mathcal F ( \mathbf{u}_{x}, \mathbf{u}_{xx}, ... \mathbf{u}_{xx...x} ) ,
$
where the $_{\mathbf{x}}$ subscripts denote spatial derivatives with respect to one of the spatial dimensions
of higher and higher order (this can of course also include derivatives with respect to different axes).

In this context we can employ DL by approximating the unknown $\mathbf{u}$ itself 
with a NN, denoted by $\tilde{\mathbf{u}}$. If the approximation is accurate, the PDE
naturally should be satisfied, i.e., the residual $R$ should be equal to zero: 
$
  R = \mathbf{u}_t - \mathcal F ( \mathbf{u}_{x}, \mathbf{u}_{xx}, ... \mathbf{u}_{xx...x} ) = 0
$.

This nicely integrates with the objective for training a neural network: similar to before
we can collect sample solutions 
$[x_0,y_0], ...[x_n,y_n]$ for $\mathbf{u}$ with $\mathbf{u}(\mathbf{x})=y$. 
This is typically important, as most practical PDEs we encounter do not have unique solutions
unless initial and boundary conditions are specified. Hence, if we only consider $R$ we might
get solutions with random offset or other undesirable components. Hence the supervised sample points
help to _pin down_ the solution in certain places.
Now our training objective becomes

$\text{arg min}_{\theta} \ \alpha_0 \sum_i (f(x_i ; \theta)-y_i)^2 + \alpha_1 R(x_i) $,

where $\alpha_{0,1}$ denote hyperparameters that scale the contribution of the supervised term and 
the residual term, respectively. We could of course add additional residual terms with suitable scaling factors here.

Note that, similar to the data samples used for supervised training, we have no guarantees that the
residual terms $R$ will actually reach zero during training. The non-linear optimization of the training process
will minimize the supervised and residual terms as much as possible, but worst case, large non-zero residual 
contributions can remain. We'll look at this in more detail in the upcoming code example, for now it's important 
to remember that physical constraints in this way only represent _soft-constraints_, without guarantees
of minimizing these constraints.

## Neural network derivatives

In order to compute the residuals at training time, it would be possible to store 
the unknowns of $\mathbf{u}$ on a computational mesh, e.g., a grid, and discretize the equations of
$R$ there. This has a fairly long "tradition" in DL, and was proposed by Tompson et al. {cite}`tompson2017` early on.

Instead, a more widely used variant of employing physical soft-constraints {cite}`raissi2018hiddenphys`
uses fully connected NNs to represent $\mathbf{u}$. This has some interesting pros and cons that we'll outline in the following.
Due to the popularity of the version, we'll also focus on it in the following code examples and comparisons.

The central idea here is that the aforementioned general function $f$ that we're after in our learning problems
can be seen as a representation of a physical field we're after. Thus, the $\mathbf{u}(\mathbf{x})$ will 
be turned into $\mathbf{u}(\mathbf{x}, \theta)$ where we choose $\theta$ such that the solution to $\mathbf{u}$ is 
represented as precisely as possible.

One nice side effect of this viewpoint is that NN representations inherently support the calculation of derivatives. 
The derivative $\partial f / \partial \theta$ was a key building block for learning via gradient descent, as explained 
in {doc}`overview`. Here, we can use the same tools to compute spatial derivatives such as $\partial \mathbf{u} / \partial x$,
Note that above for $R$ we've written this derivative in the shortened notation as $\mathbf{u}_{x}$.
For functions over time this of course also works for $\partial \mathbf{u} / \partial t$, i.e. $\mathbf{u}_{t}$ in the notation above.

Thus, for some generic $R$, made up of $\mathbf{u}_t$ and $\mathbf{u}_{x}$ terms, we can rely on the backpropagation algorithm
of DL frameworks to compute these derivatives once we have a NN that represents $\mathbf{u}$. Essentially, this gives us a 
function (the NN) that receives space and time coordinates to produce a solution for $\mathbf{u}$. Hence, the input is typically
quite low-dimensional, e.g., 3+1 values for a 3D case over time, and often produces a scalar value or a spatial vector.
Due to the lack of explicit spatial sampling points, an MLP, i.e., fully-connected NN is the architecture of choice here.

To pick a simple example, Burgers equation in 1D,
$\frac{\partial u}{\partial{t}} + u \nabla u = \nu \nabla \cdot \nabla u $ , we can directly
formulate a loss term $R = \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} u$ that should be minimized as much as possible at training time. For each of the terms, e.g. $\frac{\partial u}{\partial x}$,
we can simply query the DL framework that realizes $u$ to obtain the corresponding derivative. 
For higher order derivatives, such as $\frac{\partial^2 u}{\partial x^2}$, we can typically simply query the derivative function of the framework twice. In the following section, we'll give a specific example of how that works in tensorflow.


## Summary so far

The approach above gives us a method to include physical equations into DL learning as a soft-constraint.
Typically, this setup is suitable for _inverse problems_, where we have certain measurements or observations
for which we want to find a PDE solution. Because of the high cost of the reconstruction (to be 
demonstrated in the following), the solution manifold typically shouldn't be overly complex. E.g., it is difficult 
to capture a wide range of solutions, such as with the previous supervised airfoil example, in this way.


