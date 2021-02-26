Differentiable Physics
=======================

As a next step towards a tighter and more generic combination of deep learning
methods and deep learning we will target incorporating _differentiable physical 
simulations_ into the learning process. In the following, we'll shorten
that to "differentiable physics" (DP).

The central goal of this methods is to use existing numerical solvers, and equip
them with functionality to compute gradients with respect to their inputs.
Once this is realized for all operators of a simulation, we can leverage 
the autodiff functionality of DL frameworks with back-propagation to let gradient 
information from from a simulator into an NN and vice versa. This has numerous 
advantages such as improved learning feedback and generalization, as we'll outline below.
In contrast to physics-informed loss functions, it also enables handling more complex
solution manifolds instead of single inverse problems.

```{figure} resources/placeholder.png
---
height: 220px
name: dp-training
---
TODO, visual overview of DP training
```

## Differentiable Operators

With the DP direction we build on existing numerical solvers. I.e., 
the approach is strongly relying on the algorithms developed in the larger field 
of computational methods for a vast range of physical effects in our world.
To start with we need a continuous formulation as model for the physical effect that we'd like 
to simulate -- if this is missing we're in trouble. But luckily, we can resort to 
a huge library of established model equations, and ideally also on an established
method for discretization of the equation.

Let's assume we have a continuous formulation $\mathcal P^*(\mathbf{x}, \nu)$ of the physical quantity of 
interest $\mathbf{u}(\mathbf{x}, t): \mathbb R^d \times \mathbb R^+ \rightarrow \mathbb R^d$,
with model parameters $\nu$ (e.g., diffusion, viscosity, or conductivity constants).
The component of $\mathbf{u}$ will be denoted by a numbered subscript, i.e.,
$\mathbf{u} = (u_1,u_2,\dots,u_d)^T$.
%and a corresponding discrete version that describes the evolution of this quantity over time: $\mathbf{u}_t = \mathcal P(\mathbf{x}, \mathbf{u}, t)$.
Typically, we are interested in the temporal evolution of such a system, 
and discretization yields a formulation $\mathcal P(\mathbf{x}, \nu)$
that we can re-arrange to compute a future state after a time step $\Delta t$ via sequence of
operations $\mathcal P_1, \mathcal P_2 \dots \mathcal P_m$ such that
$\mathbf{u}(t+\Delta t) = \mathcal P_1 \circ \mathcal P_2 \circ \dots \mathcal P_m ( \mathbf{u}(t),\nu )$,
where $\circ$ denotes function decomposition, i.e. $f(g(x)) = f \circ g(x)$.

```{note} 
In order to integrate this solver into a DL process, we need to ensure that every operator
$\mathcal P_i$ provides a gradient w.r.t. its inputs, i.e. in the example above
$\partial \mathcal P_i / \partial \mathbf{u}$. 
```

Note that we typically don't need derivatives 
for all parameters of $\mathcal P$, e.g. we omit $\nu$ in the following, assuming that this is a 
given model parameter, with which the NN should not interact. 
Naturally, it can vary within the solution manifold that we're interested in, 
but $\nu$ will not be the output of a NN representation. If this is the case, we can omit
providing $\partial \mathcal P_i / \partial \nu$ in our solver. However, the following learning process
natuarlly transfers to including $\nu$ as a degree of freedom.

## Jacobians

As $\mathbf{u}$ is typically a vector-valued function, $\partial \mathcal P_i / \partial \mathbf{u}$ denotes
a Jacobian matrix $J$ rather than a single value:
% test
$$ \begin{aligned}
    \frac{ \partial \mathcal P_i }{ \partial \mathbf{u} } = 
    \begin{bmatrix} 
    \partial \mathcal P_{i,1} / \partial u_{1} 
    & \  \cdots \ &
    \partial \mathcal P_{i,1} / \partial u_{d} 
    \\
    \vdots & \ & \ 
    \\
    \partial \mathcal P_{i,d} / \partial u_{1} 
    & \  \cdots \ &
    \partial \mathcal P_{i,d} / \partial u_{d} 
    \end{bmatrix} 
\end{aligned} $$
where, as above, $d$ denotes the number of components in $\mathbf{u}$. As $\mathcal P$ maps one value of
$\mathbf{u}$ to another, the jacobian is square and symmetric here. Of course this isn't necessarily the case
for general model equations, but non-square Jacobian matrices would not cause any problems for differentiable 
simulations.

In practice, we can rely on the _reverse mode_ differentiation that all modern DL
frameworks provide, and focus on computing a matrix vector product of the Jacobian transpose
with a vector $\mathbf{a}$, i.e. the expression: 
$
    ( \frac{\partial \mathcal P_i }{ \partial \mathbf{u} } )^T \mathbf{a}
$. 
If we'd need to construct and store all full Jacobian matrices that we encounter during training, 
this would cause huge memory overheads and unnecessarily slow down training.
Instead, for backpropagation, we can provide faster operations that compute products
with the Jacobian transpose because we always have a scalar loss function at the end of the chain.

**[TODO check transpose of Jacobians in equations]**

Given the formulation above, we need to resolve the derivatives
of the chain of function compositions of the $\mathcal P_i$ at some current state $\mathbf{u}^n$ via the chain rule.
E.g., for two of them

$
    \frac{ \partial (\mathcal P_1 \circ \mathcal P_2) }{ \partial \mathbf{u} }|_{\mathbf{u}^n}
    = 
    \frac{ \partial \mathcal P_1 }{ \partial \mathbf{u} }|_{\mathcal P_2(\mathbf{u}^n)}
    \ 
    \frac{ \partial \mathcal P_2 }{ \partial \mathbf{u} }|_{\mathbf{u}^n}
$,

which is just the vector valued version of the "classic" chain rule
$f(g(x))' = f'(g(x)) g'(x)$, and directly extends for larger numbers of composited functions, i.e. $i>2$.

Here, the derivatives for $\mathcal P_1$ and $\mathcal P_2$ are still Jacobian matrices, but knowing that 
at the "end" of the chain we have our scalar loss (cf. {doc}`overview`), the right-most Jacobian will invariably
be a matrix with 1 column, i.e. a vector. During reverse mode, we start with this vector, and compute
the multiplications with the left Jacobians, $\frac{ \partial \mathcal P_1 }{ \partial \mathbf{u} }$ above,
one by one.

For the details of forward and reverse mode differentiation, please check out external materials such 
as this [nice survey by Baydin et al.](https://arxiv.org/pdf/1502.05767.pdf).

## Learning via DP Operators 

Thus, once the operators of our simulator support computations of the Jacobian-vector 
products, we can integrate them into DL pipelines just like you would include a regular fully-connected layer
or a ReLU activation.

At this point, the following (very valid) question often comes up: "_Most physics solver can be broken down into a
sequence of vector and matrix operations. All state-of-the-art DL frameworks support these, so why don't we just 
use these operators to realize our physics solver?_"

It's true that this would theoretically be possible. The problem here is that each of the vector and matrix
operations in tensorflow and pytorch is computed individually, and internally needs to store the current 
state of the forward evaluation for backpropagation (the "$g(x)$" above). For a typical 
simulation, however, we're not overly interested in every single intermediate result our solver produces.
Typically, we're more concerned with significant updates such as the step from $\mathbf{u}(t)$ to  $\mathbf{u}(t+\Delta t)$.

%provide discretized simulator of physical phenomenon as differentiable operator.
Thus, in practice it is a very good idea to break down the solving process into a sequence
of meaningful but _monolithic_ operators. This not only saves a lot of work by preventing the calculation
of unnecessary intermediate results, it also allows us to choose the best possible numerical methods 
to compute the updates (and derivatives) for these operators.
%in practice break down into larger, monolithic components
E.g., as this process is very similar to adjoint method optimizations, we can re-use many of the techniques
that were developed in this field, or leverage established numerical methods. E.g., 
we could leverage the $O(n)$ complexity of multigrid solvers for matrix inversion.

The flipside of this approach is, that it requires some understanding of the problem at hand, 
and of the numerical methods. Also, a given solver might not provide gradient calculations out of the box.
Thus, we want to employ DL for model equations that we don't have a proper grasp of, it might not be a good
idea to direclty go for learning via a DP approach. However, if we don't really understand our model, we probably
should go back to studying it a bit more anyway...

Also, in practice we can be _greedy_ with the derivative operators, and only 
provide those which are relevant for the learning task. E.g., if our network 
never produces the parameter $\nu$ in the example above, and it doesn't appear in our
loss formulation, we will never encounter a $\partial/\partial \nu$ derivative
in our backpropagation step.

---

## A practical example

As a simple example let's consider the advection of a passive scalar density $d(\mathbf{x},t)$ in
a velocity field $\mathbf{u}$ as physical model $\mathcal P^*$:

$$
  \frac{\partial d}{\partial{t}} + \mathbf{u} \cdot \nabla d = 0 
$$

Instead of using this formulation as a residual equation right away (as in {doc}`physicalloss`), 
we can discretize it with our favorite mesh and discretization scheme,
to obtain a formulation that updates the state of our system over time. This is a standard
procedure for a _forward_ solve.
Note that to simplify things, we assume that $\mathbf{u}$ is only a function in space,
i.e. constant over time. We'll bring back the time evolution of $\mathbf{u}$ later on.
%
Let's denote this re-formulation as $\mathcal P$. It maps a state of $d(t)$ into a 
new state at an evoled time, i.e.:

$$
    d(t+\Delta t) = \mathcal P ( ~ d(t), \mathbf{u}, t+\Delta t) 
$$

As a simple example of an optimization and learning task, let's consider the problem of
finding a motion $\mathbf{u}$ such that starting with a given initial state $d^{~0}$ at $t^0$,
the time evolved scalar density at time $t^e$ has a certain shape or configuration $d^{\text{target}}$.
Informally, we'd like to find a motion that deforms $d^{~0}$ into a target state.
The simplest way to express this goal is via an $L^2$ loss between the two states. So we want
to minimize the loss function $F=|d(t^e) - d^{\text{target}}|^2$. 

Note that as described here this is a pure optimization task, there's no NN involved,
and our goal is to obtain $\mathbf{u}$. We do not want to apply this motion to other, unseen _test data_,
as would be custom in a real learning task.

The final state of our marker density $d(t^e)$ is fully determined by the evolution 
of $\mathcal P$ via $\mathbf{u}$, which gives the following minimization problem as overall goal:

$$
    \text{arg min}_{~\mathbf{u}} | \mathcal P ( d^{~0}, \mathbf{u}, t^e - t^0 ) - d^{\text{target}}|^2
$$

We'd now like to find the minimizer for this objective by
_gradient descent_ (GD), where the 
gradient is determined by the differentiable physics approach described earlier in this chapter.
Once things are working with GD, we can relatively easily switch to better optimizers or bring
an NN into the picture, hence it's always a good starting point.

As the discretized velocity field $\mathbf{u}$ contains all our degrees of freedom,
what we need to update the velocity by an amount 
$\Delta \mathbf{u} = \partial L / \partial \mathbf{u}$, 
which can be decomposed into 
$\Delta \mathbf{u} = 
\frac{ \partial d }{ \partial \mathbf{u}}
\frac{ \partial L }{ \partial d}
$.
And as the evolution of $d$ is given by our discretized physical model $P$,
what we're acutally looking for is the Jacobian 
$\partial \mathcal P / \partial \mathbf{u}$ to
compute 
$\Delta \mathbf{u} = 
 \frac{ \partial \mathcal P }{ \partial \mathbf{u}}
 \frac{ \partial L }{ \partial d}$. 
We luckily don't need $\partial \mathcal P / \partial \mathbf{u}$ as a full
matrix, but instead only mulitplied by the vector obtained from the derivative of our scalar 
loss function $L$.

%the $L^2$ loss $L= |d(t^e) - d^{\text{target}}|^2$, thus

So what are the actual Jacobians here:
the one for $L$ is simple enough, we simply get a column vector with entries of the form
$2(d(t^e)_i - d^{\text{target}})_i$ for one component $i$.
$\partial \mathcal P / \partial \mathbf{u}$ is more interesting:
here we'll get derivatives of the chosen advection operator w.r.t. each component of the 
velocities.

%...to obtain an explicit update of the form $d(t+\Delta t) = A d(t)$, where the matrix $A$ represents the discretized advection step of size $\Delta t$ for $\mathbf{u}$. ... we'll get a matrix that essentially encodes linear interpolation coefficients for positions $\mathbf{x} + \Delta t \mathbf{u}$. For a grid of size $d_x \times d_y$ we'd have a 

```{figure} resources/placeholder.png
---
height: 100px
name: advection-upwing
---
TODO, small sketch of 1D advection
```

E.g., for a simple [first order upwinding scheme](https://en.wikipedia.org/wiki/Upwind_scheme) 
on a Cartesian grid in 1D, with marker density and velocity $d_i$ and $u_i$ for cell $i$
(superscripts for time $t$ are omitted for brevity), 
we get 

$$ \begin{aligned}
    & d_i^{~t+\Delta t} = d_i - u_i^+ (d_{i+1} - d_{i}) +  u_i^- (d_{i} - d_{i-1}) \text{ with }  \\
    & u_i^+ = \text{max}(u_i \Delta t / \Delta x,0) \\
    & u_i^- = \text{min}(u_i \Delta t / \Delta x,0)
\end{aligned} $$

E.g., for a positive $u_i$ we have 
$d_i^{~t+\Delta t} = (1 + \frac{u_i \Delta t }{ \Delta x}) d_i - \frac{u_i \Delta t }{ \Delta x} d_{i+1}$
and hence 
$\partial \mathcal P / \partial u_i$ from cell $i$ would be $1 + \frac{u_i \Delta t }{ \Delta x}$.
For the full gradient we'd need to add 
the potential contributions from cells $i+1$ and $i-1$, depending on the sign of their velocities.

In practice this step is similar to evaluating a transposed matrix multiplication.
If we rewrite the calculation above as 
$d^{~t+\Delta t} = A \mathbf{u}$, then $\partial \mathcal P / \partial \mathbf{u} = A^T$.
In many practical cases, a matrix free implementation of this multiplication might 
be preferable to actually constructing $A$.

## A (slightly) more complex example

**[TODO]**

a bit more complex, matrix inversion, eg Poisson solve
dont backprop through all CG steps (available in phiflow though)
rather, re-use linear solver to compute multiplication by inverse matrix

[note 1: essentialy yields implicit derivative, cf implicit function theorem & co]

[note 2: time can be "virtual" , solving for steady state
only assumption: some iterative procedure, not just single eplicit step - then things simplify.]

## Summary of Differentiable Physics so far

To summarize, using differentiable physical simulations 
gives us a tool to include phsyical equations with a chosen discretization into DL learning.
In contrast to the residual constraints of the previous chapter,
this makes it possible to left NNs seamlessly interact with physical solvers.

We'd previously fully discard our physical model and solver
once the NN is trained: in the example from {doc}`physicalloss-code` 
the NN gives us the solution directly, bypassing
any solver or model equation. With the DP approach we can train an NN alongside
a numerical solver, and thus we can make use of the physical model (as represented by 
the solver) later on at inference time. This allows us to move beyond solving single
inverse problems, and can yield NNs that quite robustly generalize to new inputs.

Let's revisit the sample problem from {doc}`physicalloss-code` in the context of DPs.
