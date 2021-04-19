Introduction to Differentiable Physics
=======================

As a next step towards a tighter and more generic combination of deep learning
methods and deep learning we will target incorporating _differentiable physical 
simulations_ into the learning process. In the following, we'll shorten
that to "differentiable physics" (DP).

The central goal of this methods is to use existing numerical solvers, and equip
them with functionality to compute gradients with respect to their inputs.
Once this is realized for all operators of a simulation, we can leverage 
the autodiff functionality of DL frameworks with backpropagation to let gradient 
information from from a simulator into an NN and vice versa. This has numerous 
advantages such as improved learning feedback and generalization, as we'll outline below.

In contrast to physics-informed loss functions, it also enables handling more complex
solution manifolds instead of single inverse problems. Thus instead of using deep learning
to solve single inverse problems, we'll show how to train ANNs that solve 
larger classes of inverse problems very quickly.

```{figure} resources/diffphys-shortened.jpg
---
height: 220px
name: diffphys-short-overview
---
Training with differentiable physics mean that one or more differentiable operators
provide directions to steer the learning process.
```

## Differentiable operators

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
naturally transfers to including $\nu$ as a degree of freedom.

## Jacobians

As $\mathbf{u}$ is typically a vector-valued function, $\partial \mathcal P_i / \partial \mathbf{u}$ denotes
a Jacobian matrix $J$ rather than a single value:
% 
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
%
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

## Learning via DP operators 

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

Thus, in practice it is a very good idea to break down the solving process into a sequence
of meaningful but _monolithic_ operators. This not only saves a lot of work by preventing the calculation
of unnecessary intermediate results, it also allows us to choose the best possible numerical methods 
to compute the updates (and derivatives) for these operators.
E.g., as this process is very similar to adjoint method optimizations, we can re-use many of the techniques
that were developed in this field, or leverage established numerical methods. E.g., 
we could leverage the $O(n)$ runtime of multigrid solvers for matrix inversion.

The flipside of this approach is, that it requires some understanding of the problem at hand, 
and of the numerical methods. Also, a given solver might not provide gradient calculations out of the box.
Thus, we want to employ DL for model equations that we don't have a proper grasp of, it might not be a good
idea to directly go for learning via a DP approach. However, if we don't really understand our model, we probably
should go back to studying it a bit more anyway...

Also, in practice we can be _greedy_ with the derivative operators, and only 
provide those which are relevant for the learning task. E.g., if our network 
never produces the parameter $\nu$ in the example above, and it doesn't appear in our
loss formulation, we will never encounter a $\partial/\partial \nu$ derivative
in our backpropagation step.

The following figure summarizes the DP-based learning approach, and illustrates the sequence of operations that are typically processed within a single PDE solve. As many of the operations are non-linear in practice, this often leads to a challenging learning task for the NN:

```{figure} resources/diffphys-overview.jpg
---
height: 220px
name: diffphys-full-overview
---
DP learning with a PDE solver that consists of $m$ individual operators $\mathcal P_i$. The gradient travels backward through all $m$ operators before influencing the network weights $\theta$.
```


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
new state at an evolved time, i.e.:

$$
    d(t+\Delta t) = \mathcal P ( ~ d(t), \mathbf{u}, t+\Delta t) 
$$

As a simple example of an optimization and learning task, let's consider the problem of
finding a motion $\mathbf{u}$ such that starting with a given initial state $d^{~0}$ at $t^0$,
the time evolved scalar density at time $t^e$ has a certain shape or configuration $d^{\text{target}}$.
Informally, we'd like to find a motion that deforms $d^{~0}$ into a target state.
The simplest way to express this goal is via an $L^2$ loss between the two states. So we want
to minimize the loss function $L=|d(t^e) - d^{\text{target}}|^2$. 

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


E.g., for a simple [first order upwinding scheme](https://en.wikipedia.org/wiki/Upwind_scheme) 
on a Cartesian grid in 1D, with marker density and velocity $d_i$ and $u_i$ for cell $i$
(superscripts for time $t$ are omitted for brevity), 
we get 

$$ \begin{aligned}
    & d_i^{~t+\Delta t} = d_i - u_i^+ (d_{i+1} - d_{i}) +  u_i^- (d_{i} - d_{i-1}) \text{ with }  \\
    & u_i^+ = \text{max}(u_i \Delta t / \Delta x,0) \\
    & u_i^- = \text{min}(u_i \Delta t / \Delta x,0)
\end{aligned} $$

```{figure} resources/diffphys-advect1d.jpg
---
height: 150px
name: advection-upwind
---
1st-order upwinding uses a simple one-sided finite-difference stencil that takes into account the direction of the motion
```

Thus, for a positive $u_i$ we have 
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

As a slightly more complex example let's consider Poisson's equation $\nabla^2 a = b$, where
$a$ is the quantity of interest, and $b$ is given. 
This is a very fundamental elliptic PDE that is important for 
a variety of physical problems, from electrostatics to gravitational fields. It also arises 
in the context of fluids, where $a$ takes the role of a scalar pressure field in the fluid, and
the right hand side $b$ is given by the divergence of the fluid velocity $\mathbf{u}$.

For fluids, we typically have 
$\mathbf{u}^{n} = \mathbf{u} - \nabla p$, with
$\nabla^2 p = \nabla \cdot \mathbf{u}$. Here, $\mathbf{u}^{n}$ denotes the _new_, divergence-free
velocity field. This step is typically crucial to enforce the hard-constraint $\nabla \cdot \mathbf{u}=0$,
and is often called _Chorin Projection_, or _Helmholtz decomposition_. It is also closely related to the fundamental theorem of vector calculus.

If we now introduce an NN that modifies $\mathbf{u}$ in an iterative solver, we inevitably have to
backpropagate through the Poisson solve. I.e., we need a gradient for $\mathbf{u}^{n}$, which in this
notation takes the form $\partial \mathbf{u}^{n} / \partial \mathbf{u}$.

In combination, $\partial \mathbf{u}^{n} = \mathbf{u} - \nabla \left(  (\nabla^2)^{-1} \nabla \cdot \mathbf{u} \right)$. The outer gradient (from $\nabla p$) and the inner divergence ($\nabla \cdot \mathbf{u}$) are both linear operators, and their gradients simple to compute. The main difficulty lies in obtaining the
matrix inverse $(\nabla^2)^{-1}$ from the Poisson's equation for pressure (we'll keep it a bit simpler here, but it's often time-dependent, and non-linear). 

In practice, the matrix vector product for $(\nabla^2)^{-1} b$ with $b=\nabla \cdot \mathbf{u}$ is not explicitly computed via matrix operations, but approximated with a (potentially matrix-free) iterative solver. E.g., conjugate gradient (CG) methods are a very popular choice here. Thus, we could treat this iterative solver as a function $S$,
with $p = S(\nabla \cdot \mathbf{u})$. Note that matrix inversion is a non-linear process, despite the matrix itself being linear. As solvers like CG are also based on matrix and vector operations, we could decompose $S$ into a sequence of simpler operations $S(x) = S_n( S_{n-1}(...S_{1}(x)))$, and backpropagate through each of them. This is certainly possible, but not a good idea: it can introduce numerical problems, and can be very slow.
By default DL frameworks store the internal states for every differentiable operator like the $S_i()$ in this example, and hence we'd organize and keep $n$ intermediate states in memory. These states are completely uninteresting for our original PDE, though. They're just intermediate states of the CG solver.

If we take a step back and look at $p = (\nabla^2)^{-1} b$, it's gradient $\partial p / \partial b$
is just $((\nabla^2)^{-1})^T$. And in this case, $(\nabla^2)$ is a symmetric matrix, and so $((\nabla^2)^{-1})^T=(\nabla^2)^{-1}$. This is the identical inverse matrix that we encountered in the original equation above, and hence we can re-use our unmodified iterative solver to compute the gradient. We don't need to take it apart and slow it down by storing intermediate states. However, the iterative solver computes the matrix-vector-products for $(\nabla^2)^{-1} b$. So what is $b$ during backpropagation? In an optimization setting we'll always have our loss function $L$ at the end of the forward chain. The backpropagation step will then give a gradient for the output, let's assume it is $\partial L/\partial p$ here, which needs to be propagated to the earlier operations of the forward pass. Thus, we can simply invoke our iterative solve during the backward pass to compute $\partial p / \partial b = S(\partial L/\partial p)$. And assuming that we've chosen a good solver as $S$ for the forward pass, we get exactly the same performance and accuracy in the backwards pass.

If you're interested in a code example, the [differentiate-pressure example]( https://github.com/tum-pbs/PhiFlow/blob/master/demos/differentiate_pressure.py) of phiflow uses exactly this process for an optimization through a pressure projection step: a flow field that is constrained on the right side, is optimized for the content on the left, such that it matches the target on the right after a pressure projection step.

The main take-away here is: it is important _not to blindly backpropagate_ through the forward computation, but to think about which steps of the analytic equations for the forward pass to compute gradients for. In cases like the above, we can often find improved analytic expressions for the gradients, which we can then compute numerically.

```{admonition} Implicit Function Theorem & Time
:class: tip

**IFT**:
The process above essentially yields an _implicit derivative_. Instead of explicitly deriving all forward steps, we've relied on the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem) to compute the derivative.

**Time**: we _can_ actually consider the steps of an iterative solver as a virtual "time",
and backpropagate through these steps. In line with other DP approaches, this enabled an NN to _interact_ with an iterative solver. An example is to learn initial guesses of CG solvers from {cite}`um2020sol`, 
[details can be found here](https://github.com/tum-pbs/CG-Solver-in-the-Loop).
```


---

## Summary of differentiable physics so far

To summarize, using differentiable physical simulations 
gives us a tool to include physical equations with a chosen discretization into DL learning.
In contrast to the residual constraints of the previous chapter,
this makes it possible to left NNs seamlessly interact with physical solvers.

We'd previously fully discard our physical model and solver
once the NN is trained: in the example from {doc}`physicalloss-code` 
the NN gives us the solution directly, bypassing
any solver or model equation. With the DP approach we can train an NN alongside
a numerical solver, and thus we can make use of the physical model (as represented by 
the solver) later on at inference time. This allows us to move beyond solving single
inverse problems, and can yield NNs that quite robustly generalize to new inputs.
Let's revisit this sample problem in the context of DPs.
