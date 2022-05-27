Models and Equations
============================

Below we'll give a brief (really _very_ brief!) intro to deep learning, primarily to introduce the notation.
In addition we'll discuss some _model equations_ below. Note that we'll avoid using _model_ to denote trained neural networks, in contrast to some other texts and APIs. These will be called "NNs" or "networks". A "model" will typically denote a set of model equations for a physical effect, usually PDEs. 

## Deep learning and neural networks

In this book we focus on the connection with physical
models, and there are lots of great introductions to deep learning. 
Hence, we'll keep it short: 
the goal in deep learning is to approximate an unknown function

$$
f^*(x) = y^* , 
$$ (learn-base) 

where $y^*$ denotes reference or "ground truth" solutions.
$f^*(x)$ should be approximated with an NN representation $f(x;\theta)$. We typically determine $f$ 
with the help of some variant of a loss function $L(y,y^*)$, where $y=f(x;\theta)$ is the output
of the NN.
This gives a minimization problem to find $f(x;\theta)$ such that $e$ is minimized.
In the simplest case, we can use an $L^2$ error, giving

$$
\text{arg min}_{\theta} | f(x;\theta) - y^* |_2^2 . 
$$ (learn-l2)

We typically optimize, i.e. _train_, 
with a stochastic gradient descent (SGD) optimizer of choice, e.g. Adam {cite}`kingma2014adam`.
We'll rely on auto-diff to compute the gradient of a _scalar_ loss $L$ w.r.t. the weights, $\partial L / \partial \theta$.
It is crucial for the calculation of gradients that this function is scalar,
and the loss function is often also called "error", "cost", or "objective" function.

<!-- general goal, minimize E for e(x,y) ... cf. eq. 8.1 from DLbook 
introduce scalar loss, always(!) scalar...  (also called *cost* or *objective* function) -->

For training we distinguish: the **training** data set drawn from some distribution, 
the **validation** set (from the same distribution, but different data),
and **test** data sets with _some_ different distribution than the training one.
The latter distinction is important. For the test set we want 
_out of distribution_ (OOD) data to check how well our trained model generalizes.
Note that this gives a huge range of possibilities for the test data set: 
from tiny changes that will certainly work,
up to completely different inputs that are essentially guaranteed to fail. 
There's no gold standard, but test data should be generated with care.

Enough for now - if all the above wasn't totally obvious for you, we very strongly recommend to 
read chapters 6 to 9 of the [Deep Learning book](https://www.deeplearningbook.org),
especially the sections about [MLPs](https://www.deeplearningbook.org/contents/mlp.html) 
and "Conv-Nets", i.e. [CNNs](https://www.deeplearningbook.org/contents/convnets.html).

```{note} Classification vs Regression

The classic ML distinction between _classification_ and _regression_ problems is not so important here:
we only deal with _regression_ problems in the following.

```

<!--
maximum likelihood estimation
Also interesting: from a math standpoint ''just'' non-linear optimization ...
-->

## Partial differential equations as physical models

The following section will give a brief outlook for the model equations
we'll be using later on in the DL examples.
We typically target continuous PDEs denoted by $\mathcal P^*$
whose solution is of interest in a spatial domain $\Omega \subset \mathbb{R}^d$ in $d \in {1,2,3} $ dimensions.
In addition, we often consider a time evolution for a finite time interval $t \in \mathbb{R}^{+}$.
The corresponding fields are either d-dimensional vector fields, for instance $\mathbf{u}: \mathbb{R}^d \times \mathbb{R}^{+} \rightarrow \mathbb{R}^d$, 
or scalar $\mathbf{p}: \mathbb{R}^d \times \mathbb{R}^{+} \rightarrow \mathbb{R}$.
The components of a vector are typically denoted by $x,y,z$ subscripts, i.e.,
$\mathbf{v} = (v_x, v_y, v_z)^T$ for $d=3$, while
positions are denoted by $\mathbf{p} \in \Omega$.

To obtain unique solutions for $\mathcal P^*$ we need to specify suitable
initial conditions, typically for all quantities of interest at $t=0$,
and boundary conditions for the boundary of $\Omega$, denoted by $\Gamma$ in 
the following.

$\mathcal P^*$ denotes
a continuous formulation, where we make mild assumptions about
its continuity, we will typically assume that first and second derivatives exist.

We can then use numerical methods to obtain approximations 
of a smooth function such as $\mathcal P^*$ via discretization. 
These invariably introduce discretization errors, which we'd like to keep as small as possible.
These errors can be measured in terms of the deviation from the exact analytical solution, 
and for discrete simulations of PDEs, they are typically expressed as a function of the truncation error 
$O( \Delta x^k )$, where $\Delta x$ denotes the spatial step size of the discretization.
Likewise, we typically have a temporal discretization via a time step $\Delta t$.

```{admonition} Notation and abbreviations
:class: seealso
If unsure, please check the summary of our mathematical notation
and the abbreviations used in: {doc}`notation`.
```

% \newcommand{\pde}{\mathcal{P}}         % PDE ops
% \newcommand{\pdec}{\pde_{s}}
% \newcommand{\manifsrc}{\mathscr{S}}    % coarse / "source"
% \newcommand{\pder}{\pde_{R}}
% \newcommand{\manifref}{\mathscr{R}}

% vc - coarse solutions
% \renewcommand{\vc}[1]{\vs_{#1}}            % plain coarse state at time t
% \newcommand{\vcN}{\vs}                     % plain coarse state without time 
% vc - coarse solutions, modified by correction
% \newcommand{\vct}[1]{\tilde{\vs}_{#1}}     % modified / over time at time t
% \newcommand{\vctN}{\tilde{\vs}}            % modified / over time without time
% vr - fine/reference solutions
% \renewcommand{\vr}[1]{\mathbf{r}_{#1}}            % fine / reference state at time t , never modified
% \newcommand{\vrN}{\mathbf{r}}                     % plain coarse state without time 

% \newcommand{\project}{\mathcal{T}}           % transfer operator fine <> coarse
% \newcommand{\loss}{\mathcal{L}}              % generic loss function
% \newcommand{\nn}{f_{\theta}}
% \newcommand{\dt}{\Delta t}                   % timestep
% \newcommand{\corrPre}{\mathcal{C}_{\text{pre}}}            % analytic correction , "pre computed"
% \newcommand{\corr}{\mathcal{C}}                         % just C for now...
% \newcommand{\nnfunc}{F} % {\text{NN}}

% discretized versions below, $d_{i,j}$ will denote the dimensionality, domain size $d_{x},d_{y},d_{z}$ for source and reference in 3D.
% with $i \in \{s,r\}$ denoting source/inference manifold and reference manifold, respectively.
%This yields $\vc{} \in \mathbb{R}^{d \times d_{s,x} \times d_{s,y} \times d_{s,z} }$ and $\vr{} \in \mathbb{R}^{d \times d_{r,x} \times d_{r,y} \times d_{r,z} }$
%Typically, $d_{r,i} > d_{s,i}$ and $d_{z}=1$ for $d=2$.

We solve a discretized PDE $\mathcal{P}$ by performing steps of size $\Delta t$.
The solution can be expressed as a function of $\mathbf{u}$ and its derivatives:
$\mathbf{u}(\mathbf{x},t+\Delta t) = 
\mathcal{P}( \mathbf{u}_{x}, \mathbf{u}_{xx}, ... \mathbf{u}_{xx...x} )$, where
  $\mathbf{u}_{x}$ denotes the spatial derivatives $\partial \mathbf{u}(\mathbf{x},t) / \partial \mathbf{x}$.

For all PDEs, we will assume non-dimensional parametrizations as outlined below,
which could be re-scaled to real world quantities with suitable scaling factors.
Next, we'll give an overview of the model equations, before getting started
with actual simulations and implementation examples on the next page.

---

## Some example PDEs 

The following PDEs are good examples, and we'll use them later on in different settings to show how to incorporate them into DL approaches.

### Burgers

We'll often consider Burgers' equation 
in 1D or 2D as a starting point. 
It represents a well-studied PDE, which (unlike Navier-Stokes)
does not include any additional constraints such as conservation of mass. 
Hence, it leads to interesting shock formations.
It contains an advection term (motion / transport) and a diffusion term (dissipation due to the second law of thermodynamics).
In 2D, it is given by:

$$\begin{aligned}
  \frac{\partial u_x}{\partial{t}} + \mathbf{u} \cdot \nabla u_x &=
  \nu \nabla\cdot \nabla u_x + g_x, 
  \\
  \frac{\partial u_y}{\partial{t}} + \mathbf{u} \cdot \nabla u_y &=
  \nu \nabla\cdot \nabla u_y + g_y \ ,
\end{aligned}$$ (model-burgers2d)

where $\nu$ and $\mathbf{g}$ denote diffusion constant and external forces, respectively.

A simpler variant of Burgers' equation in 1D without forces, 
denoting the single 1D velocity component as $u = u_x$,
is given by:
%\begin{eqnarray}

$$
\frac{\partial u}{\partial{t}} + u \nabla u = \nu \nabla \cdot \nabla u \ . 
$$ (model-burgers1d)

### Navier-Stokes

A good next step in terms of complexity is given by the
Navier-Stokes equations, which are a well-established model for fluids.
In addition to an equation for the conservation of momentum (similar to Burgers),
they include an equation for the conservation of mass. This prevents the 
formation of shock waves, but introduces a new challenge for numerical methods
in the form of a hard-constraint for divergence free motions.

In 2D, the Navier-Stokes equations without any external forces can be written as:

$$\begin{aligned}
    \frac{\partial u_x}{\partial{t}} + \mathbf{u} \cdot \nabla u_x &=
    - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla u_x  
    \\
    \frac{\partial u_y}{\partial{t}} + \mathbf{u} \cdot \nabla u_y &=
    - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla u_y  
    \\
    \text{subject to} \quad \nabla \cdot \mathbf{u} &= 0
\end{aligned}$$ (model-ns2d)

where, like before, $\nu$ denotes a diffusion constant for viscosity.

An interesting variant is obtained by including the 
[Boussinesq approximation](https://en.wikipedia.org/wiki/Boussinesq_approximation_(buoyancy))
for varying densities, e.g., for simple temperature changes of the fluid.
With a marker field $v$ that indicates regions of high temperature,
it yields the following set of equations:

$$\begin{aligned}
  \frac{\partial u_x}{\partial{t}} + \mathbf{u} \cdot \nabla u_x &= - \frac{1}{\rho} \nabla p 
  \\
  \frac{\partial u_y}{\partial{t}} + \mathbf{u} \cdot \nabla u_y &= - \frac{1}{\rho} \nabla p + \xi v
  \\
  \text{subject to} \quad \nabla \cdot \mathbf{u} &= 0,
  \\
  \frac{\partial v}{\partial{t}} + \mathbf{u} \cdot \nabla v &= 0 
\end{aligned}$$ (model-boussinesq2d)

where $\xi$ denotes the strength of the buoyancy force.

And finally, the Navier-Stokes model in 3D give the following set of equations:

$$
\begin{aligned}
  \frac{\partial u_x}{\partial{t}} + \mathbf{u} \cdot \nabla u_x &= - \frac{1}{\rho} \nabla p + \nu \nabla\cdot \nabla u_x 
  \\
  \frac{\partial u_y}{\partial{t}} + \mathbf{u} \cdot \nabla u_y &= - \frac{1}{\rho} \nabla p + \nu \nabla\cdot \nabla u_y 
  \\
  \frac{\partial u_z}{\partial{t}} + \mathbf{u} \cdot \nabla u_z &= - \frac{1}{\rho} \nabla p + \nu \nabla\cdot \nabla u_z 
  \\
  \text{subject to} \quad \nabla \cdot \mathbf{u} &= 0.
\end{aligned}
$$ (model-ns3d)

## Forward Simulations

Before we really start with learning methods, it's important to cover the most basic variant of using the above model equations: a regular "forward" simulation, that starts from a set of initial conditions, and evolves the state of the system over time with a discretized version of the model equation. We'll show how to run such forward simulations for Burgers' equation in 1D and for a 2D Navier-Stokes simulation.
