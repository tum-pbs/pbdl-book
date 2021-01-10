Physical Loss Terms
=======================


Using the equations now, but no numerical methods!

Still interesting, leverages analytic derivatives of NNs, but lots of problems



---


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


Some notation from SoL, move with parts from overview into "appendix"?



We typically solve a discretized PDE $\mathcal{P}$ by performing discrete time steps of size $\Delta t$. 
Each subsequent step can depend on any number of previous steps,
$\mathbf{u}(\mathbf{x},t+\Delta t) = \mathcal{P}(\mathbf{u}(\mathbf{x},t), \mathbf{u}(\mathbf{x},t-\Delta t),...)$, 
where
$\mathbf{x} \in \Omega \subseteq \mathbb{R}^d$ for the domain $\Omega$ in $d$
dimensions, and $t \in \mathbb{R}^{+}$.

Numerical methods yield approximations of a smooth function such as $\mathbf{u}$ in a discrete
setting and invariably introduce errors. These errors can be measured in terms
of the deviation from the exact analytical solution.
For discrete simulations of
PDEs, these errors are typically expressed as a function of the truncation, $O(\Delta t^k)$ 
for a given step size $\Delta t$ and an exponent $k$ that is discretization dependent.

The following PDEs typically work with a continuous
velocity field $\mathbf{u}$ with $d$ dimensions and components, i.e.,
$\mathbf{u}(\mathbf{x},t): \mathbb{R}^d \rightarrow \mathbb{R}^d $.
For discretized versions below, $d_{i,j}$ will denote the dimensionality
of a field such as the velocity,
with domain size $d_{x},d_{y},d_{z}$ for source and reference in 3D.

% with $i \in \{s,r\}$ denoting source/inference manifold and reference manifold, respectively.
%This yields $\vc{} \in \mathbb{R}^{d \times d_{s,x} \times d_{s,y} \times d_{s,z} }$ and $\vr{} \in \mathbb{R}^{d \times d_{r,x} \times d_{r,y} \times d_{r,z} }$
%Typically, $d_{r,i} > d_{s,i}$ and $d_{z}=1$ for $d=2$.

For all PDEs, we use non-dimensional parametrizations as outlined below,
and the components of the velocity vector are typically denoted by $x,y,z$ subscripts, i.e.,
$\mathbf{u} = (u_x,u_y,u_z)^T$ for $d=3$.

Burgers' equation in 2D. It represents a well-studied advection-diffusion PDE:

$\frac{\partial u_x}{\partial{t}} + \mathbf{u} \cdot \nabla u_x =
  \nu \nabla\cdot \nabla u_x + g_x(t), 
  \\
  \frac{\partial u_y}{\partial{t}} + \mathbf{u} \cdot \nabla u_y =
  \nu \nabla\cdot \nabla u_y + g_y(t)
$, 

where $\nu$ and $\mathbf{g}$ denote diffusion constant and external forces, respectively.

Burgers' equation in 1D without forces with $u_x = u$:
%\begin{eqnarray}
$\frac{\partial u}{\partial{t}} + u \nabla u = \nu \nabla \cdot \nabla u $ .

---

Later on, additional equations...



Navier-Stokes, in 2D:

$
    \frac{\partial u_x}{\partial{t}} + \mathbf{u} \cdot \nabla u_x =
    - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla u_x  
    \\
    \frac{\partial u_y}{\partial{t}} + \mathbf{u} \cdot \nabla u_y =
    - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla u_y  
    \\
    \text{subject to} \quad \nabla \cdot \mathbf{u} = 0
$



Navier-Stokes, in 2D with Boussinesq:

%$\frac{\partial u_x}{\partial{t}} + \mathbf{u} \cdot \nabla u_x$
%$ -\frac{1}{\rho} \nabla p $

$
  \frac{\partial u_x}{\partial{t}} + \mathbf{u} \cdot \nabla u_x = - \frac{1}{\rho} \nabla p 
  \\
  \frac{\partial u_y}{\partial{t}} + \mathbf{u} \cdot \nabla u_y = - \frac{1}{\rho} \nabla p + \eta d
  \\
  \text{subject to} \quad \nabla \cdot \mathbf{u} = 0,
  \\
  \frac{\partial d}{\partial{t}} + \mathbf{u} \cdot \nabla d = 0 
$



Navier-Stokes, in 3D:

$
  \frac{\partial u_x}{\partial{t}} + \mathbf{u} \cdot \nabla u_x = - \frac{1}{\rho} \nabla p + \nu \nabla\cdot \nabla u_x 
  \\
  \frac{\partial u_y}{\partial{t}} + \mathbf{u} \cdot \nabla u_y = - \frac{1}{\rho} \nabla p + \nu \nabla\cdot \nabla u_y 
  \\
  \frac{\partial u_z}{\partial{t}} + \mathbf{u} \cdot \nabla u_z = - \frac{1}{\rho} \nabla p + \nu \nabla\cdot \nabla u_z 
  \\
  \text{subject to} \quad \nabla \cdot \mathbf{u} = 0.
$
