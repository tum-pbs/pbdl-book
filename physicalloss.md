Physical Loss Terms
=======================

Using the equations now, but no numerical methods!

Still interesting, leverages analytic derivatives of NNs, but lots of problems

---

Some notation from SoL:

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

Later on, Navier-Stokes, in 2D:

$
    \frac{\partial u_x}{\partial{t}} + \mathbf{u} \cdot \nabla u_x =
    - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla u_x  \\
    \frac{\partial u_y}{\partial{t}} + \mathbf{u} \cdot \nabla u_y =
    - \frac{1}{\rho}\nabla{p} + \nu \nabla\cdot \nabla u_y  \\
    \text{subject to} \quad \nabla \cdot \mathbf{u} = 0,
$


