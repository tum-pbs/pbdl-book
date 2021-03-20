Physical Gradients 
=======================

Re-cap?

...


## Training via Physical Gradients


**TODO, add details ...  fomr chap5 pdf**

The discussion above already hints at PGs being a powerful tool for optimization. However, as it stands, they're restricted to functions with square Jacobians. Hence we can't directly use them in optimizations or learning problems, which typically have scalar objective functions.
In this section, we will first show how PGs can be integrated into the optimization pipeline to optimize scalar objectives.

Consider a scalar objective function $L(z)$ that depends on the result of an invertible simulator $z = \mathcal P(x)$.
Applying the chain rule and substituting the IG for the PG, the update becomes

$\begin{aligned}
    \Delta x
    &= \frac{\partial x}{\partial L} \cdot \Delta L
    \\
    &= \frac{\partial x}{\partial z} \left( \frac{\partial z}{\partial L} \cdot \Delta L \right)
    \\
    &= \frac{\partial x}{\partial z} \cdot \Delta z
    \\
    &= \mathcal P^{-1}_{(x_0,z_0)}(z_0 + \Delta z) - x_0 + \mathcal O(\Delta z^2)
    .
\end{aligned}
$

This equation does not prescribe a unique way to compute $\Delta z$ since the derivative $\frac{\partial z}{\partial L}$ as the right-inverse of the row-vector $\frac{\partial L}{\partial z}$ puts almost no restrictions on $\Delta z$.
Instead, we use equation {eq}`quasi-newton-update` to determine $\Delta z$ where $\eta$ controls the step size of the optimization steps.
Unlike with quasi-Newton methods, where the Hessian of the full system is required, here, the Hessian is needed only for $L(z)$ and its computation can be completely forgone in many cases.

Consider the case $L(z) = \frac 1 2 || z^\textrm{predicted} - z^\textrm{target}||_2^2$ which is the most common supervised objective function.
Here $\frac{\partial L}{\partial z} = z^\textrm{predicted} - z^\textrm{target}$ and $\frac{\partial^2 L}{\partial z^2} = 1$.
Using equation {eq}`quasi-newton-update`, we get $\Delta z = \eta \cdot (z^\textrm{target} - z^\textrm{predicted})$ which can be computed without evaluating the Hessian.

Once $\Delta z$ is determined, the gradient can be backpropagated to earlier time steps using the inverse simulator.


