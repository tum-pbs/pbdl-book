
Half-Inverse Gradients
=======================

The physical gradients (PGs) illustrated the importance of _inverting_ the direction of the update step (in addition to making use of higher order terms). We'll now turn to an alternative for achieving the inversion, the _Half-Inverse Gradients_ (HIGs) {cite}`schnell2022hig`. They comes with its own set of pros and cons, and thus provide an interesting alternative for computing improved update steps for physics-based deep learning tasks.

More specifically, unlike the PGs, it does not require an analytical inverse solver and it jointly inverts the neural network part as well as the physical model. As a drawback, it requires an SVD for a large Jacobian matrix. 


```{admonition} Preview: HIGs versus PGs
:class: tip

More specifically, unlike the PGs the HIGs
- do not require an analytical inverse solver 
- they also jointly invert the neural network part as well as the physical model. 

As a drawback, HIGs 
- require an SVD for a large Jacobian matrix
- They are based on first-order information, like regular gradients. 

In contrast to regular gradients, they use the full Jacobian matrix, though. So as we'll see below, they typically outperform regular SGD and Adam significantly.

```

## Derivation

As mentioned during the derivation of PGs in {eq}`quasi-newton-update`, the update for regular Newton steps 
uses the inverse Hessian matrix. If we rewrite its update for the network weights $\theta$, and neglect the mixed derivative terms, we arrive at the _Gauss-Newton_ method:

% \Delta \theta_{GN} = -\eta {\partial x}.
$$
     \Delta \theta_{\mathrm{GN}}
    = - \eta \Bigg( \bigg(\frac{\partial z}{\partial \theta}\bigg)^{T} \cdot \bigg(\frac{\partial z}{\partial \theta}\bigg) \Bigg)^{-1} \cdot
    \bigg(\frac{\partial z}{\partial \theta}\bigg)^{T} \cdot \bigg(\frac{\partial L}{\partial z}\bigg)^{\top} .
$$ (gauss-newton-update-full)

For a full-rank Jacobian $\partial z / \partial \theta$, the transposed Jacobian cancels out, and the equation simplifies to

$$
     \Delta \theta_{\mathrm{GN}}
    = - \eta \bigg(\frac{\partial z}{\partial \theta}\bigg)  ^{-1} \cdot
        \bigg(\frac{\partial L}{\partial z}\bigg)^{\top} .
$$ (gauss-newton-update)

This looks much simpler, but still leaves us with a Jacobian matrix to invert. This Jacobian is typically non-square, and has small eigenvalues, which is why even equipped with a pseudo-inverse Gauss-Newton methods are not used for practical deep learning problems. 

HIGs alleviate these difficulties by employing a partial inversion of the form

$$
    \Delta \theta_{\mathrm{HIG}} = - \eta \cdot  \bigg(\frac{\partial y}{\partial \theta}\bigg)^{-1/2} \cdot \bigg(\frac{\partial L}{\partial y}\bigg)^{\top} , 
$$ (hig-update)

where the square-root for $^{-1/2}$ is computed via an SVD, and denotes the half-inverse. I.e., for a matrix $A$, 
we compute its half-inverse via a singular value decomposition as $A^{-1/2} = V \Lambda^{-1/2} U^\top$, where $\Lambda$ contains the singular values.
During this step we can also take care of numerical noise in the form of small singular values. All entries
of $\Lambda$ smaller than a threshold $\tau$ are set to zero.

```{note} Truncation

It might seem attractive at first to clamp singular values to a small value $\tau$, instead of discarding them by setting them to zero. However, the singular vectors corresponding to these small singular values are exactly the ones which are potentially unreliable. A small $\tau$ yields a large contribution during the inversion, and thus these singular vectors would cause problems when clamping. Hence, it's a much better idea to discard their content by setting their singular values to zero.

```

explain batch stacking

$$
    \frac{\partial y}{\partial \theta} := \left(
    \begin{array}{c}
    \frac{\partial y_1}{\partial \theta}\big\vert_{x_1}\\
    \frac{\partial y_2}{\partial \theta}\big\vert_{x_2}\\
    \vdots\\
    \frac{\partial y_b}{\partial \theta}\big\vert_{x_b}\\
    \end{array}
    \right)
$$

% 

background? motivated by Adam, ...

%We've kept the $\eta$ in here for consistency, but in practice $\eta=1$ is used for Gauss-Newton


%PGs higher order, custom inverse , chain PDE & NN together

%HIG more generic, numerical inversion , joint physics & NN


