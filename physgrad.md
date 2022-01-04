Physical Gradients
=======================

**Note, this chapter is very preliminary - probably not for the first version of the book. move after RL, before BNNs?**

The next chapter will question some fundamental aspects of the formulations so far, namely the update step computed via gradients.
To re-cap, the approaches explained in the previous chapters either dealt with pure data, integrated the physical model as a physical loss term or included it via differentiable physics (DP) operators embedded into the network. 
Supervised training with physical data is straight-forward.
The latter two methods share similarities, but in the loss term case, the evaluations are only required at training time. For DP approaches, the solver itself is also employed at inference time, which enables an end-to-end training of NNs and numerical solvers. All three approaches employ _first-order_ derivatives to drive optimizations and learning processes, the latter two also using them for the physical model terms.
This is a natural choice from a deep learning perspective, but we haven't questioned at all whether this is actually a good choice.

Not too surprising after this introduction: A central insight of the following chapter will be that regular gradients are often a _sub-optimal choice_ for learning problems involving physical quantities.
It turns out that both supervised and DP gradients have their pros and cons. In the following, we'll analyze this in more detail. In particular, we'll illustrate how the multi-modal problems (as hinted at in {doc}`intro-teaser`) negatively influence NNs. Then we'll show how scaling problems of DP gradients affect NN training. Finally, we'll explain several alternatives to prevent these problems. It turns out that a key property that is missing in regular gradients is a proper _inversion_ of the Jacobian matrix.


```{admonition} A preview of this chapter
:class: tip

Below, we'll proceed in the following steps:
- We'll illustrate how the multi-modal problems (as hinted at in {doc}`intro-teaser`) negatively influence NNs
- Then we'll show how scaling problems of DP gradients affect NN training. 
- Finally we'll explain several alternatives to prevent these problems. 
- It turns out that a key property that is missing in regular gradients is a proper _inversion_ of the Jacobian matrix.

```

%- 2 remedies coming up:
%    1) Treating network and simulator as separate systems instead of a single black box, we'll derive different and improved update steps that replaces the gradient of the simulator. As this gradient is closely related to a regular gradient, but computed via physical model equations, we refer to this update (proposed by Holl et al. {cite}`holl2021pg`) as the {\em physical gradient} (PG).
%        [toolbox, but requires perfect inversion]
%    2) Treating them jointly, -> HIGs
%        [analytical, more practical approach]



XXX   PG physgrad chapter notes  from dec 23   XXX
- recap formulation P(x)=z , L() ... etc. rename z to y?
- intro after dL/dx bad, Newton? discussion is repetitive
[older commment - more intro to quasi newton?]
- GD - is "diff. phys." , rename? add supervised before?
comparison:
- why z, rename to y?
- add legends to plot
- summary "tighest possible" bad -> rather, illustrates what ideal direction can do


```{admonition} Looking ahead
:class: tip

Below, we'll proceed in the following steps:
- we'll first show the problems with regular gradient descent, especially for functions that combine small and large scales,
- a central insight will be that an _inverse gradient_ is a lot more meaningful than the regular one,
- finally, we'll show how to use inverse functions (and especially inverse PDE solvers) to compute a very accurate update that includes higher-order terms.

```


## Overview

All NNs of the previous chapters were trained with gradient descent (GD) via backpropagation, GD and hence backpropagation was also employed for the PDE solver (_simulator_) $\mathcal P$, computing the composite gradient 
$\partial L / \partial x$ for the loss function $L$:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \mathcal P(x)} \frac{\partial \mathcal P(x)}{\partial x}
$$ 

In the field of classical optimization, techniques such as Newton's method or BFGS variants are commonly used to optimize numerical processes since they can offer better convergence speed and stability.
These methods likewise employ gradient information, but substantially differ from GD in the way they 
compute the update step, typically via higher order derivatives.
% cite{nocedal2006numerical} 

The PG which we'll derive below can take into account nonlinearities to produce better optimization updates when an (full or approximate) inverse simulator is available.
In contrast to classic optimization techniques, we show how a differentiable or invertible physics 
simulator can be leveraged to compute the PG without requiring higher-order derivatives of the simulator.

In the following, we will stop using GD for everything, and instead use the aforementioned PGs for the simulator.
This update is combined with a GD based step for updating the weights in the NNs.
This setup, consisting of two fundamentally different optimization schemes, will result in an improved end-to-end training.


```{figure} resources/placeholder.png
---
height: 220px
name: pg-training
---
TODO, visual overview of PG training
```

![Divider](resources/divider3.jpg)


## Traditional optimization methods

We'll start by revisiting the most commonly used optization methods -- gradient descent (GD) and quasi-Newton methods -- and describe their fundamental limits and drawbacks on a theoretical level.

As before, let $L(x)$ be a scalar loss function, subject to minimization.
Note that we exclusively consider multivariate functions, and hence all symbols represent vector-valued expressions unless specified otherwise.


### Gradient descent

The optimization updates $\Delta x$ of GD scale with the derivative of the objective w.r.t. the inputs,

$$
    \Delta x = -\eta \cdot \frac{\partial L}{\partial x}
$$ (GD-update)

where $\eta$ is the scalar learning rate.
The Jacobian $\frac{\partial L}{\partial x}$ describes how the loss reacts to small changes of the input.
%
Surprisingly, this very widely used construction has a number of undesirable properties.

**Units** 📏

A first indicator that something is amiss with GD is that it inherently misrepresents dimensions.
Assume two parameters $x_1$ and $x_2$ have different physical units.
Then the GD parameter updates scale with the inverse of these units because the parameters appear in the denominator for the GD update above.
The learning rate $\eta$ could compensate for this discrepancy but since $x_1$ and $x_2$ have different units, there exists no single $\eta$ to produce the correct units for both parameters.

**Function sensitivity** 🔍

GD has inherent problems when functions are not normalized.
Assume the range of $L(x)$ lies on a different scale than $x$. 
Consider the function $L(x) = c \cdot x$ for example where $c \ll 1$ or $c \gg 1$.
Then the parameter updates of GD scale with $c$, i.e. $\Delta x = \eta \cdot c$.
Such behavior can occur easily in complex functions such as deep neural networks if the layers are not normalized correctly.
%
For sensitive functions, i.e. _small changes_ in $x$ cause **large** changes in $L$, GD counter-intuitively produces large $\Delta x$, causing even larger steps in $L$ (exploding gradients).
For insensitive functions where _large changes_ in the input don't change the output $L$ much, GD produces **small** updates, which can lead to the optimization coming to a standstill (that's the classic _vanishing gradients_ problem).
%
While normalization in combination with correct setting of the learning rate $\eta$ can be used to counteract this behavior in neural networks, these tools are not available when optimizing simulations.
Applying normalization to a simulation anywhere but after the last solver step would destroy the simulation.
Setting the learning rate is also difficult when simulation parameters at different time steps are optimized simultaneously or when the magnitude of the simulation output varies w.r.t. the initial state.

**Convergence near optimum** 💎

The loss landscape of any differentiable function necessarily becomes flat close to an optimum
(the gradient approaches zero upon convergence).
Therefore $\Delta x \rightarrow 0$ as the optimum is approached, resulting in slow convergence.

This is an important point, and we will revisit it below. It's also somewhat surprising at first, but it can actually
stabilize the training. On the other hand, it also makes the learning process difficult to control.




### Quasi-Newton methods

Quasi-Newton methods, such as BFGS and its variants, evaluate the gradient $\frac{\partial L}{\partial x}$ and Hessian $\frac{\partial^2 L}{\partial x^2}$ to solve a system of linear equations. The resulting update can be written as

$$
\Delta x = -\eta \cdot \left( \frac{\partial^2 L}{\partial x^2} \right)^{-1} \frac{\partial L}{\partial x}.
$$ (quasi-newton-update)

where $\eta$, the scalar step size, takes the place of GD's learning rate and is typically determined via a line search.
This construction solves some of the problems of gradient descent from above, but has other drawbacks.

**Units** 📏

Quasi-Newton methods definitely provide a much better handling of physical units than GD.
The quasi-Newton update from equation {eq}`quasi-newton-update`
produces the correct units for all parameters to be optimized, $\eta$ can stay dimensionless.

**Convergence near optimum** 💎

Quasi-Newton methods also exhibit much faster convergence when the loss landscape is relatively flat.
Instead of slowing down, they instead take larger steps, even when $\eta$ is fixed.
This is because the eigenvalues of the inverse Hessian scale inversely with the eigenvalues of the Hessian, increasing with the flatness of the loss landscape.


**Consistency in function compositions** 

So far, quasi-Newton methods address both shortcomings of GD. 
However, similar to GD, the update of an intermediate space still depends on all functions before that.
This behavior stems from the fact that the Hessian of a function composition carries non-linear terms of the gradient.

Consider a function composition $L(z(x))$, with $L$ as above, and an additional function $z(x)$.
Then the Hessian $\frac{d^2L}{dx^2} = \frac{\partial^2L}{\partial z^2} \left( \frac{\partial z}{\partial x} \right)^2 + \frac{\partial L}{\partial z} \cdot \frac{\partial^2 z}{\partial x^2}$ depends on the square of the inner gradient $\frac{\partial z}{\partial x}$. 
This means that the Hessian is influenced by the _later_ derivatives of a backpropagation pass, 
and as a consequence, the update of any latent space is unknown during the computation of the gradients.

% chain of function evaluations: Hessian of an outer function is influenced by inner ones; inversion corrects and yields quantity similar to IG, but nonetheless influenced by "later" derivatives


**Dependence on Hessian** 🎩

In addition, a fundamental disadvantage of quasi-Newton methods is their dependence on the Hessian of the full function.

The first obvious drawback is the _computational cost_.
While evaluating the exact Hessian only adds one extra pass to every optimization step, this pass involves higher-dimensional tensors than the computation of the gradient.
As $\frac{\partial^2 L}{\partial x^2}$ grows with the square of the parameter count, both its evaluation and its inversion become very expensive for large systems.
Many algorithms therefore avoid computing the exact Hessian and instead approximate it by accumulating the gradient over multiple update steps.
The memory requirements also grow quadratically.

The quasi-Newton update above additionally requires the _inverse_ Hessian matrix. Thus, a Hessian that is close to being non-invertible typically causes numerical stability problems, while inherently non-invertible Hessians require a fallback to a first order GD update.

Another related limitation of quasi-Newton methods is that the objective function needs to be _twice-differentiable_.
While this may not seem like a big restriction, note that many common neural network architectures use ReLU activation functions of which the second-order derivative is zero.
%
Related to this is the problem that higher-order derivatives tend to change more quickly when traversing the parameter space, making them more prone to high-frequency noise in the loss landscape.

```{note} 
_Quasi-Newton Methods_
are still a very active research topic, and hence many extensions have been proposed that can alleviate some of these problems in certain settings. E.g., the memory requirement problem can be sidestepped by storing only lower-dimensional vectors that can be used to approximate the Hessian. However, these difficulties illustrate the problems that often arise when applying methods like BFGS.
```

%\nt{In contrast to these classic algorithms, we will show how to leverage invertible physical models to efficiently compute physical update steps. In certain scenarios, such as simple loss functions, computing the inverse  gradient via the inverse Hessian will also provide a useful building block for our final algorithm.}
%, and how to they can be used to improve the training of neural networks.



![Divider](resources/divider4.jpg)


## Derivation of Physical Gradients

As a first step towards _physical_ gradients, we introduce _inverse_ gradients (IGs), 
which naturally solve many of the aforementioned problems.

Instead of $L$ (which is scalar), let's consider the function $z(x)$. We define the update

$$
    \Delta x = \frac{\partial x}{\partial z} \cdot \Delta z.
$$ (IG-def)

to be the IG update.
Here, the Jacobian $\frac{\partial x}{\partial z}$, which is similar to the inverse of the GD update above, encodes how the inputs must change in order to obtain a small change $\Delta z$ in the output.
%
The crucial step is the inversion, which of course requires the Jacobian matrix to be invertible (a drawback we'll get back to below). However, if we can invert it, this has some very nice properties.

Note that instead of using a learning rate, here the step size is determined by the desired increase or decrease of the value of the output, $\Delta z$.



% **Units**

**Positive Aspects** 

IGs scale with the inverse derivative. Hence the updates are automatically of the same units as the parameters without requiring an arbitrary learning rate: $\frac{\partial x}{\partial z}$ times $\Delta z$ has the units of $x$.

% **Function sensitivity**

They also don't have problems with normalization as the parameter updates from the example $L(x) = c \cdot x$ above now scale with $c^{-1}$.
Sensitive functions thus receive small updates while insensitive functions get large updates.

% **Convergence near optimum**

IGs show the opposite behavior of GD close to an optimum: they typically produce very accurate updates, which don't vanish near an optimum. This leads to fast convergence, as we will demonstrate in more detail below.

% **Consistency in function compositions**

Additionally, IGs are consistent in function composition.
The change in $x$ is $\Delta x = \Delta L \cdot \frac{\partial x}{\partial z} \frac{\partial z}{\partial L}$ and the approximate change in $z$ is $\Delta z = \Delta L  \cdot \frac{\partial z}{\partial x} \frac{\partial x}{\partial z} \frac{\partial z}{\partial L} = \Delta L \frac{\partial z}{\partial L}$.
% In the example in table~\ref{tab:function-composition-example}, the change $\Delta z$ is the same no matter what space is used as optimization target.
The change in intermediate spaces is independent of their respective dependencies, at least up to first order.
Consequently, the change to these spaces can be estimated during backpropagation, before all gradients have been computed.

Note that even Newton's method with its inverse Hessian didn't fully get this right. The key here is that if the Jacobian is invertible, we'll directly get the correctly scaled direction at a given layer, without "helpers" such as the inverse Hessian.

**Limitations**

So far so good.
The above properties make the advantages of IGs clear, but we're not done, unfortunately. There are strong limitations to their applicability.
%
The IG $\frac{\partial x}{\partial z}$ is only well-defined for square Jacobians, i.e. for functions $z$ with the same inputs and output dimensions.
In optimization, however, the input is typically high-dimensional while the output is a scalar objective function.
%
And, somewhat similar to the Hessians of quasi-Newton methods, 
even when the $\frac{\partial z}{\partial x}$ is square, it may not be invertible.

Thus, we now consider the fact that inverse gradients are linearizations of inverse functions and show that using inverse functions provides additional advantages while retaining the same benefits.



### Inverse simulators

Physical processes can be described as a trajectory in state space where each point represents one possible configuration of the system.
A simulator typically takes one such state space vector and computes a new one at another time.
The Jacobian of the simulator is, therefore, necessarily square.
%
As long as the physical process does _not destroy_ information, the Jacobian is non-singular.
In fact, it is believed that information in our universe cannot be destroyed so any physical process could in theory be inverted as long as we have perfect knowledge of the state.

While evaluating the IGs directly can be done through matrix inversion or taking the derivative of an inverse simulator, we now consider what happens if we use the inverse simulator directly in backpropagation.
Let $z = \mathcal P(x)$ be a forward simulation, and $\mathcal P(z)^{-1}=x$ its inverse (we assume it exists for now, but below we'll relax that assumption). 
Equipped with the inverse we now define an update that we'll call the **physical gradient** (PG) {cite}`holl2021pg` in the following as

% Original: \begin{equation} \label{eq:pg-def}  \frac{\Delta x}{\Delta z} \equiv \mathcal P_{(x_0,z_0)}^{-1} (z_0 + \Delta z) - x_0 = \frac{\partial x}{\partial z} + \mathcal O(\Delta z^2)

$%
    \frac{\Delta x}{\Delta z}   \equiv  \big( \mathcal P^{-1} (z_0 + \Delta z) - x_0  \big) / \Delta z
$% (PG-def)


**added $ / \Delta z $ on the right!? the above only gives $\Delta x$, see below**

Note that this PG is equal to the IG from the section above up to first order, but contains nonlinear terms, i.e.
$ \Delta x / \Delta z = \frac{\partial x}{\partial z} + \mathcal O(\Delta z^2) $.
%
The accuracy of the update also depends on the fidelity of the inverse function $\mathcal P^{-1}$.
We can define an upper limit to the error of the local inverse using the local gradient $\frac{\partial x}{\partial z}$.
In the worst case, we can therefore fall back to the regular gradient.

% We now show that these terms can help produce more stable updates than the IG alone, provided that $\mathcal P_{(x_0,z_0)}^{-1}$ is a sufficiently good approximation of the true inverse.
% Let $\mathcal P^{-1}(z)$ be the true inverse function to $\mathcal P(x)$, assuming that $\mathcal P$ is fully invertible.

The intuition for why the PG update is a good one is that when
applying the update $\Delta x = \mathcal P^{-1}(z_0 + \Delta z) - x_0$ it will produce $\mathcal P(x_0 + \Delta x) = z_0 + \Delta z$ exactly, despite $\mathcal P$ being a potentially highly nonlinear function.
When rewriting this update in the typical gradient format, $\frac{\Delta x}{\Delta z}$ replaces the gradient from the IG update above, and gives $\Delta x$.


**Fundamental theorem of calculus**

To more clearly illustrate the advantages in non-linear settings, we
apply the fundamental theorem of calculus to rewrite the ratio $\Delta x / \Delta z$ from above. This gives, 

% \begin{equation} \label{eq:avg-grad}

% $\begin{aligned}
%     \frac{\Delta z}{\Delta x} = \frac{\int_{x_0}^{x_0+\Delta x} \frac{\partial z}{\partial x} \, dx}{\Delta x}
% \end{aligned}$

% where we've integrated over a trajectory in $x$, and 
% focused on 1D for simplicity. Likewise, by integrating over $z$ we can obtain:

$\begin{aligned}
    \frac{\Delta x}{\Delta z} = \frac{\int_{z_0}^{z_0+\Delta z} \frac{\partial x}{\partial z} \, dz}{\Delta z}
\end{aligned}$

Here the expressions inside the integral is the local gradient, and we assume it exists at all points between $z_0$ and $z_0+\Delta z_0$.
The local gradients are averaged along the path connecting the state before the update with the state after the update.
The whole expression is therefore equal to the average gradient of $\mathcal P$ between the current $x$ and the estimate for the next optimization step $x_0 + \Delta x$.
This effectively amounts to _smoothing the objective landscape_ of an optimization by computing updates that can take nonlinearities of $\mathcal P$ into account.

The equations naturally generalize to higher dimensions by replacing the integral with a path integral along any differentiable path connecting $x_0$ and $x_0 + \Delta x$ and replacing the local gradient by the local gradient in the direction of the path.


![Divider](resources/divider5.jpg)



### Global and local inverse functions

Let $\mathcal P$ be a function with a square Jacobian and $z = \mathcal P(x)$.
A global inverse function $\mathcal P^{-1}$ is defined only for bijective $\mathcal P$. 
If the inverse exists, it can find $x$ for any $z$ such that $z = \mathcal P(x)$.

Instead of using this "perfect" inverse $\mathcal P^{-1}$ directly, we'll in practice often use a local inverse
$\mathcal P_{(x_0,z_0)}^{-1}(z)$, defined at the point $(x_0, z_0)$. This local inverse can be 
easier to obtain, as it only needs to exist near a given $z_0$, and not for all $z$. 
For $\mathcal P^{-1}$ to exist $\mathcal P$ would need to be globally invertible.

By contrast, a \emph{local inverse}, defined at point $(x_0, z_0)$, only needs to be accurate in the vicinity of that point.
If a global inverse $\mathcal P^{-1}(z)$ exists, the local inverse approximates it and matches it exactly as $z \rightarrow z_0$.
More formally, $\lim_{z \rightarrow z_0} \frac{\mathcal P^{-1}_{(x_0, z_0)}(z) - P^{-1}(z)}{|z - z_0|} = 0$.
Local inverse functions can exist, even when a global inverse does not.
Non-injective functions can be inverted, for example, by choosing the closest $x$ to $x_0$ such that $\mathcal P(x) = z$.

With the local inverse, the PG is defined as 

$$
    \frac{\Delta x}{\Delta z}   \equiv  \big( \mathcal P_{(x_0,z_0)}^{-1} (z_0 + \Delta z) - x_0  \big) / \Delta z
$$ (local-PG-def)

For differentiable functions, a local inverse is guaranteed to exist by the inverse function theorem as long as the Jacobian is non-singular.
That is because the inverse Jacobian $\frac{\partial x}{\partial z}$ itself is a local inverse function, albeit not the most accurate one.
Even when the Jacobian is singular (because the function is not injective, chaotic or noisy), we can usually find good local inverse functions.


---


## Summary

The update obtained with a regular gradient descent method has surprising shortcomings.
The physical gradient instead allows us to more accurately backpropagate through nonlinear functions, provided that we have access to good inverse functions.

Before moving on to including PGs in NN training processes, the next example will illustrate the differences between these approaches with a practical example.

