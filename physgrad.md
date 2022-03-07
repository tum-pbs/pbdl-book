Towards Gradient Inversion
=======================

**Note, this chapter is very preliminary - to be finalized**

The next chapter will question some fundamental aspects of the formulations so far, namely the update step computed via gradients.
To re-cap, the approaches explained in the previous chapters either dealt with purely _supervised_ training, integrated the physical model as a _physical loss term_ or included it via _differentiable physics_ (DP) operators embedded into the training graph. 
For supervised training with data from physical simulations standard procedure apply.
The latter two methods are more relevant in the context of this book. They share similarities, but in the loss term case, the physics evaluations are only required at training time. For DP approaches, the solver itself is usually also employed at inference time, which enables an end-to-end training of NNs and numerical solvers. All three approaches employ _first-order_ derivatives to drive optimizations and learning processes, and the latter two also using them for the physics terms.
This is a natural choice from a deep learning perspective, but we haven't questioned at all whether this is actually a good choice.

Not too surprising after this introduction: A central insight of the following chapter will be that regular gradients are often a _sub-optimal choice_ for learning problems involving physical quantities.
It turns out that both supervised and DP gradients have their pros and cons. In the following, we'll analyze this in more detail. In particular, 
we'll show how scaling problems of DP gradients affect NN training. 
Then, we'll also illustrate how multi-modal problems (as hinted at in {doc}`intro-teaser`) negatively influence NNs. 
Finally, we'll explain several alternatives to prevent these problems. It turns out that a key property that is missing in regular gradients is a proper _inversion_ of the Jacobian matrix.


```{admonition} A preview of this chapter
:class: tip

Below, we'll proceed in the following steps:
- Show how scaling issues and multi-modality can negatively affect NNs.
- Explain two alternatives to prevent these problems.
- Preview: What was missing in our training runs with GD or Adam so far is a proper _inversion_ of the Jacobian matrix.

```

%- 2 remedies coming up:
%    1) Treating network and simulator as separate systems instead of a single black box, we'll derive different and improved update steps that replaces the gradient of the simulator. As this gradient is closely related to a regular gradient, but computed via physical model equations, we refer to this update (proposed by Holl et al. {cite}`holl2021pg`) as the {\em physical gradient} (PG).
%        [toolbox, but requires perfect inversion]
%    2) Treating them jointly, -> HIGs
%        [analytical, more practical approach]


XXX   PG physgrad chapter notes  from dec 23   XXX
-  GD - is "diff. phys." , rename? add supervised before?
- comparison notebook: add legends to plot
- summary "tightest possible" bad -> rather, illustrates what ideal direction can do
- double check func-comp w QNewton, "later" derivatives of backprop means what?
- remove IGs? 


%```{admonition} Looking ahead
%:class: tip
%Below, we'll proceed in the following steps:
%- we'll first show the problems with regular gradient descent, especially for functions that combine small and large scales,
%- a central insight will be that an _inverse gradient_ is a lot more meaningful than the regular one,
%- finally, we'll show how to use inverse functions (and especially inverse PDE solvers) to compute a very accurate update that includes higher-order terms.
%```


![Divider](resources/divider3.jpg)


## Overview - Traditional optimization methods

As before, let $L(x)$ be a scalar loss function, subject to minimization. The goal is to compute a step in terms of the input parameters $x$ , denoted by $\Delta x$. The different versions of $\Delta x$ will be denoted by a subscript.

All NNs of the previous chapters were trained with gradient descent (GD) via backpropagation. GD with backpropagation was also employed for the PDE solver (_simulator_) $\mathcal P$. As a central quantitiy, this gives the composite gradient 
$(\partial L / \partial x)^T$ of the loss function $L$:

$$
\Big( \frac{\partial L}{\partial x} \Big)^T = 
    \Big( \frac{\partial \mathcal P(x)}{\partial x} \Big)^T
    \Big( \frac{\partial L}{\partial \mathcal P(x)} \Big)^T
$$ (loss-deriv)

We've shown that using $\partial L/\partial x$ works, but
in the field of classical optimization, other algorithms are more widely used than GD: popular are so-called quasi-Newton methods, and they use different updates.
Hence in the following we'll revisit GD, and discuss the pros and cons of the different methods on a theoretical level. Among others, it's interesting to discuss why classical optimization algorithms aren't widely used for NN training despite having some obvious advantages.

Note that we exclusively consider multivariate functions, and hence all symbols represent vector-valued expressions unless noted otherwise.

%techniques such as Newton's method or BFGS variants are commonly used to optimize numerical processes since they can offer better convergence speed and stability. These methods likewise employ gradient information, but substantially differ from GD in the way they compute the update step, typically via higher order derivatives.

%```{figure} resources/placeholder.png
%---
%height: 220px
%name: pg-training
%---
%TODO, visual overview of PG training
%```




### Gradient descent

The optimization updates $\Delta x_{\text{GD}}$ of GD scale with the derivative of the objective w.r.t. the inputs,

$$
    \Delta x_{\text{GD}} = -\eta \cdot \frac{\partial L}{\partial x}
$$ (GD-update)

where $\eta$ is the scalar learning rate.
The Jacobian $\frac{\partial L}{\partial x}$ describes how the loss reacts to small changes of the input.
Surprisingly, this very widely used update has a number of undesirable properties that we'll highlight in the following. Note that we've naturally applied this update in supervised settings such as {doc}`supervised-airfoils`, but we've also used it in the differentiable physics approaches. E.g., in {doc}`diffphys-code-sol` we've computed the derivative of the fluid solver. In the latter case, we've still only updated the NN parameters, but the fluid solver Jacobian was part of {eq}`GD-update`, as shown in {eq}`loss-deriv`.


**Units** 📏

A first indicator that something is amiss with GD is that it inherently misrepresents dimensions.
Assume two parameters $x_1$ and $x_2$ have different physical units.
Then the GD parameter updates scale with the inverse of these units because the parameters appear in the denominator for the GD update above ($\cdots / \partial x$).
The learning rate $\eta$ could compensate for this discrepancy but since $x_1$ and $x_2$ have different units, there exists no single $\eta$ to produce the correct units for both parameters.

One could argue that units aren't very important for the parameters of NNs, but nonetheless it's unnerving from a physics perspective that they're wrong.

**Function sensitivity** 🔍

GD has also inherent problems when functions are not _normalized_.
This can be illustrated with a very simple example:
consider the function $L(x) = c \cdot x$.
Then the parameter updates of GD scale with $c$, i.e. $\Delta x_{\text{GD}} = \eta \cdot c$, and 
$L(x+\Delta x_{\text{GD}})$ will even have terms on the order of $c^2$.
If $L$ is normalized via $c=1$, everything's fine. But in practice, we'll often
have $c \ll 1$, or even worse $c \gg 1$, and then our optimization will be in trouble.

More specifically, if we look at how the loss changes, the expansion around $x$ for
the update step of GD gives:
$L(x+\Delta x_{\text{GD}}) = L(x)  + \Delta x_{\text{GD}} \frac{\partial L}{\partial x}  + \cdots $.
This first-order step causes a change in the loss of
$\big( L(x) - L(x+\Delta x_{\text{GD}}) \big) = -\eta \cdot (\frac{\partial L}{\partial x})^2 + \mathcal O(\Delta x^2)$. Hence the loss changes by the squared derivative, which leads to the $c^2$ factor mentioned above. Even worse, in practice we'd like to have a normalized quantity here. For a scaling of the gradients by $c$, we'd like our optimizer to compute a quantity like $1/c^2$, in order to get a reliable update from the gradient. 

This demonstrates that
for sensitive functions, i.e. functions where _small changes_ in $x$ cause _large_ changes in $L$, GD counter-intuitively produces large $\Delta x_{\text{GD}}$. This causes even larger steps in $L$, and leads to exploding gradients.
For insensitive functions where _large changes_ in the input don't change the output $L$ much, GD produces _small_ updates, which can lead to the optimization coming to a halt. That's the classic _vanishing gradients_ problem.

Such sensitivity problems can occur easily in complex functions such as deep neural networks where the layers are typically not fully normalized.
Normalization in combination with correct setting of the learning rate $\eta$ can be used to counteract this behavior in NNs to some extent, but these tools are not available when optimizing physics simulations.
Applying normalization to a simulation anywhere but after the last solver step would destroy the state of the simulation.
Adjusting the learning rate is also difficult in practice, e.g. when simulation parameters at different time steps are optimized simultaneously or when the magnitude of the simulation output varies w.r.t. the initial state.


**Convergence near optimum** 💎

Finally, the loss landscape of any differentiable function necessarily becomes flat close to an optimum,
as the gradient approaches zero upon convergence.
Therefore $\Delta x_{\text{GD}} \rightarrow 0$ as the optimum is approached, resulting in slow convergence.

This is an important point, and we will revisit it below. It's also somewhat surprising at first, but it can actually
stabilize the training. On the other hand, it makes the learning process difficult to control.




### Quasi-Newton methods

Quasi-Newton methods, such as BFGS and its variants, evaluate the gradient $\frac{\partial L}{\partial x}$ and Hessian $\frac{\partial^2 L}{\partial x^2}$ to solve a system of linear equations. The resulting update can be written as

$$
\Delta x_{\text{QN}} = -\eta \cdot \left( \frac{\partial^2 L}{\partial x^2} \right)^{-1} \frac{\partial L}{\partial x}.
$$ (quasi-newton-update)

where $\eta$, the scalar step size, takes the place of GD's learning rate. As a further improvement, it is typically determined via a line search in many Quasi-Newton methods.
This construction solves some of the problems of gradient descent from above, but has other drawbacks.

**Units and Sensitivity** 📏

Quasi-Newton methods definitely provide a much better handling of physical units than GD.
The quasi-Newton update from equation {eq}`quasi-newton-update`
produces the correct units for all parameters to be optimized. 
As a consequence, $\eta$ can stay dimensionless.

If we now consider how the loss changes via
$L(x+\Delta x_{\text{QN}}) = L(x) + -\eta \cdot \left( \frac{\partial^2 L}{\partial x^2} \right)^{-1} \frac{\partial L}{\partial x} \frac{\partial L}{\partial x}  + \cdots $ , the second term correctly cancels out the $x$ quantities, and leaves us with a scalar update in terms of $L$. Thinking back to the example with a scaling factor $c$ from the GD section, the inverse Hessian in Newton's methods successfully gives us a factor of $1/c^2$ to couteract the undesirable scaling of our updates.

**Convergence near optimum** 💎

Quasi-Newton methods also exhibit much faster convergence when the loss landscape is relatively flat.
Instead of slowing down, they take larger steps, even when $\eta$ is fixed.
This is thanks to the eigenvalues of the inverse Hessian, which scale inversely with the eigenvalues of the Hessian, and hence increase with the flatness of the loss landscape.


**Consistency in function compositions** 

So far, quasi-Newton methods address both shortcomings of GD. 
However, similar to GD, the update of an intermediate space still depends on all functions before that.
This behavior stems from the fact that the Hessian of a composite function carries non-linear terms of the gradient.

Consider a function composition $L(y(x))$, with $L$ as above, and an additional function $y(x)$.
Then the Hessian $\frac{d^2L}{dx^2} = \frac{\partial^2L}{\partial y^2} \left( \frac{\partial y}{\partial x} \right)^2 + \frac{\partial L}{\partial y} \cdot \frac{\partial^2 y}{\partial x^2}$ depends on the square of the inner gradient $\frac{\partial y}{\partial x}$. 
This means that the Hessian is influenced by the _later_ derivatives of a backpropagation pass, 
and as a consequence, the update of any latent space is unknown during the computation of the gradients.

% chain of function evaluations: Hessian of an outer function is influenced by inner ones; inversion corrects and yields quantity similar to IG, but nonetheless influenced by "later" derivatives


**Dependence on Hessian** 🎩

In addition, a fundamental disadvantage of quasi-Newton methods that becomes apparent from the discussion above is their dependence on the Hessian. It plays a crucial role for all the improvements discussed so far.

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

As a first step towards _physical_ gradients, we'll consider what we'll call _inverse_ gradients (IGs).
They already solve many of the aforementioned problems. Unfortunately, they come with their own set of problems, which is why they only represent an intermediate step (we'll revisit them in a more pracitcal form later on).

Instead of $L$ (which is scalar), let's consider a general, potentially non-scalar function $y(x)$. 
This will typically be the physical simulator later on, but to keep things general we'll call it $y$ for now.
We define the update

$$
    \Delta x_{\text{IG}} = \frac{\partial x}{\partial y} \cdot \Delta y.
$$ (IG-def)

to be the IG update.
Here, the Jacobian $\frac{\partial x}{\partial y}$, which is similar to the inverse of the GD update above, encodes how the inputs must change in order to obtain a small change $\Delta y$ in the output.
%
The crucial step is the inversion, which of course requires the Jacobian matrix to be invertible. This is a problem somewhat similar to the inversion of the Hessian, and we'll revisit this issue below. However, if we can invert the Jacobian, this has some very nice properties.

Note that instead of using a learning rate, here the step size is determined by the desired increase or decrease of the value of the output, $\Delta y$. Thus, we need to choose a $\Delta y$ instead of an $\eta$. This $\Delta y$ will show up frequently in the following equations, and make them look quite different to the ones above at first sight. Effectively, $\Delta y$ plays the same role as the learning rate, i.e., it controls the step size of the optimization.



% **Units**

**Positive Aspects** 

IGs scale with the inverse derivative. Hence the updates are automatically of the same units as the parameters without requiring an arbitrary learning rate: $\frac{\partial x}{\partial y}$ times $\Delta y$ has the units of $x$.

% **Function sensitivity**

They also don't have problems with normalization as the parameter updates from the example $L(x) = c \cdot x$ above now scale with $c^{-1}$.
Sensitive functions thus receive small updates while insensitive functions get large updates.

% **Convergence near optimum**

IGs show the opposite behavior of GD close to an optimum: they typically produce very accurate updates, which don't vanish near an optimum. This leads to fast convergence, as we will demonstrate in more detail below.

% **Consistency in function compositions**

Additionally, IGs are consistent in function composition.
The change in $x$ is $\Delta x_{\text{IG}} = \Delta L \cdot \frac{\partial x}{\partial y} \frac{\partial y}{\partial L}$ and the approximate change in $y$ is $\Delta y = \Delta L  \cdot \frac{\partial y}{\partial x} \frac{\partial x}{\partial y} \frac{\partial y}{\partial L} = \Delta L \frac{\partial y}{\partial L}$.
% In the example in table~\ref{tab:function-composition-example}, the change $\Delta y$ is the same no matter what space is used as optimization target.
The change in intermediate spaces is independent of their respective dependencies, at least up to first order.
Consequently, the change to these spaces can be estimated during backpropagation, before all gradients have been computed.

Note that even Newton's method with its inverse Hessian didn't fully get this right. The key here is that if the Jacobian is invertible, we'll directly get the correctly scaled direction at a given layer, without helper quantities such as the inverse Hessian.

**Limitations**

So far so good.
The above properties make the advantages of IGs clear, but we're not done, unfortunately. There are strong limitations to their applicability.
The inverse of the Jacobian, $\frac{\partial x}{\partial y}$, is only well-defined for square Jacobians, meaning for functions $y$ with the same inputs and output dimensions.
In optimization, however, the input is typically high-dimensional while the output is a scalar objective function.
And, somewhat similar to the Hessians of quasi-Newton methods, 
even when the $\frac{\partial y}{\partial x}$ is square, it may not be invertible.

Thus, we now consider the fact that inverse gradients are linearizations of inverse functions and show that using inverse functions provides additional advantages while retaining the same benefits.



### Inverse simulators

Physical processes can be described as a trajectory in state space where each point represents one possible configuration of the system.
A simulator typically takes one such state space vector and computes a new one at another time.
The Jacobian of the simulator is, therefore, necessarily square.
%
As long as the physical process does _not destroy_ information, the Jacobian is non-singular.
In fact, it is believed that information in our universe cannot be destroyed so any physical process could in theory be inverted as long as we have perfect knowledge of the state.

??? While evaluating the IGs directly can be done through matrix inversion or taking the derivative of an inverse simulator,  ???

We now consider a somewhat theoretical construct: what can we do if we have access to an 

what happens if we use the inverse simulator directly in backpropagation.
Let $y = \mathcal P(x)$ be a forward simulation, and $\mathcal P(y)^{-1}=x$ its inverse (we assume it exists for now, but below we'll relax that assumption). 
Equipped with the inverse we now define an update that we'll call the **physical gradient** (PG) {cite}`holl2021pg` in the following as

$$
    \frac{\Delta x_{\text{PG}} }{\Delta y}   \equiv  \big( \mathcal P^{-1} (y_0 + \Delta y) - x_0  \big) 
$$ (PG-def)


% Original: \begin{equation} \label{eq:pg-def}  \frac{\Delta x}{\Delta z} \equiv \mathcal P_{(x_0,z_0)}^{-1} (z_0 + \Delta z) - x_0 = \frac{\partial x}{\partial z} + \mathcal O(\Delta z^2)

% add ? $ / \Delta z $ on the right!? the above only gives $\Delta x$, see below

Note that this PG is equal to the IG from the section above up to first order, but contains nonlinear terms, i.e.
$ \Delta x_{\text{PG}} / \Delta y = \frac{\partial x}{\partial y} + \mathcal O(\Delta y^2) $.
%
The accuracy of the update also depends on the fidelity of the inverse function $\mathcal P^{-1}$.
We can define an upper limit to the error of the local inverse using the local gradient $\frac{\partial x}{\partial y}$.
In the worst case, we can therefore fall back to the regular gradient.

% We now show that these terms can help produce more stable updates than the IG alone, provided that $\mathcal P_{(x_0,z_0)}^{-1}$ is a sufficiently good approximation of the true inverse.
% Let $\mathcal P^{-1}(z)$ be the true inverse function to $\mathcal P(x)$, assuming that $\mathcal P$ is fully invertible.

The intuition for why the PG update is a good one is that when
applying the update $\Delta x_{\text{PG}} = \mathcal P^{-1}(y_0 + \Delta y) - x_0$ it will produce $\mathcal P(x_0 + \Delta x) = y_0 + \Delta y$ exactly, despite $\mathcal P$ being a potentially highly nonlinear function.
When rewriting this update in the typical gradient format, $\frac{\Delta x_{\text{PG}}}{\Delta y}$ replaces the gradient from the IG update above, and gives $\Delta x$.


**Fundamental theorem of calculus**

To more clearly illustrate the advantages in non-linear settings, we
apply the fundamental theorem of calculus to rewrite the ratio $\Delta x_{\text{PG}} / \Delta y$ from above. This gives, 

% \begin{equation} \label{eq:avg-grad}

% $\begin{aligned}
%     \frac{\Delta z}{\Delta x} = \frac{\int_{x_0}^{x_0+\Delta x} \frac{\partial z}{\partial x} \, dx}{\Delta x}
% \end{aligned}$

% where we've integrated over a trajectory in $x$, and 
% focused on 1D for simplicity. Likewise, by integrating over $z$ we can obtain:

$\begin{aligned}
    \frac{\Delta x_{\text{PG}}}{\Delta y} = \frac{\int_{y_0}^{y_0+\Delta y} \frac{\partial x}{\partial y} \, dy}{\Delta y}
\end{aligned}$

Here the expressions inside the integral is the local gradient, and we assume it exists at all points between $y_0$ and $y_0+\Delta y_0$.
The local gradients are averaged along the path connecting the state before the update with the state after the update.
The whole expression is therefore equal to the average gradient of $\mathcal P$ between the current $x$ and the estimate for the next optimization step $x_0 + \Delta x_{\text{PG}}$.
This effectively amounts to _smoothing the objective landscape_ of an optimization by computing updates that can take nonlinearities of $\mathcal P$ into account.

The equations naturally generalize to higher dimensions by replacing the integral with a path integral along any differentiable path connecting $x_0$ and $x_0 + \Delta x_{\text{PG}}$ and replacing the local gradient by the local gradient in the direction of the path.


![Divider](resources/divider5.jpg)



### Global and local inverse functions

Let $\mathcal P$ be a function with a square Jacobian and $y = \mathcal P(x)$.
A global inverse function $\mathcal P^{-1}$ is defined only for bijective $\mathcal P$. 
If the inverse exists, it can find $x$ for any $y$ such that $y = \mathcal P(x)$.

Instead of using this "perfect" inverse $\mathcal P^{-1}$ directly, we'll in practice often use a local inverse
$\mathcal P_{(x_0,y_0)}^{-1}(y)$, defined at the point $(x_0, y_0)$. This local inverse can be 
easier to obtain, as it only needs to exist near a given $y_0$, and not for all $y$. 
For $\mathcal P^{-1}$ to exist $\mathcal P$ would need to be globally invertible.

By contrast, a \emph{local inverse}, defined at point $(x_0, y_0)$, only needs to be accurate in the vicinity of that point.
If a global inverse $\mathcal P^{-1}(y)$ exists, the local inverse approximates it and matches it exactly as $y \rightarrow y_0$.
More formally, $\lim_{y \rightarrow y_0} \frac{\mathcal P^{-1}_{(x_0, y_0)}(y) - P^{-1}(y)}{|y - y_0|} = 0$.
Local inverse functions can exist, even when a global inverse does not.
Non-injective functions can be inverted, for example, by choosing the closest $x$ to $x_0$ such that $\mathcal P(x) = y$.

With the local inverse, the PG is defined as 

$$
    \frac{\Delta x_{\text{PG}}}{\Delta y}   \equiv  \big( \mathcal P_{(x_0,y_0)}^{-1} (y_0 + \Delta y) - x_0  \big) / \Delta y
$$ (local-PG-def)

For differentiable functions, a local inverse is guaranteed to exist by the inverse function theorem as long as the Jacobian is non-singular.
That is because the inverse Jacobian $\frac{\partial x}{\partial y}$ itself is a local inverse function, albeit not the most accurate one.
Even when the Jacobian is singular (because the function is not injective, chaotic or noisy), we can usually find good local inverse functions.


---


## Summary

The update obtained with a regular gradient descent method has surprising shortcomings.
The physical gradient instead allows us to more accurately backpropagate through nonlinear functions, provided that we have access to good inverse functions.

Before moving on to including PGs in NN training processes, the next example will illustrate the differences between these approaches with a practical example.

