Supervised Training
=======================

_Supervised_ here essentially means: "doing things the old fashioned way". Old fashioned in the context of 
deep learning (DL), of course, so it's still fairly new. Also, "old fashioned" of course also doesn't 
always mean bad - it's just that we'll be able to do better than simple supervised training later on. 

In a way, the viewpoint of "supervised training" is a starting point for all projects one would encounter in the context of DL, and
hence is worth studying. And although it typically yields inferior results to approaches that more tightly 
couple with physics, it nonetheless can be the only choice in certain application scenarios where no good
model equations exist.

## Problem Setting

For supervised training, we're faced with an 
unknown function $f^*(x)=y$, collect lots of pairs of data $[x_0,y_0], ...[x_n,y_n]$ (the training data set)
and directly train a NN to represent an approximation of $f^*$ denoted as $f$, such
that $f(x)=y$.

The $f$ we can obtain is typically not exact, 
but instead we obtain it via a minimization problem:
by adjusting weights $\theta$ of our representation with $f$ such that

$\text{arg min}_{\theta} \sum_i (f(x_i ; \theta)-y_i)^2$.

This will give us $\theta$ such that $f(x;\theta) \approx y$ as accurately as possible given
our choice of $f$ and the hyperparameters for training. Note that above we've assumed 
the simplest case of an $L^2$ loss. A more general version would use an error metric $e(x,y)$
to be minimized via $\text{arg min}_{\theta} \sum_i e( f(x_i ; \theta) , y_i) )$. The choice
of a suitable metric is topic we will get back to later on.

Irrespective of our choice of metric, this formulation
gives the actual "learning" process for a supervised approach.

The training data typically needs to be of substantial size, and hence it is attractive 
to use numerical simulations to produce a large number of training input-output pairs.
This means that the training process uses a set of model equations, and approximates
them numerically, in order to train the NN representation $\tilde{f}$. This
has a bunch of advantages, e.g., we don't have measurement noise of real-world devices
and we don't need manual labour to annotate a large number of samples to get training data.

On the other hand, this approach inherits the common challenges of replacing experiments
with simulations: first, we need to ensure the chosen model has enough power to predict the 
bheavior of real-world phenomena that we're interested in.
In addition, the numerical approximations have numerical errors
which need to be kept small enough for a chosen application. As these topics are studied in depth
for classical simulations, the existing knowledge can likewise be leveraged to
set up DL training tasks.

```{figure} resources/placeholder.png
---
height: 220px
name: supervised-training
---
TODO, visual overview of supervised training
```

## Show me some code!

Let's directly look at an implementation within a more complicated context:
_turbulent flows around airfoils_ from {cite}`thuerey2020deepFlowPred`.

