Discussion of Physical Losses
=======================

The good news so far is - we have a DL method that can include 
physical laws in the form of soft constraints by minimizing residuals.
However, as the very simple previous example illustrates, this causes
new difficulties, and is just a conceptual starting point.

On the positive side, we can leverage DL frameworks with backpropagation to compute
the derivatives of the model. At the same time, this makes the loss landscape more
complicated, relies on the  learned
representation regarding the reliability of the derivatives. Also, each derivative
requires backpropagation through the full network. This can be very expensive, especially 
for higher-order derivatives.

And while the setup is relatively simple, it is generally difficult to control. The NN
has flexibility to refine the solution by itself, but at the same time, tricks are necessary
when it doesn't focus on the right regions of the solution.

## Generalization?


One aspect to note with the previous PINN optimization is that the
positions where we test and constrain the solution are the final positions we are interested in.
As such, from a classic ML standpoint, there is no real distinction between training, validation and test sets.
Computing the solution for a known and given set of samples is much more akin to classical optimization,
where inverse problems like the previous Burgers example stem from.

For machine learning, we typically work under the assumption that the final performance of our 
model will be evaluated on a different, potentially unknown set of inputs. The _test data_
should usually capture such _out of distribution_ (OOD) behavior, so that we can make estimates
about how well our model will generalize to "real-world" cases that we will encounter when 
we deploy it in an application. The v1 version, using a prescribed discretization actually 
had this property, and could generalized to new inputs.

In contrast, for the PINN training as described here, we reconstruct a single solution in a known 
and given space-time region. As such, any samples from this domain follow the same distribution
and hence don't really represent test or OOD samples. As the NN directly encodes the solution,
there is also little hope that it will yield different solutions, or perform well outside
of the training range. If we're interested in a different solution, we 
have to start training the NN from scratch.

![Divider](resources/divider5.jpg)

## Summary

Thus, the physical soft constraints allow us to encode solutions to 
PDEs with the tools of NNs. As they're more widely used, we'll focus on PINNs (v2) here:
An inherent drawback is that they typically yield single solutions or very narrow solution manifolds,
and that they do not combine with traditional numerical techniques well. 
In comparison to the Neural surrogates/operators from {doc}`supervised` we've made a step backwards in some way.

E.g., the learned representation is not suitable to be refined with 
a classical iterative solver such as the conjugate gradient method. 
This means many
powerful techniques that were developed in the past decades cannot be used in this context.
Bringing these numerical methods back into the picture will be one of the central
goals of the next sections.

✅ Pro: 
- Uses physical model
- Derivatives can be conveniently computed via backpropagation

❌ Con: 
- Problematic convergence
- Physical constraints are enforced only as soft constraints
- Largely incompatible with _classical_ numerical methods
- Usefulness of derivatives relies on learned representation

To address these issues,
we'll next look at how we can leverage existing numerical methods to improve the DL process
by making use of differentiable solvers.

