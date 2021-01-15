Discussion of Physical Soft-Constraints
=======================

The good news so far is - we have a DL method that can include 
physical laws in the form of soft constraints by minimizing residuals.
However, as the very simple previous example illustrates, this is just a conceptual
starting point.

On the positive side, we can leverage DL frameworks with backpropagation to compute
the derivatives of the model. At the same time, this puts us at the mercy of the learned
representation regarding the reliability of these derivatives. Also, each derivative
requires backpropagation through the full network, which can be very slow. Especially so
for higher-order derivatives.

And while the setup is realtively simple, it is generally difficult to control. The NN
has flexibility to refine the solution by itself, but at the same time, tricks are necessary
when it doesn't pick the right regions of the solution.

## Is it "Machine Learning"

TODO, discuss - more akin to classical optimization:
we test for space/time positions at training time, and are interested in the  
solution there afterwards.

hence, no real generalization, or test data with different distribution.
more similar to inverse problem that solves single state e.g. via BFGS or Newton.

## Summary

In general, a fundamental drawback of this approach is that it does combine with traditional
numerical techniques well. E.g., learned representation is not suitable to be refined with 
a classical iterative solver such as the conjugate gradient method. This means many
powerful techniques that were developed in the past decades cannot be used in this context.
Bringing these numerical methods back into the picture will be one of the central
goals of the next sections.

✅ Pro: 
- uses physical model
- derivatives via backpropagation

❌ Con: 
- slow ...
- only soft constraints
- largely incompatible _classical_ numerical methods
- derivatives rely on learned representation

Next, let's look at how we can leverage numerical methods to improve the DL accuracy and efficiency
by making use of differentiable solvers.
