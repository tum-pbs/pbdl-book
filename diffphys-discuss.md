Discussion
=======================

The training via differentiable physics as described so far allows us
to integrate full numerical simulations and the training of deep neural networks
interacting with these simulations. While we've only hinted at what could be
achieved via DP approaches it is nonetheless a good time to summarize the pros and cons.

## Summary

✅ Pro: 
- uses physical model and numerical methods for discretization
- efficiency of selected methods carries over to training
- tight coupling of physical models and NNs possible

❌ Con: 
- not compatible with all simulators (need to provide gradients)
- require more heavy machinery (in terms of framework support) than previously discussed methods

Especially the last point is one that is bound to strongly improve in a fairly short time, but for now it's important to keep in mind that not every simulator is suitable for DP training out of the box. Hence, in this book we'll focus on examples using phiflow, which was designed for interfacing with deep learning frameworks. 
Next we can target more some complex scenarios to showcase what can be achieved with differentiable physics.

