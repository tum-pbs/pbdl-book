Discussion
=======================

The training via differentiable physics as described so far allows us
to integrate full numerical simulations and the training of deep neural networks
interacting with these simulations. While we've only hinted at what could be
achieved via DP approaches it is nonetheless a good time to summarize the pros and cons.


## Alternatives - Noise

It is worth mentioning here that other works have proposed perturbing the inputs and 
the iterations at training time with noise {cite}`sanchez2020learning` (somewhat similar to
regularizers like dropout). 
This can help to prevent overfitting to the training states, and hence shares similarities
with the goals of training with DP. 

However, the noise is typically undirected, and hence not as accurate as training with 
the actual evolutions of simulations. Hence, this noise can be a good starting point 
for training that tends to overfit, but if possible, it is preferable to incorporate the
actual solver in the training loop via a DP approach.


## Summary

To summarize the pros and cons of training NNs via differentiable physics:

✅ Pro: 
- Uses physical model and numerical methods for discretization.
- Efficiency of selected methods carries over to training.
- Tight coupling of physical models and NNs possible.

❌ Con: 
- Not compatible with all simulators (need to provide gradients).
- Require more heavy machinery (in terms of framework support) than previously discussed methods.

Especially the last point is one that is bound to strongly improve in a fairly short time, but for now it's important to keep in mind that not every simulator is suitable for DP training out of the box. Hence, in this book we'll focus on examples using phiflow, which was designed for interfacing with deep learning frameworks. 
Next we can target more some complex scenarios to showcase what can be achieved with differentiable physics.

