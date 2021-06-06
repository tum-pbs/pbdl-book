Complex Examples Overview
=======================

The following sections will give code examples of more complex cases to 
show what can be achieved via differentiable physics training.

First, we'll show a scenario that employs deep learning to represent the errors
of numerical simulations, following Um et al. {cite}`um2020sol`.
This is a very fundamental task, and requires the learned model to closely
interact with a numerical solver. Hence, it's a prime example of 
situations where it's crucial to bring the numerical solver into the 
deep learning loop.

Next, we'll show how to let NNs solve tough inverse problems, namely the long-term control
of a Navier-Stokes simulation, following Holl et al.  {cite}`holl2019pdecontrol`. 
This task requires long term planning,
and hence needs two networks, one to _predict_ the evolution, 
and another one to _act_ to reach the desired goal. (Later on, in {doc}`reinflearn-code` we will compare
this approach to another DL variant using reinforcement learning.)

Both cases require quite a bit more resources than the previous examples, so you 
can expect these notebooks to run longer (and it's a good idea to use check-pointing
when working with these examples).
