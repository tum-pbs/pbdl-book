Complex Examples with DP
=======================

The following two sections with show code examples of two more complex cases that 
will show what can be achieved via differentiable physics training.

First, we'll show a scenario that employs deep learning to learn the errors
of a numerical simulation, following Um et al. {cite}`um2020sol`.
This is a very fundamental task, and requires the learned model to closely
interact with a numerical solver. Hence, it's a prime example of 
situations where it's crucial to bring the numerical solver into the 
deep learning loop.

Next, we'll show how to let NNs solve tough inverse problems, namely the long-term control
of a fluid simulation, following Holl et al.  {cite}`holl2019pdecontrol`. 
This task requires long term planning,
and hence needs two networks, one to _predict_ the evolution, 
and another one to _act_ to reach the desired goal. 

Both cases require quite a bit more resources than the previous examples, so you 
can expect these notebooks to run longer (and it's a good idea to use the check-pointing
mechanisms when working with these examples).

