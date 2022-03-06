Integrating DP into NN Training
=======================

% ## Time steps and iterations

We'll now target integrations of differentiable physics (DP) setups into NNs.
When using DP approaches for learning applications, 
there is a lot of flexibility w.r.t. the combination of DP and NN building blocks. 
As some of the differences are subtle, the following section will go into more detail.
We'll especially focus on solvers that repeat the PDE and NN evaluations multiple times,
e.g., to compute multiple states of the physical system over time.

To re-cap, here's the previous figure about combining NNs and DP operators. 
In the figure these operators look like a loss term: they typically don't have weights,
and only provide a gradient that influences the optimization of the NN weights:

```{figure} resources/diffphys-shortened.jpg
---
height: 220px
name: diffphys-short
---
The DP approach as described in the previous chapters. A network produces an input to a PDE solver $\mathcal P$, which provides a gradient for training during the backpropagation step.
```

This setup can be seen as the network receiving information about how it's output influences the outcome of the PDE solver. I.e., the gradient will provide information how to produce an NN output that minimizes the loss. 
Similar to the previously described _physical losses_ (from {doc}`physicalloss`), this can mean upholding a conservation law.

**Switching the Order** 

However, with DP, there's no real reason to be limited to this setup. E.g., we could imagine a swap of the NN and DP components, giving the following structure:

```{figure} resources/diffphys-switched.jpg
---
height: 220px
name: diffphys-switch
---
A PDE solver produces an output which is processed by an NN.
```

In this case the PDE solver essentially represents an _on-the-fly_ data generator. That's not necessarily always useful: this setup could be replaced by a pre-computation of the same inputs, as the PDE solver is not influenced by the NN. Hence, we could replace the $\mathcal P$ invocations by a "loading" function. On the other hand, evaluating the PDE solver at training time with a randomized sampling of the parameter domain of interest can lead to an excellent sampling of the data distribution of the input, and improve the NN training. If done correctly, the solver can also alleviate the need to store and load large amounts of data, and instead produce them more quickly at training time, e.g., directly on a GPU.

**Time Stepping** 

In general, there's no combination of NN layers and DP operators that is _forbidden_ (as long as their dimensions are compatible). One that makes particular sense is to "unroll" the iterations of a time stepping process of a simulator, and let the state of a system be influenced by an NN.

In this case we compute a (potentially very long) sequence of PDE solver steps in the forward pass. In-between these solver steps, an NN modifies the state of our system, which is then used to compute the next PDE solver step. During the backpropagation pass, we move backwards through all of these steps to evaluate contributions to the loss function (it can be evaluated in one or more places anywhere in the execution chain), and to backprop the gradient information through the DP and NN operators. This unrolling of solver iterations essentially gives feedback to the NN about how it's "actions" influence the state of the physical system and resulting loss. Due to the iterative nature of this process, many errors start out very small, and then slowly increase exponentially over the course of iterations. Hence they are extremely difficult to detect in a single evaluation, e.g., from a simple supervised training setup. In these cases it is crucial to provide feedback to the NN at training time who the errors evolve over course of the iterations. Additionally, a pre-computation of the states is not possible for such iterative cases, as the iterations depend on the state of the NN. Naturally, the NN state is unknown before training time and changes while being trained. Hence, a DP-based training is crucial to provide the NN with gradients about how it influences the solver iterations.

```{figure} resources/diffphys-multistep.jpg
---
height: 180px
name: diffphys-mulitstep
---
Time stepping with interleaved DP and NN operations for $k$ solver iterations. The dashed gray arrows indicate optional intermediate evaluations of loss terms (similar to the solid gray arrow for the last step $k$).
```

Note that in this picture (and the ones before) we have assumed an _additive_ influence of the NN. Of course, any differentiable operator could be used here to integrate the NN result into the state of the PDE. E.g., multiplicative modifications can be more suitable in certain settings, or in others the NN could modify the parameters of the PDE in addition to or instead of the state space. Likewise, the loss function is problem dependent and can be computed in different ways.

DP setups with many time steps can be difficult to train: the gradients need to backpropagate through the full chain of PDE solver evaluations and NN evaluations. Typically, each of them represents a non-linear and complex function. Hence for larger numbers of steps, the vanishing and exploding gradient problem can make training difficult (see {doc}`diffphys-code-sol` for some practical tips how to alleviate this).

![Divider](resources/divider4.jpg)


## Alternatives: noise

It is worth mentioning here that other works have proposed perturbing the inputs and 
the iterations at training time with noise {cite}`sanchez2020learning` (somewhat similar to
regularizers like dropout). 
This can help to prevent overfitting to the training states, and hence shares similarities
with the goals of training with DP. 

However, the noise is typically undirected, and hence not as accurate as training with 
the actual evolutions of simulations. Hence, this noise can be a good starting point 
for training setups that tend to overfit. However, if possible, it is preferable to incorporate the
actual solver in th
e training loop via a DP approach.

---

## Complex Examples

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
