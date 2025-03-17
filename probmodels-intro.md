Introduction to Probabilistic Learning
=======================

So far we've treated the target function $f(x)=y$ as being deterministic, with a unique solution $y$ for every input. That's certainly a massive simplification: in practice, solutions can be ambiguous, our learned model might mix things up, and both effects could show up in combination. This all calls for moving towards a probabilistic setting, which we'll address here. The machinery from previous sections will come in handy, as the probabilistic viewpoint essentially introduces another dimension for the problem. Instead of a single $y$, we now have a multitude of solutions drawn from a distribution $Y$, each with a probability $p_Y(y)$, often shortened to $p(y)$.
Samples $y \sim p(y)$ drawn from the distribution should follow this probability, so that we can distinguish rare and and frequent cases. 

To summarize, instead of individual solutions $y$ we're facing a large number of samples $y \sim p(y)$.

![Divider](resources/divider5.jpg)

## Uncertainty 

All measurements, models, and discretizations that we are working with exhibit uncertainties. For measurements and observations, they typically appear in the form of measurement errors. Model equations, on the other hand, usually encompass only parts of a system we're interested in (leaving the remainder as an uncertainty), while for numerical simulations we inevitably introduce discretization errors. In the context of machine learning, we additionally have errors introduced by the trained model. All these errors and unclear aspects make up the uncertainties of the predicted outcomes, the _predictive uncertainty_. For practical applications it's crucial to have means for quantifying this uncertainty. This is a central motivation for working with probabilistic models, and for adjacent fields such as in "uncertainty quantification" (UQ).


```{note} Aleatoric vs. Epistemic Uncertainty.
The predictive uncertainty in many cases can 
be distinguished in terms of two types of uncertainty:

- _Aleatoric_ uncertainty denotes uncertainty within the data, e.g., noise in measurements.

- _Epistemic_ uncertainty, on the other hand, describes uncertainties within a model such as a trained neural network.

A word of caution is important here:
while this distinction seems clear cut, both effects overlay and can be difficult to tell apart. E.g., when facing discretization errors, uncertain outcomes could be caused by unknown ambiguities in the data, or by a suboptimal discrete representation.
These aspects can be very difficult to disentangle in practice. 
```

Closely aligned, albeit taking a slightly different perspective, are so-called _simulation-based inference_ (SBI) methods. Here the main motivation is to estimate likelihoods in computer-based simulations, so that reliable probability distributions for the solutions can be obtained. The SBI viewpoint provides a methodological approach for working with computer simulations and uncertainties, and will provide a red thread for the following sections.


## Forward or Backward?

At this point it's important to revisit the central distinction between forward and inverse ("backward") problems: most classic numerical methods target ➡️ **forward** ➡️ problems to compute solutions for steady-state or future states of a system.

Forward problems arise in many settings, but across the board, at least as many problems are ⬅️ **inverse** ⬅️ problems, where a forward simulation plays a central role, but the main question is not a state that it generates, but rather the value of parameter of simulator to explain a certain measurement or observation. To formalize this, our simulator $f$ is parametrized by a set of inputs $\nu$, e.g., a viscosity, and takes states $x$ to produce a modified state $y$. We have an observation $\tilde{y}$ and are interested in the value of $\nu$ to produce the observation. In the easiest case this inverse problem can tackled as a minimization problem 
$\text{arg min}_{\nu} | f(x;\nu) - \tilde{y} |_2^2$. Solving it would tell us the viscosity of an observed material, and similar problems arise in pretty much all fields, from material science to cosmology. To simplify  the notation, we'll merge $\nu$ into $x$, and minimize for $x$ correspondingly, but it's important to keep in mind that $x$ can encompass any set of parameters or state samples that we'd like to solve for with our inverse problem.

In the following, we will focus on inverse problems, as these best illustrate the capabilities of the probabilistic modeling, but the algorithms discussed are not exclusively applicable to inverse problems (an example will follow).

## Simulation-based Inference

For inverse problems, it is in practice not sufficient to match a single observation $\tilde{y}$. Rather, we'd like to ensure that the parameter we obtain explains a wide range of observations, and we might be interested in the possibility of multiple values explaining our observations. Similarly, quantifying the uncertainty of the estimate is important in real world settings: is the observation explained by only a very narrow range of parameters, or could the parameter vary by orders of magnitude without really influencing the observation? These questions require a statistical analysis, typically called _inference_, to draw conclusions about the results obtained from the inverse problem solve. To connect this viewpoint with the distinction regarding epistemic and aleatoric uncertainties above, we're primarily addressing the latter here: which uncertainties lie in our observations, given a scientific hypothesis in the form of a simulator.

To formalize these inverse problems let's consider
a vector-valued input￼$x$ that can contain states and / or
the aforementioned parameters (like $\nu$).
We also have a
distribution of latent variables ￼ 
$z \sim p(z|x)$ that describes the unknown part of our system.
Examples for z are unobservable and stochastic variables , intermediate simulation steps, or the control flow of simulator.

```{note} Bayes theorem is fundamental for all of the following. For completeness, here it is: $p(x|y)~p(y) = p(y|x)~p(x)$. And it's worth keeping in mind that both sides are equivalent to the joint probabilities, i.e. $... = p(x,y) = p(y,x)$.
```


For $x$ there is a prior distribution X with a probability density  $p(x)$￼for the inputs, 
and the simulator produces an observation or output ￼$y \sim p(y | x, z)$. Thus, $x$ can take different values, maybe it contains some noise, and the $z$ is out of our control, and can likewise influence the $y$ that are produced.

The function for the conditional probability ￼$p(y|x)$ is called the **likelihood** function, and is a crucial value in the following. Note that it does not depend on $z$, as these latent states are out of our control. 
So we actually need to 
compute the marginal likelihood  ￼$p(y|x) = \int p(y, z | x) dz$  by integrating over all possible $z$. 
This is necessary because the likelihood function shouldn't depend on $z$, otherwise we'd need to know the exact values of $z$ before being able to calculate the likelihood.
Unfortunately, this is often intractable, as $z$ can be difficult to sample, and in some case we can't even control it in a reasonable way. 
Some algorithms have been proposed to compute likelihoods, one popular one is Approximate Bayesian Computation (ABC), but all approaches are highly expensive and require a lot of expert knowledge to set up. They suffer from the _curse of dimensionality_, i.e. become very expensive when facing larger numbers of degrees of freedom. Thus,
obtaining good approximations of the likelihood will be a topic that we'll revisit below.

With a function for the likelihood we can compute the 
**distribution of the posterior**, the main quantity we're after,
in the following way:
$p(x|y) = \frac{p(y|x)p(x)}{\int p(y|x') p(x') dx'}$, 
where the denominator
$\int p(y|x') p(x') dx'$ is called the _evidence_. 
The evidence is just $p(y)$, which shows
that the equation for the posterior follows directly from Bayes' theorem $p(x|y) = p(y|x) p(x) / p(y)$. 

The evidence can be computed with stochastic methods such  as Markov Chain Monte Carlo (MCMC).
It primarily "normalizes" our posterior distribution and is typically easier to obtain than the likelihood, but nonetheless still a challenging term.


```{admonition} Leveraging Deep Learning
:class: tip

This is were deep learning turns out to be extremely useful: we can use it to train a conditional density estimator ￼$q_\theta(x|y)$ for the posterior ￼$p(x|y)$ that allows sampling, and can be trained from simulations ￼$y \sim p(y|x)$ alone.

```

Deep learning has been instrumental to provide new ways of addressing the classic challenges of obtaining accurate estimates of posterior distributions, and this is what we'll focus on in this chapter.  Previously, we called our neural networks $f_\theta$, but in the following we'll use $q_\theta = f_\theta$ to make clear we're dealing with  a learned probability. Specifically, we'll target neural networks that learn a probability density, i.e. $\int q_\theta(x) dx = 1$.
We'll often first target unconditional densities, and then show how they can be modified to learn conditional versions $q_\theta(x|y)$. 

Looking ahead, the learned SBI methods, i.e. approaches for computing posterior distributions, have the following properties:

✅ Pro:
* Fast inference (once trained)
* Less affected by curse of dimensionality
* Can represent arbitrary priors

❌ Con:
* Require costly upfront training 
* Lacks rigorous theoretical guarantees

In the following we'll explain how to obtain and derive a very popular and powerful family of methods that can be summarized as **diffusion models**. We could simply provide the final algorithm (which will turn out to be surprisingly simple), but it's actually very interesting to see where it all comes from. 
We'll focus on the basics, and leave the _physics-based extensions_ (i.e. including differentiable simulators) for a later section. The path towards diffusion models also introduces a few highly interesting concepts from machine learning along the way, and provides a nice "red thread" for discussing seminal papers from the past few years. Here we go...

<br>

![Divider](resources/divider6.jpg)

```{note} Historic Alternative: Bayesian Neural Networks

A classic variant that should be mentioned here are "Bayesian Neural Networks". They 
follow Bayes more closely, and pre-scribe a prior distribution on the neural network 
parameters to learn the posterior distribution. Every weight and bias in the NN are assumed to be Gaussian with an own mean and variance, which are adjusted at training time. For inference, we can then "sample" a network, and use it like any regular NN.
Despite being a very good idea on paper, this method turned out to have problems with learning complex distributions, and requires careful tuning of the hyperparameters involved. Hence, these days, it's strongly recommended to use flow matching (or at least a diffusion model) instead.
If you're interested in details, BNNs with a code example can be found, e.g., in v0.3 of PBDL: https://arxiv.org/abs/2109.05237v3 .
```

