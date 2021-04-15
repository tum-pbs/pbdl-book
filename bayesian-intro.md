Introduction to Posterior Inference
=======================

We should keep in mind that for all measurements, models, and discretizations we have uncertainties. For the former, this typically appears in the form of measurements errors, while model equations usually encompass only parts of a system we're interested in, and for numerical simulations we inherently introduce discretization errors. So a very important question to ask here is how sure we can be sure that an answer we obtain is the correct one. From a statistics viewpoint, we'd like to know the probability distribution for the posterior, i.e., the different outcomes that are possible.

This admittedly becomes even more difficult in the context of machine learning:
we're typically facing the task of approximating complex and unknown functions.
From a probabilistic perspective, the standard process of training an NN here
yields a _maximum likelihood estimation_ (MLE) for the parameters of the network.
However, this MLE viewpoint does not take any of the uncertainties mentioned above into account:
for DL training, we likewise have a numerical optimization, and hence an inherent
approximation error and uncertainty regarding the learned representation.
Ideally, we should reformulate our learning problem such that it enables _posterior inference_,
i.e. learn to produce the full output distribution. However, this turns out to be an
extremely difficult task.

This where so called _Bayesian neural network_ (BNN) approaches come into play. They 
make a form of posterior inference possible by making assumptions about the probability 
distributions of individual parameters of the network. With a distribution for the
parameters we can evaluate the network multiple times to obtain different versions
of the output, and in this way sample the distribution of the output.

Nonetheless, the task
remains very challenging. Training a BNN is typically significantly more difficult
than training a regular NN. However, this should come as no surprise, as we're trying to 
learn something fundamentally different here: a full probability distribution 
instead of a point estimate. (All previous chapters "just" dealt with
learning such point estimates.)

![Divider](resources/divider5.jpg)

## Introduction to Bayesian Neural Networks


**TODO, integrate Maximilians intro section here**
...


## A practical example

As a first real example for posterior inference with BNNs, let's revisit the
case of turbulent flows around airfoils, from {doc}`supervised-airfoils`. However,
in contrast to the point estimate learned in this section, we'll now aim for
learning the full posterior.

