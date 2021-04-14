Introduction to Posterior Inference
=======================

We have to keep in mind that for all measurements, models, and discretizations we have uncertainties. In the former, this typically appears in the form of measurements errors, model equations usually encompass only parts of a system we're interested in, and for numerical simulations we inherently introduce discretization errors. So a very important question to ask here is how sure we can be sure that an answer we obtain is the correct one. From a statistics viewpoint, we'd like to know the probability distribution for the posterior, i.e., the outcomes.

This admittedly becomes even more difficult in the context of machine learning:
we're typically facing the task of approximating complex and unknown functions.
From a probabilistic perspective, the standard process of training an NN here
yields a _maximum likelihood estimation_ (MLE) for the parameters of the network.
However, this MLE viewpoint does not take any of the uncertainties mentioned above into account:
for DL training, we likewise have a numerical optimization, and hence an inherent
approximation error and uncertainty regarding the learned representation.
Ideally, we could change our learning problem such that we could do _posterior inference_,
i.e. learn to produce the full output distribution. However, this turns out to be an
extremely difficult task.

This where so called _Bayesian neural network_ (BNN) approaches come into play. They 
make posterior inference possible by making assumptions about the probability 
distributions of individual parameters of the network. Nonetheless, the task
remains very challenging. Training a BNN is typically significantly more difficult
than training a regular NN. However, this should come as no surprise, as we're trying to 
learn something fundamentally different in this case: a full probability distribution 
instead of a point estimate.

![Divider](resources/divider5.jpg)

## A practical example

first example here with airfoils, extension from {doc}`supervised-airfoils`


