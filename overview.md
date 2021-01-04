Overview
============================

The following "book" of targets _"Physics-Based Deep Learning"_ techniques
(PBDL), i.e., the field of methods with combinations of physical modeling and
deep learning (DL) techniques. Here, DL will typically refer to methods based
on artificial neural networks. The general direction of PBDL represents a very
active, quickly growing and exciting field of research. As such, this collection 
of materials is a living document, and will grow and change over time. Feel free
to contribute 😀

[TUM Physics-based Simulation Group](https://ge.in.tum.de).

[Link collection](https://github.com/thunil/Physics-Based-Deep-Learning)

## Motivation

....

## Categorization

Within the area of _physics-based deep learning_, 
we can distinguish a variety of different 
approaches, from targeting designs, constraints, combined methods, and
optimizations to applications. More specifically, all approaches either target
_forward_ simulations (predicting state or temporal evolution) or _inverse_
problems (e.g., obtaining a parametrization for a physical system from
observations).

![An overview of categories of physics-based deep learning methods](resources/physics-based-deep-learning-overview.jpg)

No matter whether we're considering forward or inverse problem, 
the most crucial differentiation for the following topics lies in the 
nature of the integration  between DL techniques
and the domain knowledge, typically in the form of model euqations.
Looking ahead, we will particularly aim for a very tight intgration
of the two, that goes beyond soft-constraints in loss functions.
Taking a global perspective, the following three categories can be
identified to categorize _physics-based deep learning_ (PBDL)
techniques:

- _Data-driven_: the data is produced by a physical system (real or simulated),
  but no further interaction exists.

- _Loss-terms_: the physical dynamics (or parts thereof) are encoded in the
  loss function, typically in the form of differentiable operations. The
  learning process can repeatedly evaluate the loss, and usually receives
  gradients from a PDE-based formulation.

- _Interleaved_: the full physical simulation is interleaved and combined with
  an output from a deep neural network; this requires a fully differentiable
  simulator and represents the tightest coupling between the physical system and
  the learning process. Interleaved approaches are especially important for
  temporal evolutions, where they can yield an estimate of future behavior of the
  dynamics.

Thus, methods can be roughly categorized in terms of forward versus inverse
solve, and how tightly the physical model is integrated into the
optimization loop that trains the deep neural network. Here, especially approaches
that leverage _differentiable physics_ allow for very tight integration
of deep learning and numerical simulation methods.

The goal of this document is to introduce the different PBDL techniques,
ordered in terms of growing tightness of the integration, give practical 
starting points with code examples, and illustrate pros and cons of the 
different approaches. In particular, it's important to know in which scenarios 
each of the different techniques is particularly useful.


