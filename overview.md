Overview
============================

The following collection of digital documents, i.e. "book", 
targets _Physics-Based Deep Learning_ techniques.
By that we mean combining physical modeling and numerical simulations with
methods based on artificial neural networks. 
The general direction of Physics-Based Deep Learning represents a very
active, quickly growing and exciting field of research -- we want to provide 
a starting point for new researchers as well as a hands-on introduction into
state-of-the-art resarch topics. 




```{figure} resources/overview-pano.jpg
---
height: 240px
name: overview-pano
---
Understanding our environment, and predicting how it will evolve is one of the key challenges of humankind.
```

## Motivation

From weather and climate forecasts {cite}`stocker2014climate`,
over quantum physics {cite}`o2016scalable`,
to the control of plasma fusion {cite}`maingi2019fesreport`,
using numerical analysis to obtain solutions for physical models has
become an integral part of science.

At the same time, machine learning technologies and deep neural networks in particular,
have led to impressive achievements in a variety of fields:
from image classification {cite}`krizhevsky2012` over
natural language processing {cite}`radford2019language`, 
and more recently also for protein folding {cite}`alquraishi2019alphafold`.
The field is very vibrant, and quickly developing, with the promise of vast possibilities.

At the same time, the successes of deep learning (DL) approaches 
has given rise to concerns that this technology has 
the potential to replace the traditional, simulation-driven approach to
science. Instead of relying on models that are carefully crafted
from first principles, can data collections of sufficient size
be processed to provide the correct answers instead?
In short: this concern is unfounded. As we'll show in the next chapters,
it is crucial to bring together both worlds: _classical numerical techniques_
and _deep learning_.

One central reason for the importance of this combination is
that DL approaches are simply not powerful enough by themselves.
Given the current state of the art, the clear breakthroughs of DL
in physical applications are outstanding, the proposed techniques are novel,
sometimes difficult to apply, and
significant practical difficulties combing physics and DL persist.
Also, many fundamental theoretical questions remain unaddressed, most importantly
regarding data efficienty and generalization.

Over the course of the last decades,
highly specialized and accurate discretization schemes have
been developed to solve fundamental model equations such
as the Navier-Stokes, Maxwell’s, or Schroedinger’s equations.
Seemingly trivial changes to the discretization can determine
whether key phenomena are visible in the solutions or not.
Rather than discarding the powerful methods that have been
carefully developed in the field of numerical mathematics, it 
is highly beneficial for DL to use them as much as possible.

```{admonition} Goals of this document
:class: tip
Thus, the key aspects that we want to address in the following are:
- explain how to use DL,
- and how to combine it with existing knowledge of physics and simulations,
- **without throwing away** all existing numerical knowledge and techniques.
```

Thus, we want to build on all the powerful techniques that we have
at our disposal, and use them wherever we can.
I.e., our goal is to _reconcile_ the data-centered
viewpoint and the physical simuation viewpoint.



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


## More Specifically

To be a bit more specific, _physics_ is a huge field, and we can't cover everything... 

```{note}
For now our focus are:
- _field-based simulations_ (no Lagrangian methods)
- combinations with _deep learning_ (plenty of other interesting ML techniques, but not here)
- experiments as _outlook_ (replace synthetic data with real)
```

It's also worth noting that we're starting to build the methods from some very
fundamental steps. Here are some considerations for skipping ahead to the later chapters.

```{admonition} You can skip ahead if...
:class: tip

- you're very familiar with numerical methods and PDE solvers, and want to get started with DL topics right away. The _Supervised Learning_ chapter is a good starting point then.

- On the other hand, if you're already deep into ANNs&Co, and you'd like to skip ahead to the research related topics, we recommend starting in the _Physical Loss Terms_ chapter, which lays the foundations for the next chapters.

A brief look at our _Notation_ won't hurt in both cases, though!
```

---
<br>
<br>
<br>

<!-- ## A brief history of PBDL in the context of Fluids

First:

Tompson, seminal...

Chu, descriptors, early but not used

Ling et al. isotropic turb, small FC, unused?

PINNs ... and more ... -->


## Deep Learning and Neural Networks

TODO

Very brief intro, basic equations... approximate $f^*(x)=y$ with NN $f(x;\theta)$ ...

learn via GD, $\partial f / \partial \theta$ 

general goal, minimize E for e(x,y) ... cf. eq. 8.1 from DLbook

introduce scalar loss, always(!) scalar...
  (also called *cost* or *objective* function)

distuingish: training, validation and (out of distribution!) test sets.

Read chapters 6 to 9 of the [Deep Learning book](https://www.deeplearningbook.org),
especially about [MLPs]https://www.deeplearningbook.org/contents/mlp.html and 
"Conv-Nets", i.e. [CNNs](https://www.deeplearningbook.org/contents/convnets.html).

**Note:** Classic distinction between _classification_ and _regression_ problems not so important here,
we only deal with _regression_ problems in the following.

maximum likelihood estimation

Also interesting: from a math standpoint ''just'' non-linear optimization ...

