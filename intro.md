Welcome ... 
============================

Welcome to the Physics-based Deep Learning Book 👋

**TL;DR**: 
This document targets a variety of combinations of physical simulations with deep learning.
As much as possible, the algorithms will come with hands-on code examples to quickly get started.
Beyond standard _supervised_ learning from data, we'll look at _physical loss_ constraints, and 
more tightly coupled learning algorithms with _differentiable simulations_.


```{figure} resources/teaser.png
---
height: 220px
name: pbdl-teaser
---
Some visual examples of hybrid solvers, i.e. numerical simulators that are enhanced by trained neural networks.
```
% Teaser, simple version:
% ![Teaser, simple version](resources/teaser.png)

## Coming up

As a _sneak preview_, in the next chapters we'll show:

- How to train networks to infer fluid flows around shapes like airfoils in one go, i.e., a _surrogate model_ that replaces a traditional numerical simulation.

- We'll show how to use model equations as residual to train networks that represent solutions, and how to improve upon these residual constraints by using _differentiable simulations_.

- How to more tightly interact with a full simulator for _control problems_. E.g., we'll demonstrate how to circumvent the convergence problems of standard reinforcement learning techniques by leveraging simulators in the training loop.

This _book_, where "book" stands for a collection of texts, equations, images and code examples,
is maintained by the
[TUM Physics-based Simulation Group](https://ge.in.tum.de). Feel free to contact us via
[old fashioned email](mailto:i15ge@cs.tum.edu) if you have any comments. 
If you find mistakes, please also let us know! We're aware that this document is far from perfect,
and we're eager to improve it. Thanks in advance!

This collection of materials is a living document, and will grow and change over time. 
Feel free to contribute 😀 
We also maintain a [link collection](https://github.com/thunil/Physics-Based-Deep-Learning) with recent research papers.

```{admonition} Executable code, right here, right now
:class: tip
We focus on jupyter notebooks, a key advantage of which is that all code examples
can be executed _on the spot_, out of a browser. You can modify things and 
immediately see what happens -- give it a try...
<br><br>
Oh, and it's great because it's [literate programming](https://en.wikipedia.org/wiki/Literate_programming).
```


---


## Thanks!

The contents of the following files would not have been possible without the help of many people. Here's an alphabetical list. Big kudos to everyone 🙏

- [Li-wei Chen](https://ge.in.tum.de/about/dr-liwei-chen/)
- [Philipp Holl](https://ge.in.tum.de/about/)
- [Maximilian Mueller](https://www.tum.de)
- [Patrick Schnell](https://ge.in.tum.de/about/patrick-schnell/)
- [Nils Thuerey](https://ge.in.tum.de/about/n-thuerey/)
- [Kiwon Um](https://ge.in.tum.de/about/kiwon/)

<!-- % some markdown tests follow ...

---

a b c

```{admonition} My title2
:class: seealso
See also... Test link: {doc}`supervised`
```

✅  Do this , ❌  Don't do this

% ---------------- -->

---


## TODOs , include

- DP intro, check transpose of Jacobians in equations
- DP SoL, generate result
- DP control, show targets at bottom?
- include latent space physics, mention LSTM, end of supervised?


## Other planned content

Supervised simple starting point

- add surrogates for shape opt?

Physical losses 

-    PINNs -> are unsupervised a la tompson; all DL NNs are "supervised" during learning, unsup just means not precomputed and goes through function

-    discuss CG solver, tompson as basic ''unsupervisedd'' example?

Diff phys, start with overview of idea: gradients via autodiff, then run GD

-    illustrate and discuss gradients -> mult. for chain rule; (later: more general PG chain w func composition)

beyond GD: re-cap newton & co

Phys grad (PGs) as fundamental improvement, PNAS case; add more complex one?
        PG update of poisson eq? see PNAS-template-main.tex.bak01-poissonUpdate , explicitly lists GD and PG updates

- PGa 2020 Sept, content: ML & opt
    Gradients.pdf, -> overleaf-physgrad/ 

- PGb 201002-beforeVac, content: v1,v2,old - more PG focused
    -> general intro versions

TODO, for version 2.x add: 

time series, sequence prediction?] {cite}`wiewel2019lss,bkim2019deep,wiewel2020lsssubdiv`
    include DeepFluids variant?

[BAYES , prob?]
    include results Jakob / Maximilian

[unstruct / lagrangian] {cite}`prantl2019tranquil,ummenhofer2019contconv`
    include ContConv / Lukas


