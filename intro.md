Welcome ... 
============================

```{figure} resources/logo-xl.jpg
---
name: pbdl-logo-large
---
```

Welcome to the _Physics-based Deep Learning Book_ (v0.2) 👋

**TL;DR**: 
This document contains a practical and comprehensive introduction of everything
related to deep learning in the context of physical simulations.
As much as possible, all topics come with hands-on code examples in the 
form of Jupyter notebooks to quickly get started.
Beyond standard _supervised_ learning from data, we'll look at _physical loss_ constraints, 
more tightly coupled learning algorithms with _differentiable simulations_, 
training algorithms tailored to physics problems,
as well as 
reinforcement learning and uncertainty modeling.
We live in exciting times: these methods have a huge potential to fundamentally 
change what computer simulations can achieve.

```{note} 
_What's new in v0.2?_
For readers familiar with v0.1 of this text, the extended section {doc}`diffphys-examples` and the
brand new chapter on improved learning methods for physics problems (starting with {doc}`physgrad`) are highly recommended starting points.
```

---

## Coming up

As a _sneak preview_, the next chapters will show:

- How to train networks to infer a fluid flow around shapes like airfoils, and estimate the uncertainty of the prediction. This gives a _surrogate model_ that replaces a traditional numerical simulation.

- How to use model equations as residuals to train networks that represent solutions, and how to improve upon these residual constraints by using _differentiable simulations_.

- How to more tightly interact with a full simulator for _inverse problems_. E.g., we'll demonstrate how to circumvent the convergence problems of standard reinforcement learning techniques by leveraging simulators in the training loop.

- We'll also discuss the importance of _inversion_ for the update steps, and how higher-order information can be used to speed up convergence, and obtain more accurate neural networks.

Throughout this text,
we will introduce different approaches for introducing physical models
into deep learning, i.e., _physics-based deep learning_ (PBDL) approaches.
These algorithmic variants will be introduced in order of increasing
tightness of the integration, and the pros and cons of the different approaches
will be discussed. It's important to know in which scenarios each of the
different techniques is particularly useful.


```{admonition} Executable code, right here, right now
:class: tip
We focus on Jupyter notebooks, a key advantage of which is that all code examples
can be executed _on the spot_, from your browser. You can modify things and 
immediately see what happens -- give it a try by 
[[running this teaser example in your browser]](https://colab.research.google.com/github/tum-pbs/pbdl-book/blob/main/intro-teaser.ipynb).

Plus, Jupyter notebooks are great because they're a form of [literate programming](https://en.wikipedia.org/wiki/Literate_programming).
```



## Comments and suggestions

This _book_, where "book" stands for a collection of digital texts and code examples,
is maintained by the
[Physics-based Simulation Group](https://ge.in.tum.de) at [TUM](https://www.tum.de). 
Feel free to contact us if you have any comments, e.g., via [old fashioned email](mailto:i15ge@cs.tum.edu).
If you find mistakes, please also let us know! We're aware that this document is far from perfect,
and we're eager to improve it. Thanks in advance 😀! 
Btw., we also maintain a [link collection](https://github.com/thunil/Physics-Based-Deep-Learning) with recent research papers.


```{figure} resources/divider-mult.jpg
---
height: 220px
name: divider-mult
---
Some visual examples of numerically simulated time sequences. In this book, we explain how to realize algorithms that use neural networks alongside numerical solvers.
```


## Thanks!

This project would not have been possible without the help of many people who contributed. Thanks to everyone 🙏 Here's an alphabetical list:

- [Philipp Holl](https://ge.in.tum.de/about/philipp-holl/)
- [Maximilian Mueller](https://ge.in.tum.de/)
- [Patrick Schnell](https://ge.in.tum.de/about/patrick-schnell/)
- [Felix Trost](https://ge.in.tum.de/)
- [Nils Thuerey](https://ge.in.tum.de/about/n-thuerey/)
- [Kiwon Um](https://ge.in.tum.de/about/kiwon/)

Additional thanks go to 
Georg Kohl for the nice divider images (cf. {cite}`kohl2020lsim`), 
Li-Wei Chen for the airfoil data image, 
and to 
Chloe Paillard for proofreading parts of the document.

% future:
% - [Georg Kohl](https://ge.in.tum.de/about/georg-kohl/)

## Citation

If you find this book useful, please cite it via:
```
@book{thuerey2021pbdl,
  title={Physics-based Deep Learning},
  author={Nils Thuerey and Philipp Holl and Maximilian Mueller and Patrick Schnell and Felix Trost and Kiwon Um},
  url={https://physicsbaseddeeplearning.org},
  year={2021},
  publisher={WWW}
}
```

