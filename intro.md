Welcome ... 
============================

```{figure} resources/logo-xl.jpg
---
name: pbdl-logo-large
---
```

Welcome to the _Physics-based Deep Learning Book_ (v0.3, the _GenAI_ edition) 👋

**TL;DR**: 
This document is a hands-on, comprehensive guide to deep learning in the realm of physical simulations. Rather than just theory, we emphasize practical application: every concept is paired with interactive Jupyter notebooks to get you up and running quickly. Beyond traditional supervised learning, we dive into physical _loss-constraints_, _differentiable_ simulations, _diffusion-based_ approaches for _probabilistic generative AI_, as well as reinforcement learning and advanced neural network architectures. These foundations are paving the way for the next generation of scientific _foundation models_.
We are living in an era of rapid transformation. These methods have the potential to redefine what’s possible in computational science.

```{note} 
_What's new in v0.3?_
This latest edition adds a major new chapter on generative modeling, covering powerful techniques like denoising, flow-matching, autoregressive learning, physics-integrated constraints, and diffusion-based graph networks. We've also introduced a dedicated section on neural architectures specifically designed for physics simulations. All code examples have been updated to leverage the latest frameworks.
```

---

## Coming up

As a _sneak preview_, the next chapters will show:

- How to train neural networks to [predict the fluid flow around airfoils with diffusion modeling](probmodels-ddpm-fm). This gives a probabilistic _surrogate model_ that replaces and outperforms traditional simulators.

- How to use model equations as residuals to train networks that [represent solutions](diffphys-dpvspinn), and how to improve upon these residual constraints by using [differentiable simulations](diffphys-code-sol).

- How to more tightly interact with a full simulator for [inverse problems](diffphys-code-control). E.g., we'll demonstrate how to circumvent the convergence problems of standard reinforcement learning techniques by leveraging [simulators in the training loop](reinflearn-code).

- We'll also discuss the importance of [choosing the right network architecture](supervised-arch): whether to consider global or local interactions, continuous or discrete representations, and structured versus unstructured graph meshes.

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

This project would not have been possible without the help of the many people who contributed to it. A big thanks to everyone 🙏 Here's an alphabetical list:

- [Benjamin Holzschuh](https://ge.in.tum.de/about/)
- [Philipp Holl](https://ge.in.tum.de/about/philipp-holl/)
- [Georg Kohl](https://ge.in.tum.de/about/georg-kohl/)
- [Mario Lino](https://ge.in.tum.de/about/mario-lino/)
- [Qiang Liu](https://ge.in.tum.de/about/qiang-liu/)
- [Patrick Schnell](https://ge.in.tum.de/about/patrick-schnell/)
- [Felix Trost](https://ge.in.tum.de/about/)
- [Nils Thuerey](https://ge.in.tum.de/about/n-thuerey/)


Additional thanks go to 
Li-Wei Chen, 
Xin Luo,
Maximilian Mueller,
Chloe Paillard,
Kiwon Um,
and all github contributors!

## Citation

If you find this book useful, please cite it via:
```
@book{thuerey2021pbdl,
  title={Physics-based Deep Learning},
  author={N. Thuerey and B. Holzschuh  and P. Holl  and G. Kohl  and M. Lino  and Q. Liu and P. Schnell  and F. Trost},
  url={https://physicsbaseddeeplearning.org},
  year={2021},
  publisher={WWW}
}
```

## Time to get started

The future of simulation is being rewritten, and with the following AI and deep learning techniques, you’ll be at the forefront of these developments. Let’s dive in!

