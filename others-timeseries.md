Model Reduction and Time Series
=======================

An inherent challenge for many practical PDE solvers is the large dimensionality of the problem.
Our model $\mathcal{P}$ is typically discretized with $\mathcal{O}(n^3)$ samples for a 3 dimensional 
problem (with $n$ denoting the number of samples along one axis), 
and for time-dependent phenomena we additionally have a discretization along
time. The latter typically scales in accordance to the spatial dimensions, giving an
overall number of samples on the order of $\mathcal{O}(n^4)$. Not surprisingly, 
the workload in these situations quickly explodes for larger $n$ (and for practical high-fidelity applications we want $n$ to be as large as possible).

One popular way to reduce the complexity is to map a spatial state of our system $\mathbf{s_t} \in \mathbb{R}^{n^3}$
into a much lower dimensional state $\mathbf{c_t} \in \mathbb{R}^{m}$, with $m \ll n^3$. Within this latent space,
we estimate the evolution of our system by inferring a new state $\mathbf{c_{t+1}}$, which we then decode to obtain $\mathbf{s_{t+1}}$. In order for this to work, it's crucial that we can choose $m$ large enough that it captures all important structures in our solution manifold, and that the time prediction of $\mathbf{c_{t+1}}$ can be computed efficiently, such that we obtain a gain in performance despite the additional encoding and decoding steps. In practice, due to the explosion in terms of unknowns for regular simulations (the $\mathcal{O}(n^3)$ above) coupled a super-linear complexity for computing a new state, working with the latent space points $\mathbf{c}$ quickly pays off for small $m$.

However, it's crucial that encoder and decoder do a good job at reducing the dimensionality of the problem. This is a very good task for DL approaches. Furthermore, we then need a time evolution of the latent space states $\mathbf{c}$, and for most practical model equations, we cannot find closed form solutions to evolve $\mathbf{c}$. Hence, this likewise poses a very good problem for learning methods. To summarize, we're facing to challenges: learning a good spatial encoding and decoding, together with learning an accurate time evolution.
Below, we will describe an approach to solve this problem following Wiewel et al.
{cite}`wiewel2019lss` & {cite}`wiewel2020lsssubdiv`, which in turn employs 
the encoder/decoder of Kim et al. {cite}`bkim2019deep`.


```{figure} resources/timeseries-lsp-overview.jpg
---
height: 200px
name: timeseries-lsp-overview
---
For time series predictions with ROMs, we encode the state of our system with an encoder $f_e$, predict 
the time evolution with $f_t$, and then decode the full spatial information with a decoder $f_d$.
```


## Reduced Order Models 

Reducing the order of computational models, often called _reduced order modeling_ (ROM) or _model reduction_,
as a classic topic in the computational field. Traditional techniques often employ techniques such as principal component analysis to arrive at a basis for a chosen space of solution. However, being linear by construction, these approaches have inherent limitations when representing complex, non-linear solution manifolds. And in practice, all "interesting" solutions are highly non-linear.


$\text{arg min}_{\theta} | f_d( f_e(x;\theta_e) ;\theta_d) - x |_2^2$

$f_e: \mathbb{R}^{n^3} \rightarrow \mathbb{R}^{m}$

$f_d: \mathbb{R}^{m} \rightarrow \mathbb{R}^{n^3}$


separable model



## Time Series


...

