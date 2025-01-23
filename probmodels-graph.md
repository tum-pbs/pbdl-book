Graph-based Diffusion Models
=======================

Similar to classical numerics, regular grids are ideal for certain situations, but sub-optimal for others. Diffusion models are no different, but luckily the concepts of the previous sections do carry over when replacing the regular grids of the previous sections with graphs. Importantly, denoising and flow matching work similarly well on unstrucuted Eulerian meshes, as will be demonstrated below. This example will illustrate another important aspect: diffusion models excel at _completing_ distributions. I.e., even when the training has an incomplete distribution for a single example, the "global" view of learning from different examples let's the networks _complete_ the posterior distribution over the course of seeing many partial examples.

Many simulation problems like fluid flows are often poorly represented by a single mean solution. E.g., for many practical applications involving turbulence, it is crucial to **access the full distribution of possible flow states**, from which relevant statistics (e.g., RMS and two-point correlations) can be derived. This is where diffusion models can leverage their strengths: instead of having to simulate a lengthy transient phase to converge towards an equilibrium state, diffusion models can completely skip the transient warm-up, and directly produce the desired samples. Hence, this allows for computing the relevant flow statistics very efficiently compared to classic solvers.

## Diffusion Graph Nodes 

In the following, we'll demonstrate these capabilities based on the _diffusion graph net_ (DGN) approach {cite}`lino2024dgn`, the full source code for which [can be found here](https://github.com/tum-pbs/dgn4cfd/).

To learn the probability distribution of dynamical states of physical systems, defined by their discretization mesh and their physical parameters,  the DDPM and flow matching frameworks can directly be applied to the mesh nodes. Additionally, DGN introduces a second model variant, which operates in a pre-trained semantic _latent space_ rather than directly in the physical space (these variants will be called LDGN).

In contrast to relying on regular grid discretizations as in previous sections, the system’s geometry is now represented using a mesh with nodes $\mathcal{V}_M$ and edges ${\mathcal{E}}_M$, where each node $i$ is located at ${x}_i$. The system’s state at time $t$, ${Y}(t)$, is defined by $F$ continuous fields sampled at the mesh nodes: ${Y}(t) := \{ {y}_i(t) \in \mathbb{R}^{F} \ | \ i \in {\mathcal{V}}_M \}$, with the short form ${y}_i(t) \equiv {y}({x}_i,t)$. Simulators evolve the system through a sequence of states, $\mathcal{Y} = \{{Y}(t_0), {Y}(t_1), \dots, {Y}(t_n), \dots \}$, starting from an initial state ${Y}(t_0)$.
We assume that after an initial transient phase, the system reaches a statistical equilibrium. In this stage, statistical measures of ${Y}$, computed over sufficiently long time intervals, are time-invariant, even if the dynamics display oscillatory or chaotic behavior. The states in the equilibrium stage, ${\mathcal{Z}} \subset {\mathcal{Y}}$, depend only on the system’s geometry and physical parameters, and not on its initial state. This is illustrated in the following picture.

```{figure} resources/probmodels-graph-over.jpg
---
height: 180px
name: probmodels-graph-over
---
(a) DGN learns the probability distribution of the systems' converged states provided only a short trajectory of length $\delta << T$ per system. (b) An example with a turbulent wing experiment. The distribution learned by the LDGN model accurately captures the variance of all states (bottom right), despite seeing only an incomplete distribution for each wing during training (top right).
```

In many engineering applications, such as aerodynamics and structural vibrations, the primary focus is not on each individual state along the trajectory, but rather on the statistics that characterize the system’s dynamics. However, simulating a trajectory of converged states $\mathcal{Z}$ long enough to accurately capture these statistics can be very expensive, especially for real-world problems involving 3D chaotic systems. The following DGN approachs aims for directly sampling converged states ${Z}(t) \in \mathcal{Z}$ without simulating the initial transient phase. Subsequently, we can analyze the system's dynamics by drawing enough samples.

Given a dataset of short trajectories from $N$ systems, $\mathfrak{Z} = \{\mathcal{Z}_1, \mathcal{Z}_2, ..., \mathcal{Z}_N\}$, the goal in the following is to learn a probabilistic model of $\mathfrak{Z}$ that enables sampling of a converged state ${Z}(t) \in \mathcal{Z}$, conditioned on the system's mesh, boundary conditions, and physical parameters. This model must capture the underlying probability distributions even when trained on trajectories that are too short to fully characterize their individual statistics. Although this is an ill-posed problem, given sufficient training trajectories, diffusion models on graphs manage to uncover the statistical correlations and shared patterns, enabling interpolation across the condition space.


## Diffusion on Graphs

We'll use DDPM (and later on flow matching) to generate states ${Z}(t)$ by denoising a sample ${Z}^R \in \mathbb{R}^{|\mathcal{V}_M| \times F}$ drawn from an isotropic Gaussian distribution. The system’s conditional information is encoded in a directed graph ${\mathcal{G}} :=({\mathcal{V}}, {\mathcal{E}})$, where ${\mathcal{V}} \equiv {\mathcal{V}}_M$ and the mesh edges ${\mathcal{E}}_M$ are represented as bi-directional graph edges ${\mathcal{E}}$. Node attributes ${V}_c = \{{v}_{i}^c \ | \ i \in {\mathcal{V}} \}$ and edge attributes ${E}_c = \{{e}_{ij}^c \ | \ (i,j) \in {\mathcal{E}} \}$ encode the conditional features, including the relative positions between adjacent node, ${x}_j - {x}_i$. Domain-specific details on the node and edge encodings can be found in Appendix~\ref{app:datasets} and Table~\ref{tab:systems-io}.

In the \emph{diffusion} (or \emph{forward}) process, node features from ${Z}^1 \in \mathbb{R}^{|\mathcal{V}| \times F}$ to ${Z}^R \in \mathbb{R}^{|\mathcal{V}| \times F}$ are generated by sequentially adding Gaussian noise:
$
q({Z}^r|{Z}^{r-1})=\mathcal{N}({Z}^r; \sqrt{1-\beta_r} {Z}^{r-1}, \beta_r \mathbf{I}),
$
where $\beta_r \in (0,1)$, and $Z^0 \equiv Z(t)$. Any ${Z}^r$ can be sampled directly via:

$$
\begin{equation}
{Z}^r =  \sqrt{\bar{\alpha}_r} {Z}^0 +  \sqrt{1-\bar{\alpha}_r} {\epsilon},
\label{eq:noise}
\end{equation}
$$

with $\alpha_r := 1 - \beta_r$, $\bar{\alpha}_r := \prod_{s=1}^r \alpha_s$ and ${\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
The denoising process removes noise through learned Gaussian transitions:
$
p_\theta({Z}^{r-1}|{Z}^r) =\mathcal{N} ({Z}^{r-1}; {\mu}_\theta^r, {\Sigma}_\theta^r),
$
where the mean and variance are parameterized as:

$$
\begin{equation}
{\mu}_\theta^r = \frac{1}{\sqrt{\alpha_r}} \left( {Z}^r - \frac{\beta_r}{\sqrt{1-\bar{\alpha}_r}} {\epsilon}_\theta^r \right),
\qquad
{\Sigma}_\theta^r = \exp\left( \mathbf{v}_\theta^r \log \beta_r + (1-\mathbf{v}_\theta^r)\log \tilde{\beta}_r \right),
\label{eq:mu_param}
\end{equation}
$$

with $\tilde{\beta}_r := (1 - \bar{\alpha}_{r-1}) / (1 - \bar{\alpha}_r) \beta_r$. Here, ${\epsilon}_\theta^r \in \mathbb{R}^{|\mathcal{V}| \times F}$ predicts the noise ${\epsilon}$ in equation~(\ref{eq:noise}), and $\mathbf{v}_\theta^r \in \mathbb{R}^{|\mathcal{V}| \times F}$ interpolates between the two bounds of the process' entropy,  $\beta_r$ and $\tilde{\beta}_r$.

DGNs predict ${\epsilon}_\theta^r$ and $\mathbf{v}_\theta^r$ using a regular message-passing-based GNN {cite}`sanchez2020learning`. This takes ${Z}^{r-1}$ as input, and it is conditioned on graph ${\mathcal{G}}$, its node and edge features, and the diffusion step $r$:

$$
\begin{equation}
[{\epsilon}_\theta^r, \mathbf{v}_\theta^r] \leftarrow \text{{DGN}}_\theta({Z}^{r-1}, {\mathcal{G}}, {V}_c, {E}_c, r).
\end{equation}
$$

The _DGN_ network is trained using the loss function in equation~(\ref{eq:loss}). The full denoising process requires $R$ evaluations of the DGN to transition from ${Z}^R$ to ${Z}^0$.

DGN follows the widely used encoder-processor-decoder GNN architecture. In addition to the node and edge encoders, our encoder includes a diffusion-step encoder, which generates a vector ${r}_\text{emb} \in \mathbb{R}^{F_\text{emb}}$ that embeds the diffusion step $r$. The node encoder processes the conditional node features ${v}_i^c$, alongside ${r}_\text{emb}$. Specifically, the diffusion-step encoder and the node encoder operate as follows:

$$
\begin{equation}
{r}_\text{emb} \leftarrow
    \phi \circ {\small Linear} \circ {\small SinEmb} (r),
\quad
{v}_i \leftarrow {\small Linear} \left( \left[ \phi \circ {\small Linear} ({v}_i^c) \ | \ {r}_\text{emb} 
    \right] \right), 
\quad \forall i \in \mathcal{V},
\end{equation}
$$

where $\phi$ denotes the activation function and ${\small SinEmb}$ is the sinusoidal embedding function. The edge encoder applies a linear layer to the conditional edge features ${e}_{ij}^c$. 
The encoded node and edge features are $\mathbb{R}^{F_{h}}$-dimensional vectors  ($F_\text{emb} = 4 \times F_h$). We condition each message-passing layer on $r$ by projecting ${r}_\text{emb}$ to an $F_{h}$-dimensional space and adding the result to the node features before each of these layers -- i.e.,  ${v}_i \leftarrow  {v}_i + {\small Linear}({r}_\text{emb})$. Details on message passing can be found in Appendix~\ref{app:dgn_details}.

Previous work on graph-based diffusion models has used sequential message passing to propagate node features across the graph. However, this approach fails for large-scale phenomena, such as the flows studied in the context of DGN,  as denoising of global features becomes bottlenecked by the reach of message passing.
To address this, a multi-scale GNN is adopted for the processor, applying message passing on ${\mathcal{G}}$ and multiple coarsened versions of it in a U-Net fashion. This design leverages the U-Net’s effectiveness in removing both high- and low-frequency noise. To obtain each lower-resolution graph from its higher-resolution counterpart, we use Guillard’s coarsening algorithm, originally developed for fast mesh coarsening in CFD applications. As in the conventional U-Net, pooling and unpooling operations, now based on message passing, are used to transition between higher- and lower-resolution graphs. 


## Diffusion in Latent Space

Diffusion models can also operate in a lower-dimensional graph-based representation that is perceptually equivalent to $\mathfrak{Z}$. This space is defined as the latent space of a Variational Graph Auto-Encoder (VGAE) trained to reconstruct ${Z}(t)$. We'll refer to a DGN trained on this latent space as a Latent DGN (LDGN).

```{figure} resources/probmodels-graph-arch.jpg
---
height: 220px
name: probmodels-graph-arch
---
(a) The VGAE consists of a condition encoder, a (node) encoder, and a (node) decoder. The multi-scale latent features from the condition encoder serve as conditioning inputs to both the encoder and the decoder. (b) During LDGN inference, Gaussian noise is sampled in the VGAE latent space and, after multiple denoising steps conditioned on the low-resolution outputs from the VGAE's condition encoder, transformed into the physical space by the VGAE's decoder.
```


In this configuration, the VGAE captures high-frequency information (e.g., spatial gradients and small vortices), while the LDGN focuses on modeling mid- to large-scale patterns (e.g., the wake and vortex street). By decoupling these two tasks, the generative learning process is simplified, allowing the LDGN to concentrate on more meaningful latent representations that are less sensitive to small-scale fluctuations. Additionally, during inference, the VGAE’s decoder helps remove residual noise from the samples generated by the LDGN. This approach significantly reduces sampling costs since the LDGN operates on a smaller graph rather than directly on ${\mathcal{G}}$.

For the VGAE, an encoder-decoder architecture is used with an additional condition encoder to handle conditioning inputs (Figure~\ref{fig:diagram}a). The condition encoder processes ${V}_c$ and ${E}_c$, encoding these into latent node features ${V}^\ell_c$ and edge features ${E}^\ell_c$ across $L$ graphs $\{{\mathcal{G}}^\ell := ({\mathcal{V}}^\ell, {\mathcal{E}}^\ell) {I}d 1 \leq \ell \leq L\}$, where ${\mathcal{G}}^1 \equiv {\mathcal{G}}$ and the size of the graphs decreases progressively, i.e., $|{\mathcal{V}}^1| > |{\mathcal{V}}^2| > \dots > |{\mathcal{V}}^L|$. This transformation begins by linearly projecting ${V}_c$ and ${E}_c$ to a $F_\text{ae}$-dimensional space and applying two message-passing layers to yield ${V}^1_c$ and ${E}^1_c$. Then, $L-1$ encoding blocks are applied sequentially:

$$
\begin{equation}
\left[{V}^{\ell+1}_c, {E}^{\ell+1}_c \right] \leftarrow {\small MP} \circ {\small MP} \circ {\small GraphPool} \left({V}^\ell_c, {E}^\ell_c \right), \quad \text{for} \ l = 1, 2, \dots, L-1, 
\end{equation}
$$

where _MP_ denotes a message-passing layer and _GraphPool_ denotes a graph-pooling layer (see the diagram on Figure~\ref{fig:vgae}a).

The encoder produces two $F_L$-dimensional vectors for each node $i \in {\mathcal{V}}^L$, the mean ${\mu}_i$ and standard deviation ${\sigma}_i$ that parametrize a Gaussian distribution over the latent space. It takes as input a state ${Z}(t)$, which is linearly projected to a $F_\text{ae}$-dimensional vector space and then passed through $L-1$ sequential down-sampling blocks (message passing + graph pooling), each conditioned on the outputs of the condition encoder:

$$
\begin{equation}
    {V} \leftarrow {\small GraphPool} \circ {\small MP} \circ {\small MP} \left( {V} + {\small Linear}\left({V}^\ell_c \right), {\small Linear}\left({E}^\ell_c \right) \right), \ \text{for} \ l = 1, 2, \dots, L-1;
\end{equation}
$$

and a bottleneck block:

$$
\begin{equation}
    {V} \leftarrow {\small MP} \circ {\small MP} \left( {V} + {\small Linear}\left({V}^L_c \right), {\small Linear}\left({E}^L_c \right) \right).
\end{equation}
$$

The output features are passed through a node-wise MLP that returns ${\mu}_i$ and ${\sigma}_i$ for each node $i \in {\mathcal{V}}^L$. The latent variables are then computed as ${\zeta}_i = {\small BatchNorm}({\mu}_i + {\sigma}_i {\epsilon}_i$), where ${\epsilon}_i \sim \mathcal{N}(0, {I})$. Finally, the decoder mirrors the encoder, employing a symmetric architecture (replacing graph pooling by graph unpooling layers) to upsample the latent features back to the original graph ${\mathcal{G}}$ (Figure~\ref{fig:vgae}c). Its blocks are also conditioned on the outputs of the condition encoder. The message passing and the graph pooling and unpooling layers in the VGAE are the same as in the (L)DGN.

The VGAE is trained to reconstruct states ${Z}(t) \in \mathfrak{Z}$ with a KL-penalty towards a standard normal distribution on the learned latent space. Once trained, the LDGN can be trained following the approach in Section~\ref{sec:DGN}. However, the objective is now to learn the distribution of the latent states ${\zeta}$, defined on the coarse graph ${\mathcal{G}}^L$, conditioned on the outputs ${V}^L_c$ and ${E}^L_c$ from the condition encoder.
As illustrated in Figure~\ref{fig:diagram}b, during inference, the condition encoder generates the conditioning features ${V}^\ell_c$ and ${E}^\ell_c$ (for $l = 1, 2, \dots, L$), and after the LDGN completes its denoising steps, the decoder transforms the generated ${\zeta}_0$ back into the physical feature-space defined on ${\mathcal{G}}$.

Unlike in conventional VGAEs, the condition encoder is necessary because, at inference time, an encoding of ${V}_c$ and ${E}_c$ is needed on graph ${\mathcal{G}}^L$, where the LDGN operates. This encoding cannot be directly generated by the encoder, as it also requires ${Z}(t)$ as input, which is unavailable during inference. An alternative approach would be to define the conditions directly in the coarse representation of the system provided by ${\mathcal{G}}^L$, but this representation lacks fine-grained details, leading to sub-optimal results.


![Divider](resources/divider7.jpg)


## Turbulent Flows around Wings in 3D

Let's directly turn to a complex case to illustrate the capabilities of DGN. (A more basic case will be studied in the Jupyter notebook on the following page.)

The Wing experiments of the DGN project target wings in 3D turbulent flow, characterized by detailed vortices that form and dissipate on the wing surface.  This task is particularly challenging due to the high-dimensional, chaotic nature of turbulence and its inherent multi-scale interactions across a wide range of scales.
The geometry of the wings varies in terms of relative thickness, taper ratio, sweep angle, and twist angle. 
These simulations are computationally expensive, and using GNNs allows us to concentrate computational effort on the wing's surface, avoiding the need for costly volumetric fields. A regular grid around the wing would require over $10^5$ cells, in contrast to approximately 7,000 nodes for the surface mesh representation. The surface pressure can be used to determine both the aerodynamic performance of the wing and its structural requirements.
Fast access to the probabilistic distribution of these quantities would be highly valuable for aerodynamic modeling tasks. 
The training dataset for this task was generated using Detached Eddy Simulation (DES) with OpenFOAM’s PISO solver,
using 250 consecutive states shortly after the data-generating simulator reached statistical equilibrium.
This represents about **10%** of the states needed to achieve statistically stationary variance, thus the models are trained with a very partial view on each case.




## Distributional accuracy

A high accuracy for each sample does not necessarily imply that a model is learning the true distribution. In fact, these properties often conflict. For instance, in VGAEs, the KL-divergence penalty allows control over whether to prioritize sample quality or mode coverage.
To evaluate how well models capture the probability distribution of system states, we use the Wasserstein-2 distance. This metric can be computed in two ways: (i) by treating the distribution at each node independently and averaging the result across all nodes, or (ii) by considering the joint distribution across all nodes in the graph. These metrics are denoted as $W_2^\text{node}$ and $W_2^\text{graph}$, respectively. The node-level measure ($W_2^\text{node}$) provides insights into how accurately the model estimates point-wise statistics, such as the mean and standard deviation at each node. However, it does not penalize inaccurate spatial correlations, whereas the graph-wise measure ($W_2^\text{graph}$) does.

To ensure stable results when computing these metrics, the target distribution is represented by 2,500 consecutive states, and the predicted one by 3,000 samples.
While the trajectories in the training data are long enough to capture the mean flow, they fall short of capturing the standard deviation, spatial correlations, or higher-order statistics. Despite these challenges, the DGN, and especially the LDGN, are capable of accurately learning the complete probability distributions of the training trajectories and accurately generating new distribution for both in- and out-of-distribution physical settings. The figure below shows a qualitative evaluation together with correlation measurements. Both DGN variants also fare much better than the _Gaussian-Mixture model_ baseline denoted as GM-GNN.

```{figure} resources/probmodels-graph-wing.jpg
---
height: 220px
name: probmodels-graph-wing
---
(a) The _Wing_ task targets pressure distributions on a wing in 3D turbulent flow. (b) The standard deviation of the distribution generated by the LDGN is the closest to the ground-truth (shown here in terms or correlation).
```

In terms of Wasserstein distance $W_2^\text{graph}$, the latent-space diffusion model also outperforms the others:     
with a distance of $\textbf{1.95 ± 0.89}$, while DGN follows with $2.12 ± 0.90$, and the gaussian mixture model gives $4.32 ± 0.86$.

## Computational Performance

While comparisons between runtimes of different implementations always should be taken with a grain of salt.
Nonetheless, for the Wing experiments, the ground-truth simulator, running on 8 CPU threads, required 2,989 minutes to simulate the initial transient phase plus 2,500 equilibrium states. This duration is just enough to obtain a well converged variance. In contrast, the LDGN model took only 49 minutes on 8 CPU threads and 2.43 minutes on a single GPU to generate 3,000 samples.
If we consider the generation of a single converged state (for use as an initial condition in another simulator, for example), the speedup is four orders of magnitude on the CPU, and five orders of magnitude on the GPU. 
Thanks to its latent space, the LDGN model is not only more accurate, but also $8\times$ faster than the DGN model, while requiring only about 55\% more training time. 
These significant efficiency advantages suggest that graph-based diffusion models can be particularly valuable in scenarios where computational costs are otherwise prohibitive.

These results indicate that diffusion modeling in the context of unstructured simulations represent a significant step towards leveraging probabilistic methods in real-world engineering applications.
To highlight the aspects of DGN and its implementation, we now turn to a simpler test case that can be analyzed in detail within a Jupyter notebook.

