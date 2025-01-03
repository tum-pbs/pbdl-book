Neural Network Architectures
=======================

The connectivity of the individual "neurons" in a neural network has a substantial influence on the capabilities of a network consisting of a large number of them. Over the course of many years, several key architectures have emerged as particularly useful choices, and in the following we'll go over the main considerations for choosing an architecture. Our focus is to introduce ways of incorporating PDE-based models (the "physics"), rather than the subtleties of NN architectures, and hence this section will be kept relatively brief.

# Spatial Arrangement

A first, fundamental aspect to consider for choosing an architecture (and for ruling out unsuitable options) is the spatial arrangement of the data samples. 
We can distinguish four cases here: 

1. No spatial arrangement,
2. A regular spacing on a grid (_structured),
3. An irregular arrangement (_unstructured_), and
4. Irregular positions without connectivity (_particle_ / _point_-based).

For certain problems, there is no spatial information or arrangement (case 1). E.g., predicting the temporal evolution of temperature and pressure of a single measurement probe over time does not involve any spatial dimension. The opposite case are probes placed on a perfect grid (case 2), or at arbitrary locations (3). The last variant (case 4) is a slightly special case of the third one, where no clear or persistent links between sample points are known. This can, e.g., be the case for particle-based representations of turbulent fluids, where neighborhood change quickly over time.

## No spatially arranged inputs

The first case is a somewhat special one: without any information about spatial arrangements, only _dense_ ("fully connected" / MLP) neural networks are applicable. 

If you decide to use a **neural fields** like approach where the network receives the position as input, this has the same effect: the NN will not have any direct means of querying neighbors via architectural tricks ("inductive biases"). In all these cases, the building blocks below won't be applicable, and it's worth considering whether you can introduce more structure via a discretization.

Note that _physics-informed neural networks_ (PINNs) also fall into this category. We'll go into more detail here later on ({doc}`diffphys`), but generally it's advisable to consider switching to an approach that employs prior knowledge in the form or a discretization. This usually substantially improves inference accuracy and improves convergence. PINNs were found to be mostly unsuited for real-world problems.

Even when focusing on dense layers, this still leaves a few choices concerning the number of layers, their size, and activations. The other three cases have the same choices, and these hyperparameters of the architectures are typically determined over the course of training runs. Hence, we'll focus on the remaining three cases with spatial information in the following, as differences can have a profound impact here. So, below we'll focus on cases where we have a "computational domain" for a region of interest, in which the samples are located.

# Local vs Global

The most important aspect of different architectures then is the question of the receptive field: this means for any single sample in our domain, which neighborhood of other sample points can influence the solution at this point. This is similar to classic considerations for PDE solving, where denoting a PDE as hyperbolic indicates its local, wave-like behavior in contrast to an elliptic one with global behavior. Certain NN architectures such as the classic convolutional neural networks (CNNs) support only local influences and receptive fields, while hierarchies with pooling expand these receptive field to effectively global ones. An interest variant here are spectral architectures like FNOs, which provide global receptive fields at the expense of other aspects. In addition Transformers (with attention mechanisms), provide a more complicated but scalable alternative here.


|             | Grid            | Unstructured      | Points            | Non-spatial |
|-------------|-----------------|-------------------|-------------------|-------------|
| Local       | CNN , ResNet    | GNN               | CConv             | -           |
| Global      |                 |                   |                   |  MLP        |
| - Hierarchy | U-Net, Dilation | Multi-scale GNN   | Multi-scale CConv | -           |
| - Spectral  | FNO             | Spectral GNN      | (-)               | -           |
| - Sequence  | Transformer     | Graph Transformer | Point Trafo.      | -           |


Thus, a fundamental distinction should be made in terms of local vs global architectures, and for the latter, how they realize the global receptive field. The following table provides a first overview, and below we'll discuss the pros and cons of each variant.

```{note}
Knowledge about the dependencies in your data, i.e., whether the dependencies are local or global, is important knowledge that should be leveraged. 

If your data has primarily **local** influences, choosing an architecture with support for global receptive fields will most likely have negative effects on accuracy: the network will "waste" resources to try and capture global effects, or worst case approximate local effects with smoothed out global modes.

Vice versa, trying to approximate a **global** influence with a limited receptive field will be an unsolvable task, and most likely introduce substantial errors.
```

# Regular, unstructured and point-wise data

The most natural start for making use of spatially arranged data is to employ a regular grid. Note that it doesn't have to be a
Cartesian grid, but could be e deformed and adaptive grid {cite}`chen2021highacc`. The only requirement is a grid-like connectivity of the samples, even if 
they have an irregular spacing. 

For unstructured data, graph-based neural networks (GNNs) are a good choice. While they're often discussed in terms of _message-passing_ operations,
the main approach (and most algorithms) from the grid world directly transfer: the basic operation of a message-passing
step on a GNN is the same as a convolution on a grid. And hierarchies can be build in very similar ways to grids. Hence, while we'll primarily discuss
grids below, keep in mind that the approaches carry over to GNNs. As dealing with graph structures makes the implementation more
complicated, we'll go into more detail here later on in {doc}`graphs`.

Finally, point-wise (Lagrangian) samples can be seen as unstructured grids without connectivity. However, it can be worth explicitly treating
them in this way for improved learning and inference performance. Nonetheless, the two main ideas of convolutions and hierarchies carry over
to Lagrangian data.

# Hierarchies

A powerful and natural tool to work with **local** dependencies are convolutional layers. The corresponding neural networks (CNNs) are 
a classic building block of deep learning, and very well researched and supported throughout. They are comparatively easy to 
train, and usually very efficiently implemented in APIs. They also provide a natural connection to classical numerics: classic discretizations
of differential operators such as gradient and Laplacians are often thought of in terms of "stencils", which are an equivalent of
a convolutional layer with a set of chosen weights. E.g., consider the classic stencil for a normalized Laplacian $\nabla^2$ in 1D: $[1, -2, 1]$. 
It can directly be mapped to a 1D convolution with kernel size 3 and a single input and output channel.

[TODO, image simple and deformed grids]

Using convolutional layers is quite straight forward, but the question of how to incorporate **global** dependencies into 
CNNs is an interesting one. Over time, two fundamental approaches have been established here in the field:
_hierarchical_ networks via pooling (U-Nets {cite}`ronneberger2015unet`), and sparse, point wise samples with enlarged spacing (Dilation {cite}`yu2015dilate`). They both reach
the goal of establishing a global receptive field, but have a few interesting differences under the hood.

* U-Nets are based on _pooling_ operations. Akin to a geometric multigrid hierarchy, the spatial samples are downsampled to coarser and
coarser grids, and upsampled in the later half of the network. This means that even if we keep the kernel size of a convolution fixed, the 
convolution will be able to "look" further in terms of physical space due to the previous downsampling operation. While the different 
re-sampling methods (mean, average, point-wise ...) have a minor effect, a crucial ingredient for U-Net are _skip connection_. The connect
the earlier layers of the first half directly with the second half via feature concatenation. This turns out to be crucial to avoid 
a loss of information. Typically, the deepest "bottle-neck" layer with the coarsest representation has trouble storing all details 
of the finest one. Providing this information explicitly via a skip-connection is crucial for improving accuracy.

* Dilation in the form of _dilated convolutions_ places the sampling points for convolutions further apart. Hence instead of, e.g., looking at a 3x3 neighborhood, 
a convolution considers a 5x5 neighborhood but only includes 3x3 samples when calculating the convolution. The other samples in-between the used points are
typically simply ignored.

While both approaches reach the goal, and can perform very well, there's an interesting tradeoff: U-Nets take a bit more effort to implement, but can be much faster. The reason for the performance boost is the sub-optimal memory access of the dilated convolutions: they skip through memory with a large stride, which gives a slower performance. The U-Nets, on the other had, basically precompute a compressed memory representation in the form of a coarse grid. Convolutions on this coarse grid are then highly efficient to compute. However, this requires slightly more effort to implement in the form of adding appropriate pooling layers (dilated convolutions can be as easy to implement as replacing the call to the regular convolution with a dilated one). The implementation effort of a U-Net can pay off significantly in the long run, when a trained network should be deployed in an application.

Note that this difference is not present for graph nets: here the memory access is always irregular, and dilation is unpopular as the strides would be costly to compute on general graphs. Hence, hierarchies in the form of multi-scale GNNs are highly recommended if global dependencies exist in the data.


# Spectral methods

A fundamentally different avenue for establishing global receptive fields is provided by spectral methods, typically making use of Fourier transforms to transfer spatial data to the frequency domain. The most popular approach from this class of methods are _Fourier Neural Operators_ (FNOs) {cite}`fno20xx`. An interesting aspect is the promise of a continuous representation via the functional representation, where a word of caution is appropriate: the function spaces are typically truncated, so it is often questionable whether the frequency representation really yields suitable solutions beyond the resolution of the training data.

In the following, however, we'll focus on the aspect of receptive fields in conjunction with performance aspects. Here, FNO-like methods have an interesting behavior: they modify frequency information with a dense layer. As the frequency signal after a Fourier transform would have the same size as the input, the dense layer works on a set of the $M$ largest frequencies. For a two dimensional input that means $M^2$ modes, and the corresponding dense layer thus requires $M^4$ parameters.

An inherent advantage and consequence of the frequency domain is that all basis functions have global support. That means despite only working with a subset of all frequencies, FNOs can process (and modify) all parts of an input signal. This natural treatment of **global dependencies** is a inherent advantage of spectral methods.

Unfortunately, they're not well suited for higher dimensional problems: Moving from two to three dimensions increases the size of the frequencies to be handled to $M^3$. For the dense layer, this means $M^6$ parameters, a cubic increase. In contrast, a regular convolution with kernel size $K$ requires $K^2$ weights in 2D, and $K^3$ in 3D. Thus, architectures like CNNs require much fewer weights when being applied to 3D problems, and correspondingly, FNOs are not recommended for 3D (or higher dimensional) problems.


# Attention and Transformers 

A newer and exciting develpoment in the deep learning field are attention mechanisms. They've been hugely successful in the form of _Transformers_ for processing language and natural images, and bear promise for physics-related problems. However, it's still open, whether they're really generally preferable over more "classic" architectures. The following section  will give an overview of the main pros and cons.

Transformers generally work in two steps: the input is encoded into _tokens_ with an encoder-decoder network. This step can take many forms, and usually primarily serves to reduce the number of inputs, e.g., to work with pieces of an image rather than individual pixels. The attention mechanism then computes a weighting for a collection if incoming tokens. This is a floating point number for each token, traditionally interpreted as indicating which parts of the input are important, and which aren't. In modern architectures, the floating point weighting of the attention are directly used to modify an input. In _self-attention_, the weighting is computed from each input towards all other input tokens. This is a mechanism to handle **global dependencies**, and hence directly fits into the discussion above. In practice, the attention is computed via three matrices: the query $Q$, the key matrix $K$, and a value matrix $V$. For $N$ tokens, the outer product $Q K^T$ produces an $N \times N$ matrix, and runs through a softmax layer, after which it is multiplied with $V$ (containing a linear projection of the input tokens) to produce the attention output vector. 

In a Transformer architecture, the attention output is used as component of a building block: the attention is calculated and used as a residual (added to the input), stabilized with a layer normalization, and then processed in a two-layer _feed forward_ network. The latter is simply a combination of two dense layers with an activation in between. This Transformer block is applied multiple times before the final output is decoded into the original space.

This Transformer architecture was shown to scale extremely well to networks with larger numbers of parameters, one of the key advantages of Transformers. Note that a large part of the weights typically ends up in the matrices of the attention, and not just in the dense layers. At the same time, attention offers a powerful way for working with global dependencies in inputs. This comes at the cost of a more complicated architecture, and an inherent difficulty of the self-attention mechanism above is that it's quadratic in the number of tokens $N$. This naturally puts a limit on the size and resolution of inputs. Under the hood, it's also surprisingly simple: the attention algorithm computes an $N \times N$ matrix, which is not too far from applying a simple dense layer (this would likewise come with an $N \times N$ weight matrix) to resolve global influences.

This bottleneck can be addressed with _linear attention_: it changes the algorithm above to multiply Q and (K^T V) instead, applying a non-linearity (e.g., an exponential) to both parts beforehand. This avoids the  $N \times N$ matrix and scales linearly in $N$. However, this improvement comes at the cost of a more approximative attention vector.

An interesting aspect of Transformer architectures is also that they've been applied to structured as well as unstructured inputs. I.e., they've been used for graphs, points as well as grid-based data. In all cases the differences primarily lie in how the tokens are constructed. The attention is typically still "dense" in the token space. This is a clear limitation: for problems with a known spatial structure, discarding this information will inevitably need to be compensated for, e.g., with a larger weight count or lower inference accuracy. Nonetheless, Transformers are an extremely active field within DL, and clearly a potential contender for future NN algorithms.


# Summary of Architectures

The paragraphs above have given an overview over several fundamental considerations when choosing a neural network architecture for a physics-related problem. To re-cap, the
main consideration when choosing an architecture is knowledge local or global dependencies in the data. Tailoring an architecture to this difference can have a big impact. 
And while the spatial structure of the data seems to dictate certain choices, it can be worth considering to transfer the data to another data structure. E.g., to project unstructured points onto a (deformed) regular grid to potentially improve accuracy and performance.

Also, it should be mentioned that hybrids of the _canonical_ architectures mentioned above exist: e.g., classic U-Nets with skip connections have been equipped with tricks for Transformer architectures (like attention and normalization) to yield an improved performance. E.g., an implementation of such a "modernized" U-Net can be found in {doc}`probmodels-time`.
