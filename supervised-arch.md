Neural Network Architectures
=======================

The connectivity of the individual "neurons" in a neural network has a substantial influence on the capabilities of the network. Typical ones consist of a large number of these connected "neuron" units. Over the course of many years, several key architectures have emerged as particularly useful choices, and in the following we'll go over the main considerations for choosing an architecture. Our focus is to introduce ways of incorporating PDE-based models (the "physics"), rather than the subtleties of NN architectures.

```{figure} resources/arch01.jpg
---
height: 100px
name: arch01-overview
---
We'll discuss a range of architecture, from regular convolutions over graph- and particle-based convolutions to newer attention-based variants.
```

## Spatial Arrangement

A first, fundamental aspect to consider for choosing an architecture in the context of physics simulations 
(and for ruling out unsuitable options) is the spatial arrangement of the data samples. 
We can distinguish four main cases: 

1. No spatial arrangement,
2. A regular spacing on a grid (_structured_),
3. An irregular arrangement (_unstructured_), and
4. Irregular positions without connectivity (_particle_ / _point_-based).

For certain problems, there is no spatial information or arrangement (case 1). E.g., predicting the temporal evolution of temperature and pressure of a single measurement probe over time does not involve any spatial dimension. The opposite case are probes placed on a nicely aligned grid (case 2), or at arbitrary locations (3). The last variant (case 4) is a slightly special case of the third one, where no clear or persistent links between sample points are known. This can, e.g., be the case for particle-based representations of turbulent fluids, where neighborhood change quickly over time.

## No spatially arranged inputs

The first case is a somewhat special one: without any information about spatial arrangements, only _dense_ ("fully connected" / MLP) neural networks are applicable. 

If you decide to use a **neural fields** approach where the network receives the position as input, this has the same effect: the NN will not have any direct means of querying neighbors via architectural tricks ("inductive biases"). In this case, the building blocks below won't be applicable, and it's worth considering whether you can introduce more structure via a discretization.

Note that _physics-informed neural networks_ (PINNs) also fall into this category. We'll go into more detail here later on ({doc}`diffphys`), but generally it's advisable to consider switching to an approach that employs prior knowledge in the form of a discretization. This usually substantially improves inference accuracy and improves convergence. That PINNs can't solve real-world problems despite many years of research points to the fundamental problems of this approach.

Focusing on dense layers still leaves a few choices concerning the number of layers, their size, and activations. The other three cases have the same choices, and these hyperparameters of the architectures are typically determined over the course of training runs. General recommendations are that _ReLU_ and smoother variants like _GELU_ are good choices, and that the number of layers should scale together with their size.
Next, we'll focus on the remaining three cases with spatial information in the following, as differences can have a profound impact here. So, below we target cases where we have a "computational domain" specifying the region of interest in which the samples are located.

## Local vs Global

Probably the most important aspect of different architectures then is the question of their _receptive field_: this means for any single sample in our domain, which neighborhood of other sample points can influence the solution at this point. This is similar to classic considerations for PDE solving, where denoting a PDE as _hyperbolic_ indicates its local, wave-like behavior in contrast to an _elliptic_ one with global behavior. Certain NN architectures such as the classic convolutional neural networks (CNNs) support only local influences and receptive fields, while hierarchies with pooling expand these receptive field to effectively global ones. An interesting variant here are spectral architectures like FNOs, which provide global receptive fields at the expense of other aspects. In addition Transformers (with attention mechanisms), provide a more complicated but scalable alternative. 

Thus, a fundamental distinction can be made in terms of spatially local vs global architectures, and for the latter, how they realize the global receptive field. The following table provides a first overview, and below we'll discuss the pros and cons of each variant.

|             | Grid            | Unstructured      | Points            | Non-spatial |
|-------------|-----------------|-------------------|-------------------|-------------|
| **Local**   | CNN , ResNet    | GNN               | CConv             | -           |
| **Global**  |                 |                   |                   |  MLP        |
| - Hierarchy | U-Net, Dilation | Multi-scale GNN   | Multi-scale CConv | -           |
| - Spectral  | FNO             | Spectral GNN      | (-)               | -           |
| - Sequence  | Transformer     | Graph Transformer | Point Trafo.      | -           |



```{note}
Knowledge about the dependencies in your data, i.e., whether the dependencies are local or global, is important knowledge that should be leveraged. 

If your data has primarily **local** influences, choosing an architecture with support for global receptive fields will most likely have negative effects on accuracy: the network will "waste" resources trying to capture global effects, or worst case approximate local effects with smoothed out global modes.

Vice versa, trying to approximate a **global** influence with a limited receptive field will be an unsolvable task, and most likely introduce substantial errors.
```

## Regular, unstructured and point-wise data

The most natural start for making use of spatially arranged data is to employ a regular grid. Note that it doesn't have to be a
Cartesian grid, but could be e deformed and adaptive grid {cite}`chen2021highacc`. The only requirement is a grid-like connectivity of the samples, even if 
they have an irregular spacing. 

```{figure} resources/arch05.jpg
---
height: 180px
name: arch04-defogrids-conv
---
A 3x3 convolution (orange) shown for differently deformed regular multi-block grids.
```

For unstructured data, graph-based neural networks (GNNs) are a good choice. While they're often discussed in terms of _message-passing_ operations,
they share a lot of similarities with structured grids: the basic operation of a message-passing step on a GNN is equivalent to a convolution on a grid {cite}`sanchez2020learning`. 
Hierarchies can likewise be constructed by graph coarsening {cite}`lino2025dgn`. Hence, while we'll primarily discuss grids below, keep in mind that the approaches carry over to GNNs. As dealing with graph structures makes the implementation more complicated, we won't go into details until later.

```{figure} resources/arch02.jpg
---
height: 240px
name: arch02-convolutions
---
Convolutions (and hierarchies) work very similarly irrespective of the structure of the data. Convolutions apply to grids, graphs and point-samples, as shown above. 
Likewise, the concepts discussed for grid-based algorithms in this book carry over to graphs and point collections.
```

Finally, point-wise (Lagrangian) samples can be seen as unstructured grids without connectivity. However, it can be worth explicitly treating
them in this way for improved learning and inference performance. Nonetheless, the two main ideas of convolutions and hierarchies carry over
to Lagrangian data: continuous convolution kernels are a suitable tool, and neighborhood based coarsening yields hierarchies {cite}`prantl2022guaranteed`.

## Hierarchies

A powerful and natural tool to work with **local** dependencies are convolutional layers on regular grids. The corresponding neural networks (CNNs) are 
a classic building block of deep learning, and very well researched and supported throughout. They are comparatively easy to 
train, and usually very efficiently implemented in APIs. They also provide a natural connection to classical numerics:  discretizations
of differential operators such as gradient and Laplacians are often thought of in terms of "stencils", which are an equivalent of
a convolutional layer with a set of specific weights. E.g., consider the classic stencil for a normalized Laplacian $\nabla^2$ in 1D: $[1, -2, 1]$. 
It can directly be mapped to a 1D convolution with kernel size 3 and a single input and output channel. The non trainable weights of the kernel
can be set to the coefficients of the Laplacian stencil above.

Using convolutional layers is quite straight forward, but the question of how to incorporate **global** dependencies into 
CNNs is an interesting one. Over time, two fundamental approaches have been established here in the field:
_hierarchical_ networks via pooling (U-Nets {cite}`ronneberger2015unet`), and sparse, point wise samples with enlarged spacing (Dilation {cite}`yu2015dilate`). They both reach
the goal of establishing a global receptive field, but have a few interesting differences under the hood.

```{figure} resources/arch03.jpg
---
height: 200px
name: arch03-hierarchy
---
A 3x3 convolution shown for a pooling-based hierarchy (left), and a dilation-based convolution (right). Not that in both cases the convolutions cover larger ranges of the input data. However, the hierarchy processes $O(log N)$ less data, while the dilation processes the full input with larger strides. Hence the latter has an increased cost due to the larger number of sample points, and the less regular data access.
```

* U-Nets are based on _pooling_ operations. Akin to a geometric multigrid hierarchy, the spatial samples are downsampled to coarser and
coarser grids, and upsampled in the later half of the network. This means that even if we keep the kernel size of a convolution fixed, the 
convolution will be able to "look" further in terms of physical space due to the previous downsampling operation. 
The number of sample points decreases logarithmically, making convolutions on lower hierarchy levels very efficient.
While the different re-sampling methods (mean, average, point-wise ...) have a minor effect, a crucial ingredient for U-Net are _skip connection_. They connect
the earlier layers of the first half directly with the second half via feature concatenation. This turns out to be crucial to avoid 
a loss of information. Typically, the deepest "bottle-neck" layer with the coarsest representation has trouble storing all details 
of the finest one. Providing this information explicitly via a skip-connection is crucial for improving accuracy.

* Dilation in the form of _dilated convolutions_ places the sampling points for convolutions further apart. Hence instead of, e.g., looking at a 3x3 neighborhood, 
a convolution considers a 5x5 neighborhood but only includes 3x3 samples when calculating the convolution. The other samples in-between the used points are
typically simply ignored. In contrast to a hierarchy, the number of sample points remains constant.

While both approaches reach the goal, and can perform very well, there's an interesting tradeoff: U-Nets take a bit more effort to implement, but can be much faster. The reason for the performance boost is the sub-optimal memory access of the dilated convolutions: they skip through memory with a large stride, which gives a slower performance. The U-Nets, on the other had, basically precompute a compressed memory representation in the form of a coarse grid. Convolutions on this coarse grid are then highly efficient to compute. However, this requires slightly more effort to implement in the form of adding appropriate pooling layers (dilated convolutions can be as easy to implement as replacing the call to the regular convolution with a dilated one). The implementation effort of a U-Net can pay off significantly in the long run, when a trained network should be deployed in an application.

As mentioned above hierarchies are likewise important for graph nets. However, the question whether to "dilate or not" is not present for graph nets: here the memory access is always irregular, and dilation is unpopular as the strides would be costly to compute on general graphs. Hence, regular hierarchies in the form of multi-scale GNNs are highly recommended if global dependencies exist in the data.


## Spectral methods

A fundamentally different avenue for establishing global receptive fields is provided by spectral methods, typically making use of Fourier transforms to transfer spatial data to the frequency domain. The most popular approach from this class of methods are _Fourier Neural Operators_ (FNOs) {cite}`li2021fno`. An interesting aspect is the promise of a continuous representation via the functional representation, where a word of caution is appropriate: the function spaces are typically truncated, so it is often questionable whether the frequency representation really yields suitable solutions beyond the resolution of the training data.

In the following, however, we'll focus on the aspect of receptive fields in conjunction with performance aspects. Here, FNO-like methods have an interesting behavior: they modify frequency information with a dense layer. As the frequency signal after a Fourier transform would have the same size as the input, the dense layer works on a set of the $M$ largest frequencies. For a two dimensional input that means $M^2$ modes, and the corresponding dense layer thus requires $M^4$ parameters.

An inherent advantage and consequence of the frequency domain is that all basis functions have global support. That means despite only working with a subset of all frequencies, FNOs can process (and modify) all parts of an input signal. This natural treatment of **global dependencies** is a inherent advantage of spectral methods.

```{figure} resources/arch06-fno.jpg
---
height: 200px
name: arch06-fno
---
Spatial convolutions (left, kernel in orange) and frequency processing in FNOs (right, coverage of dense layer in yellow). Not only do FNOs scale less well in 3D (**6th** instead of 5th power), their scaling constant is also proportional to the domain size, and hence typically larger.
```

Unfortunately, they're not well suited for higher dimensional problems: Moving from two to three dimensions increases the size of the frequencies to be handled to $M^3$. For the dense layer, this means $M^6$ parameters, a cubic increase. For convolutions, there's no huge difference in 2D:
 a regular convolution with kernel size $K$ requires $K^2$ weights in 2D, and induces another $O(K^2)$ scaling for processing features, in total $O(K^4 N^2)$ for a domain of sie $N^2$.
However, as $K<<N$, regular convolutions scale much better in 3D: the kernel size increases to $K^3$, giving an overall complexity of $O(K^5 N^3)$ for a 3D domain with side length $N$. 

The frequency coverage $M$ of FNOs needs to scale with the size of the spatial domain, hence typically $M>K$ and $M^6 \gg K^5$. 
Thus, as $K$ is typically much smaller than $N$ and $M$, and scales with an exponent of 5, CNNs will usually scale much better than FNOs with their 6th power scaling.
They would require intractable amounts of parameters to capture finer features, and are thus not recommendable for 3D (or higher dimensional) problems. CNN-based architectures require much fewer weights, and in conjunction with hierarchies can still handle global dependencies efficiently.

<br>

![Divider](resources/divider2.jpg)

## Attention and Transformers 

A newer and exciting development in the deep learning field are attention mechanisms. They've been hugely successful in the form of _Transformers_ for processing language and natural images, and bear promise for physics-related problems. However, it's still open, whether they're really generally preferable over more "classic" architectures. The following section will give an overview of the main pros and cons.

Transformers generally work in two steps: the input is encoded into _tokens_ with an encoder-decoder network. This step can take many forms, and usually primarily serves to reduce the number of inputs, e.g., to work with pieces of an image rather than individual pixels. The attention mechanism then computes a weighting for a collection if incoming tokens. This is a floating point number for each token, traditionally interpreted as indicating which parts of the input are important, and which aren't. In modern architectures, the floating point weighting of the attention are directly used to modify an input. In _self-attention_, the weighting is computed from each input towards all other input tokens. This is a mechanism to handle **global dependencies**, and hence directly fits into the discussion above. In practice, the attention is computed via three matrices: the query $Q$, the key matrix $K$, and a value matrix $V$. For $N$ tokens, the outer product $Q K^T$ produces an $N \times N$ matrix, and runs through a Softmax layer, after which it is multiplied with $V$ (containing a linear projection of the input tokens) to produce the attention output vector. 

In a Transformer architecture, the attention output is used as component of a building block: the attention is calculated and used as a residual (added to the input), stabilized with a layer normalization, and then processed in a two-layer _feed forward_ network (FFN). The latter is simply a combination of two dense layers with an activation in between. This _Transformer block_, summarized below visually, is applied multiple times before the final output is decoded into the original space.

```{figure} resources/overview-arch-tblock.jpg
---
height: 150px
name: overview-arch-transformer-block
---
Visual summary of a single transformer block. A full network repeats this structure several times to infer the result.
```


This Transformer architecture was shown to scale extremely well to networks with huge numbers of parameters, one of the key advantages of Transformers. Note that a large part of the weights typically ends up in the matrices of the attention, and not just in the dense layers. At the same time, attention offers a powerful way for working with global dependencies in inputs. This comes at the cost of a more complicated architecture. An inherent problem of the self-attention mechanism above is that it's quadratic in the number of tokens $N$. This naturally puts a limit on the size and resolution of inputs. Under the hood, it's also surprisingly simple: the attention algorithm computes an $N \times N$ matrix, which is not too far from applying a simple dense layer (this would likewise come with an $N \times N$ weight matrix) to resolve global influences.

This bottleneck can be addressed with _linear attention_: it changes the algorithm above to multiply $Q$ and ($K^T V$) instead, applying a non-linearity (e.g., an exponential) to both parts beforehand. This avoids the  $N \times N$ matrix and scales linearly in $N$. However, this improvement comes at the cost of a more approximate attention vector.

An interesting aspect of Transformer architectures is also that they've been applied to structured as well as unstructured inputs. I.e., they've been used for graphs, points as well as grid-based data. In all cases the differences primarily lie in how inputs are mapped to the tokens. The attention is typically still "dense" in the token space. This is a clear limitation: for problems with a known spatial structure, discarding this information will inevitably need to be compensated for, e.g., with a larger weight count or lower inference accuracy. Nonetheless, Transformers are an extremely active field within DL, and clearly a potential contender for future NN algorithms.


![Divider](resources/divider7.jpg)


## Summary of Architectures

The paragraphs above have given an overview over several fundamental considerations when choosing a neural network architecture for a physics-related problem. To re-cap, the
main consideration when choosing an architecture is knowledge about **local** or **global** dependencies in the data. Tailoring an architecture to this difference can have a big impact. 
And while the spatial structure of the data seems to dictate choices, it can be worth considering to transfer the data to another data structure. E.g., to project unstructured points onto a (deformed) regular grid to potentially improve accuracy and performance.

Also, it should be mentioned that hybrids of the _canonical_ architectures mentioned above exist: e.g., classic U-Nets with skip connections have been equipped with components of Transformer architectures (like attention and normalization) to yield an improved performance. An implementation of such a "modernized" U-Net can be found in {doc}`probmodels-time`.

## Show me some code!

Let's finally look at a code example that trains a neural network: we'll replace a full solver for _turbulent flows around airfoils_ with a surrogate model from {cite}`thuerey2020dfp` using a U-Net with a global receptive field as operator. 
