Discussion of Supervised Approaches
=======================

The previous example illustrates that we can quite easily use 
supervised training to solve quite complex tasks. The main workload is
collecting a large enough dataset of examples. Once that exists, we can
train a network to approximate the solution manifold sampled
by these solutions, and the trained network can give us predictions
very quickly. There are a few important points to keep in mind when 
using supervised training.

## Some things to keep in mind...

### Natural starting point

_Supervised training_ is the natural starting point for **any** DL project. It always,
and we really mean **always** here, makes sense to start with a fully supervised
test using as little data as possible. This will be a pure overfitting test,
but if your network can't quickly converge and give a very good performance 
on a single example, then there's something fundamentally wrong
with your code or data. Thus, there's no reason to move on to more complex
setups that will make finding these fundamental problems more difficult.

Hence: **always** start with a 1-sample overfitting test,
and then increase the complexity of the setup.

### Stability

A nice property of the supervised training is also that it's very stable.
Things won't get any better when we include more complex physical 
models, or look at more complicated NN architectures.

Thus, again, make sure you can see a nice exponential falloff in your training 
loss when starting with the simple overfitting tests. This is a good
setup to figure out an upper bound and reasonable range for the learning rate
as the most central hyperparameter.
You'll probably need to reduce it later on, but you should at least get a 
rough estimate of suitable values for $\eta$.

### Where's the magic? 🦄 

A comment that you'll often hear when talking about DL approaches, and especially
when using relatively simple training methodologies is: "Isn't it just interpolating the data?"

Well, **yes** it is! And that's exactly what the NN should do. In a way - there isn't 
anything else to do. This is what _all_ DL approaches are about. They give us smooth
representations of the data seen at training time. Even if we'll use fancy physical 
models at training time later on, the NNs just adjust their weights to represent the signals
they receive, and reproduce it.

Due to the hype and numerous success stories, people not familiar with DL often have 
the impression that DL works like a human mind, and is able to detect fundamental
and general principles in data sets (["messages from god"](https://dilbert.com/strip/2000-01-03) anyone?).
That's not what happens with the current state of the art. Nonetheless, it's
the most powerful tool we have to approximate complex, non-linear functions.
It is a great tool, but it's important to keep in mind, that once we set up the training
correctly, all we'll get out of it is an approximation of the function the NN
was trained for - no magic involved.

An implication of this is that you shouldn't expect the network 
to work on data it has never seen. In a way, the NNs are so good exactly 
because they can accurately adapt to the signals they receive at training time,
but in contrast to other learned representations, they're actually not very good
at extrapolation. So we can't expect an NN to magically work with new inputs.
Rather, we need to make sure that we can properly shape the input space,
e.g., by normalization and by focusing on invariants. In short, if you always train
your networks for inputs in the range $[0\dots1]$, don't expect it to work
with inputs of $[10\dots11]$. You might be able to subtract an offset of $10$ beforehand,
and re-apply it after evaluating the network.
As a rule of thumb: always make sure you
actually train the NN on the kinds of input you want to use at inference time.

This is important to keep in mind during the next chapters: e.g., if we
want an NN to work in conjunction with another solver or simulation environment,
it's important to actually bring the solver into the training process, otherwise
the network might specialize on pre-computed data that differs from what is produced
when combining the NN with the solver, i.e _distribution shift_.

### Meshes and grids

The previous airfoil example use Cartesian grids with standard 
convolutions. These typically give the most _bang-for-the-buck_, in terms
of performance and stability. Nonetheless, the whole discussion here of course 
also holds for less regular convolutions, e.g., a less regular mesh
in conjunction with graph-convolutions. You will typically see reduced learning
performance in exchange for improved stability when switching to these.

Finally, a word on fully-connected layers or _MLPs_ in general: we'd recommend
to avoid these as much as possible. For any structured data, like spatial functions,
or _field data_ in general, convolutions are preferable, and less likely to overfit.
E.g., you'll notice that CNNs typically don't need dropout, as they're nicely
regularized by construction. For MLPs, you typically need quite a bit to
avoid overfitting.

---

## Supervised Training in a nutshell

To summarize:

✅ Pros: 
- very fast training
- stable and simple
- great starting point

❌ Con: 
- lots of data needed
- sub-optimal performance, accuracy and generalization

Outlook: interactions with external "processes" (such as embedding into a solver) are tricky with supervised training.
First, we'll look at bringing model equations into the picture via soft-constraints, and afterwards
we'll revisit the challenges of bringing together numerical simulations and learned approaches.


