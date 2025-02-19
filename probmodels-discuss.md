Discussion of Probabilistic Learning
=======================

As the previous sections have demonstrated, probabilistic learning offers a wide range of very exciting possibilities in the context of physics-based learning. First, these methods come with a highly interesting and well developed theory. Surprisingly, some parts are actually more developed than basic questions about simpler learning approaches.

At the same time, they enable a fundamentally different way to work with simulations: they provide a simple way to work with complex distributions of solutions. This is of huge importance for inverse problems, e.g. in the context of obtaining likelihood-based estimates for _simulation-based inference_. 

That being said, diffusion based approaches will not show relatively few advantages for deterministic settings: they are not more accurate, and typically induce slightly larger computational costs. An interesting exception is the long-term stability, as discussed in {doc}`probmodels-uncond`. 

![Divider](resources/divider1.jpg)

To summarize the key aspects of probabilistic deep learning approaches:

✅ Pro: 
- Enable training and inference for distributions
- Well developed theory
- Stable training

❌ Con: 
- (Slightly) increased inference cost
- No real advantage for deterministic settings

![Divider](resources/divider7.jpg)

To summarize: if your problems contains ambiguities, diffusion modeling in the form of _flow matching_ is the method of choice. If your data contains reliable input-output pairs, go with simpler _deterministic training_ instead.

Next, we can turn to a new viewpoint on learning problems, the field of _reinforcement learning_. As the next sections will point out, it is actually not so different from the topics of the previous chapters despite the new viewpoint.

