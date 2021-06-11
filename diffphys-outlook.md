Summary and Discussion
=======================

The previous sections have explained the _differentiable physics_ approach for deep learning, and have given a range of examples: from a very basic gradient calculation, all the way to complex learning setups powered by advanced simulations. This is a good time to take a step back and evaluate: in the end, the differentiable physics components of these approaches are not too complicated. They are largely based on existing numerical methods, with a focus on efficiently using those methods not only to do a forward simulation, but also to compute gradient information. What is primarily exciting in this context are the implications that arise from the combination of these numerical methods with deep learning.

![Divider](resources/divider6.jpg)

## Integration

Most importantly, training via differentiable physics allows us to seamlessly bring the two fields together:
we can obtain _hybrid_ methods, that use the best numerical methods that we have at our disposal for the simulation itself, as well as for the training process. We can then use the trained model to improve forward or backward solves. Thus, in the end, we have a solver that combines a _traditional_ solver and a _learned_ component that in combination can improve the capabilities of numerical methods.

## Interaction

One key aspect that is important for these hybrids to work well is to let the NN _interact_ with the PDE solver at training time. Differentiable simulations allow a trained model to "explore and experience" the physical environment, and receive directed feedback regarding its interactions throughout the solver iterations. This combination nicely fits into the broader context of machine learning as _differentiable programming_. 

## Generalization

The hybrid approach also bears particular promise for simulators: it improves generalizing capabilities of the trained models by letting the PDE-solver handle large-scale _changes to the data distribution_ such that the learned model can focus on localized structures not captured by the discretization. While physical models generalize very well, learned models often specialize in data distributions seen at training time. This was, e.g., shown for the models reducing numerical errors of the previous chapter: the trained models can deal with solution manifolds with significant amounts of varying physical behavior, while simpler training variants quickly deteriorate over the course of recurrent time steps.

![Divider](resources/divider7.jpg)

Training NNs via differentiable physics solvers is a very generic approach that is applicable to a wide range of combinations of PDE-based models and deep learning. Nonetheless, the next chapters will discuss several variants that are orthogonal to the general DP version, or can yield benefits in more specialized settings.

