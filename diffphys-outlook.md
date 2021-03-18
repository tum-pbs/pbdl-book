Summary and Discussion
=======================

The previous sections have explained the differentiable physics approach for deep learning, and have given a range of examples: from a very basic gradient calculation, all the way to complex learning setups powered by simulations. This is a good time to pause and take a step back, to take a look at what we have: in the end, the _differentiable physics_ part is not too complicated. It's largely based on existing numerical methods, with a focus on efficiently using those methods to not only do a forward simulation, but also to compute gradient information. What's more exciting is the combination of these methods with deep learning. 

## Integration

Most importantly, training via differentiable physics allows us to seamlessly bring the two fields together:
we can obtain _hybrid_ methods, that use the best numerical methods that we have at our disposal for the simulation itself, as well as for the training process. We can then use the trained model to improve the forward or backward solve. Thus, in the end we have a solver that employs both a _traditional_ solver and a _learned_ component.

## Interaction

One key component for these hybrids to work well is to let the NN _interact_ with the PDE solver at training time. Differentiable simulations allow a trained model to explore and experience the physical environment, and receive directed feedback regarding its interactions throughout the solver iterations. This combination nicely fits into the broader context of machine learning as _differentiable programming_. 

## Generalization

The hybrid approach also bears particular promise for simulators: it improves generalizing capabilities of the trained models by letting the PDE-solver handle large-scale _changes to the data distribution_ such that the learned model can focus on localized structures not captured by the discretization. While physical models generalize very well, learned models often specialize in data distributions seen at training time. This was, e.g., shown for the models reducing numerical errors of the previous chapter: the trained models can deal with solution manifolds with significant amounts of varying physical behavior, while simpler training variants quickly deteriorate over the course of recurrent time steps.

---

Despite being a very powerful method, the DP approach is clearly not the end of the line. In the next chapters we'll consider further improvements and extensions.

