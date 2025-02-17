Discussion of Differentiable Physics
=======================

The previous sections have explained the _differentiable physics_ approach for deep learning, and have given a range of examples: from a very basic gradient calculation, all the way to complex learning setups powered by advanced simulations. This is a good time to take a step back and evaluate: in the end, the differentiable physics components of these approaches are not too complicated. They are largely based on existing numerical methods, with a focus on efficiently using those methods not only to do a forward simulation, but also to compute gradient information. 
What is primarily exciting in this context are the implications that arise from the combination of these numerical methods with deep learning.

![Divider](resources/divider6.jpg)

## Integration

Most importantly, training via differentiable physics allows us to seamlessly bring the two fields together:
we can obtain _hybrid_ methods, that use the best numerical methods that we have at our disposal for the simulation itself, as well as for the training process. We can then use the trained model to improve forward or backward solves. Thus, in the end, we have a solver that combines a _traditional_ solver and a _learned_ component that in combination can improve the capabilities of numerical methods.

## Reducing data shift via interaction

One key aspect that is important for these hybrids to work well is to let the NN _interact_ with the PDE solver at training time. Differentiable simulations allow a trained model to "explore and experience" the physical environment, and receive directed feedback regarding its interactions throughout the solver iterations. 

This addresses the classic **data shift** problem of machine learning: rather than relying on a _a-priori_ specified distribution for training the network, the training process generates new trajectories via unrolling on the fly, and computes training signals from them. This can be seen as an _a-posteriori_ approach, and makes the trained NN significantly more resilient to unseen inputs. As we'll evaluate in more detail in {doc}`probmodels-uncond`, it's actually hard to beat a good unrolling setup with other approaches.

Note that the topic of _differentiable physics_ nicely fits into the broader context of machine learning as _differentiable programming_. 

## Generalization

The hybrid approach also bears particular promise for simulators: it improves generalizing capabilities of the trained models by letting the PDE-solver handle large-scale _changes to the data distribution_. This allows the learned model to focus on localized structures not captured by the discretization. While physical models generalize very well, learned models often specialize in data distributions seen at training time. Hence, this aspect benefits from the previous reduction of data shift, and effectively allows for even larger differences in terms of input distribution. If the NN is set up correctly, these can be handled by the classical solver in a hybrid approach.

These benefits were, e.g., shown for the models reducing numerical errors of {doc}`diffphys-code-sol`: the trained models can deal with solution manifolds with significant amounts of varying physical behavior, while simpler training variants would deteriorate over the course of recurrent time steps.


![Divider](resources/divider5.jpg)

To summarize, the pros and cons of training NNs via DP:

✅ Pro: 
- Uses physical model and numerical methods for discretization.
- Efficiency and accuracy of selected methods carries over to training.
- Very tight coupling of physical models and NNs possible.
- Improved resilience and generalization.

❌ Con: 
- Not compatible with all simulators (need to provide gradients).
- Require more heavy machinery (in terms of framework support) than previously discussed methods.

_Outlook_: the last negative point (regarding heavy machinery) is strongly improving at the moment. Many existing simulators, e.g. the popular open source framework _OpenFoma_, as well as many commercial simulators are working on tight integrations with NNs. However, there's still plenty room for improvement, and in this book we're focusing on examples using phiflow, which was designed for interfacing with deep learning frameworks from ground up. 

The training via differentiable physics (DP) allows us to integrate full numerical simulations into the training of deep neural networks. 
This effectively provides **hard constraints**, as the coupled solver can project and enforce constraints just like classical solvers would.
It is a very generic approach that is applicable to a wide range of combinations of PDE-based models and deep learning. 

In the next chapters, we will first expand the scope of the learning tasks to incorporate uncertainties, i.e. to work with full distributions rather than single deterministic states and trajectories. Afterwards, we'll also compare DP training to reinforcement learning, and target the underlying learning process to obtain even better NN states.

