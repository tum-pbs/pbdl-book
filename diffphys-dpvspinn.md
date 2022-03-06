So Far so Good - a First Discussion
=======================

In the previous section we've seen an example reconstructions that used physical residuals as soft constraints, in the form of the variant 2 (PINNs), and reconstructions that used a differentiable physics (DP) solver. While both methods can find minimizers for similar inverse problems, the obtained solutions differ substantially, as does the behavior of the non-linear optimization problem that we get from each formulation. In the following we discuss these differences in more detail, and we will combine conclusions drawn from the behavior of the Burgers case of {doc}`physicalloss-code` and {doc}`diffphys-code-burgers` with observations from external research papers {cite}`holl2019pdecontrol`.

![Divider](resources/divider3.jpg)


## Compatibility with existing numerical methods

It is very obvious that the PINN implementation is quite simple, which is a positive aspect, but at the same time it differs strongly from "typical" discretizations and solution approaches that are usually employed to solve PDEs like Burgers equation. The derivatives are computed via the neural network, and hence rely on a fairly accurate representation of the solution to provide a good direction for optimization problems.

The DP version on the other hand inherently relies on a numerical solver that is tied into the learning process. As such it requires a discretization of the problem at hand, and via this discretization can employ existing, and potentially powerful numerical techniques. This means solutions and derivatives can be evaluated with known and controllable accuracy, and can be evaluated efficiently.

## Discretization

The reliance on a suitable discretization requires some understanding and knowledge of the problem under consideration. A sub-optimal discretization can impede the learning process or, worst case, lead to diverging training runs. However, given the large body of theory and practical realizations of stable solvers for a wide variety of physical problems, this is typically not an unsurmountable obstacle.

The PINN approaches on the other hand do not require an a-priori choice of a discretization, and as such seems to be "discretization-less". This, however, is only an advantage on first sight. As they yield solutions in a computer, they naturally _have_ to discretize the problem. They construct this discretization over the course of the training process, in a way that lies at the mercy of the underlying nonlinear optimization, and is not easily controllable from the outside. Thus, the resulting accuracy is determined by how well the training manages to estimate the complexity of the problem for realistic use cases, and how well the training data approximates the unknown regions of the solution.

E.g., as demonstrated with the Burgers example, the PINN solutions typically have significant difficulties propagating information _backward_ in time. This is closely coupled to the efficiency of the method.

## Efficiency

The PINN approaches typically perform a localized sampling and correction of the solutions, which means the corrections in the form of weight updates are likewise typically local. The fulfillment of boundary conditions in space and time can be correspondingly slow, leading to long training runs in practice.

A well-chosen discretization of a DP approach can remedy this behavior, and provide an improved flow of gradient information. At the same time, the reliance on a computational grid means that solutions can be obtained very quickly. Given an interpolation scheme or a set of basis functions, the solution can be sampled at any point in space or time given a very local neighborhood of the computational grid. Worst case, this can lead to slight memory overheads, e.g., by repeatedly storing mostly constant values of a solution.

For the PINN representation with fully-connected networks on the other hand, we need to make a full pass over the potentially large number of values in the whole network to obtain a sample of the solution at a single point. The network effectively needs to encode the full high-dimensional solution, and its size likewise determines the efficiency of derivative calculations.

## Efficiency continued

That being said, because the DP approaches can cover much larger solution manifolds, the structure of these manifolds is typically also difficult to learn. E.g., when training a network with a larger number of iterations (i.e. a long look-ahead into the future), this typically represents a signal that is more difficult to learn than a short look ahead. 

As a consequence, these training runs not only take more computational resources per NN iteration, they also often need longer to converge. Regarding resources, each computation of the look-ahead potentially requires a large number of simulation steps, and typically a similar amount of resources for the backpropagation step. Thus, while they may seem costly and slow to converge at times, this is usually caused by the the more complex signal that needs to be learned. 


![Divider](resources/divider2.jpg)


## Summary

The following table summarizes these pros and cons of physics-informed (PI) and differentiable physics (DP) approaches:

| Method   |  ✅ Pro   |  ❌ Con  |
|----------|-------------|------------|
| **PI** | - Analytic derivatives via backpropagation.  | - Expensive evaluation of NN, and very costly derivative calculations. | 
|          | - Easy to implement.  | - Incompatible with existing numerical methods.     | 
|          |                  | - No control of discretization.  | 
| | | |
| **DP** | - Leverage existing numerical methods. | - More complicated implementation.  | 
|          | - Efficient evaluation of simulation and derivatives. | - Require understanding of problem to choose suitable discretization. |
| | | |

As a summary, both methods are definitely interesting, and have a lot of potential. There are numerous more complicated extensions and algorithmic modifications that change and improve on the various negative aspects we have discussed for both sides.

However, as of this writing, the physics-informed (PI) approach has clear limitations when it comes to performance and compatibility with existing numerical methods. Thus, when knowledge of the problem at hand is available, which typically is the case when we choose a suitable PDE model to constrain the learning process, employing a differentiable physics solver can significantly improve the training process as well as the quality of the obtained solution. So, in the following we'll focus on DP variants, and illustrate their capabilities with more complex scenarios in the next chapters. First, we'll consider a case that very efficiently computes space-time gradients for a transient fluid simulations.
