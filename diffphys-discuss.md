Discussion
=======================

In the previous sections we've seen example reconstuctions that used physical residuals as soft constraints, in the form of the PINNs, and reconstuctions that used a differentiable physics (DP) solver. While both methods can find minimizers for the same minimization problem, the solutions the obtained differ substantially, as do the behavior of the non-linear optimization problem that we get from each formulation. In the following we discuss these differences in more detail, and we will combine conclusions drawn from the behavior of the Burgers case of the previous sections with observations from research papers.

## Compatibility with Existing Numerical Methods

It is very obvious that the PINN implementation is quite simple, which is a positive aspect, but at the same time it differs strongly from "typical" discretziations and solution approaches that are usually to employed equations like Burgers equation. The derivatives are computed via the neural network, and hence rely on a fairly accurate representation of the solution to provide a good direction for optimizaion problems.

The DP version on the other hand inherently relies on a numerical solver that is tied into the learning process. As such it requires a discretization of the problem at hand, and given this discretization can employ existing, and potentially powerful numerical techniques. This means solutions and derivatives can be evaluated with known and controllable accuracy, and can be evaluated efficiently.

## Discretization

The reliance on a suitable discretization requires some understanding and knowledge of the problem under consideration. A sub-optimal discretization can impede the learning prcoess or, worst case, lead to diverging trainig runs. However, given the large body of theory and practical realizations of stable solvers for a wide variety of physical problems, this is typically not an unsurmountable obstacle.

The PINN approaches on the other hand do not require an a-priori choice of a discretization, and as such seem to be "discretization less". This, however, is only an advantage on first sight. As they yield solutions in a computer, they naturally _have_ to discretize the problem, but they construct this discretization over the coure of the training process, in a way that is not easily controllable from the outside. Thus, the resulting accuracy is determined by how well the training manages to estimate the complexity of the problem for realistic use cases, and how well the training data approximates the unknown regions of the solution.

As demonstrated with the Burgers example, the PINN solutions typically have significant difficulties propagating information _backward_ in time. This is closely coupled to the efficiency of the method.

## Efficiency

The PINN approach typically perform a localized sampling and correction of the solutions, which means the corrections in the form of weight updates are likewise typically local. The fulfilment of boundary conditions in space and time can be correspondingly slow, leading to long training runs in practice.

A well-chosen discretization of a DP approach can remedy this behavior, and provide an improved flow of gradient information. At the same time, the reliance on a computational grid means that solutions can be obtained very quickly. Given an interpolation scheme or set of basis functions, the solution can be sampled at any point in space or time given a very local neighborhood of the computational grid. Worst case, this can lead to slight memory overheads, e.g., by repeatedly storing mostly constand values of a solution.

For the PINN representation with fully-connected networks on the other hand, we need to make a full pass over the potentially large number of values in the whole network to obtain a sample of the solution at a single point. The network effectively needs to encode the full high-dimensional solution. Its size likewise determines the efficiency of derivative calculations.

## Summary

The following table summarizes these findings:

| Method   |  Pro   |  Con  |
|----------|-------------|------------|
| **PINN** | - Analytic derivatives via back-propagation  | - Expensive evaluation of NN, as well as derivative calculations | 
|          | - Simple to implement  | - Incompatible with existing numerical methods     | 
|          |                  | - No control of discretization  | 
| **DiffPhys** | - Leverage existing numerical methods | - More complicated to implement  | 
|          | - Efficient evaluation of simulation and derivatives | - Require understanding of problem to choose suitable discretization |

As a summary, both methods are definitely interesting, and leave a lot of room for improvement with more complicated extensions and algorithmic modifications that change and improve on the various negative aspects we have discussed for both sides.

However, as of this writing, the physics-informed (PI) approach has clear limitations when it comes to performance and campatibility with existing numerical methods. Thus, when knowledge of the problem at hand is available, which typically is the case when we choose a suitable PDE model to constrain the learning process, employing a differentiable physics (DP) solver can significantly improve the training process as well as the quality of the obtained solution. Next, we will target a more setting, i.e., fluids with Navier Stokes, to illustrate this behavior.
