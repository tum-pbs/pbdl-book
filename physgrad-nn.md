Physical Gradients and NNs
=======================

The discussion in the previous two sections already hints at physical gradients (PGs) being a powerful tool for optimization. However, we've actually cheated a bit in the previous code example {doc}`physgrad-comparison` and used PGs in a way that will be explained in more detail below. 

By default, PGs would be restricted to functions with square Jacobians. Hence we wouldn't be able to directly use them in optimizations or learning problems, which typically have scalar objective functions.
In this section, we will first show how PGs can be integrated into the optimization pipeline to optimize scalar objectives.

## Physical Gradients and loss functions

As before, we consider a scalar objective function $L(y)$ that depends on the result of an invertible simulator $y = \mathcal P(x)$. In {doc}`physgrad` we've outlined the inverse gradient (IG) update $\Delta x = \frac{\partial x}{\partial L} \cdot \Delta L$, where $\Delta L$ denotes a step to take in terms of the loss. 

By applying the chain rule and substituting the IG $\frac{\partial x}{\partial L}$ for the PG, we obtain 

$$
\begin{aligned}
    \Delta x
    &= \frac{\partial x}{\partial L} \cdot \Delta L
    \\
    &= \frac{\partial x}{\partial y} \left( \frac{\partial y}{\partial L} \cdot \Delta L \right)
    \\
    &= \frac{\partial x}{\partial y} \cdot \Delta y
    \\
    &= \mathcal P^{-1}_{(x_0,y_0)}(y_0 + \Delta y) - x_0 + \mathcal O(\Delta y^2)
    .
\end{aligned}
$$

This equation has turned the step w.r.t. $L$ into a step in $y$ space: $\Delta y$. 
However, it does not prescribe a unique way to compute $\Delta y$ since the derivative $\frac{\partial y}{\partial L}$ as the right-inverse of the row-vector $\frac{\partial L}{\partial y}$ puts almost no restrictions on $\Delta y$.
Instead, we use a Newton step (equation {eq}`quasi-newton-update`) to determine $\Delta y$ where $\eta$ controls the step size of the optimization steps.

Here an obvious questions is: Doesn't this leave us with the disadvantage of having to compute the inverse Hessian, as discussed before?
Luckily, unlike with regular Newton or quasi-Newton methods, where the Hessian of the full system is required, here, the Hessian is needed only for $L(y)$. Even better, for many typical $L$ its computation can be completely forgone.

E.g., consider the case $L(y) = \frac 1 2 || y^\textrm{predicted} - y^\textrm{target}||_2^2$ which is the most common supervised objective function.
Here $\frac{\partial L}{\partial y} = y^\textrm{predicted} - y^\textrm{target}$ and $\frac{\partial^2 L}{\partial y^2} = 1$.
Using equation {eq}`quasi-newton-update`, we get $\Delta y = \eta \cdot (y^\textrm{target} - y^\textrm{predicted})$ which can be computed without evaluating the Hessian.

Once $\Delta y$ is determined, the gradient can be backpropagated to earlier time steps using the inverse simulator $\mathcal P^{-1}$. We've already used this combination of a Newton step for the loss and PGs for the PDE in {doc}`physgrad-comparison`.


## NN training 

The previous step gives us an update for the input of the discretized PDE $\mathcal P^{-1}(x)$, i.e. a $\Delta x$. If $x$ was an output of an NN, we can then use established DL algorithms to backpropagate the desired change to the weights of the network.
We have a large collection of powerful methodologies for training neural networks at our disposal, 
so it is crucial that we can continue using them for training the NN components.
On the other hand, due to the problems of GD for physical simulations (as outlined in {doc}`physgrad`),  
we aim for using PGs to accurately optimize through the simulation.

Consider the following setup: 
A neural network makes a prediction $x = \mathrm{NN}(a \,;\, \theta)$ about a physical state based on some input $a$ and the network weights $\theta$.
The prediction is passed to a physics simulation that computes a later state $y = \mathcal P(x)$, and hence
the objective $L(y)$ depends on the result of the simulation. 


```{admonition} Combined training algorithm
:class: tip

To train the weights $\theta$ of the NN, we then perform the following updates:

* Evaluate $\Delta y$ via a Newton step as outlined above
* Compute the PG $\Delta x = \mathcal P^{-1}_{(x, y)}(y + \Delta y) - x$ using an inverse simulator
* Use GD or a GD-based optimizer to compute the updates to the network weights, $\Delta\theta = \eta_\textrm{NN} \cdot \frac{\partial y}{\partial\theta} \cdot \Delta y$

```

The combined optimization algorithm depends on both the **learning rate** $\eta_\textrm{NN}$ for the network as well as the step size $\eta$ from above, which factors into $\Delta y$.
To first order, the effective learning rate of the network weights is $\eta_\textrm{eff} = \eta \cdot \eta_\textrm{NN}$.
We recommend setting $\eta$ as large as the accuracy of the inverse simulator allows, before choosing $\eta_\textrm{NN} = \eta_\textrm{eff} / \eta$ to achieve the target network learning rate.
This allows for nonlinearities of the simulator to be maximally helpful in adjusting the optimization direction.


**Note:**
For simple objectives like a loss of the form $L=|y - y^*|^2$, this procedure can be easily integrated into an  GD autodiff pipeline by replacing the gradient of the simulator only.
This gives an effective objective function for the network

$$
L_\mathrm{NN} = \frac 1 2  | x - \mathcal P_{(x,y)}^{-1}(y + \Delta y) |^2
$$

where $\mathcal P_{(x,y)}^{-1}(y + \Delta y)$ is treated as a constant.


## Iterations and time dependence

The above procedure describes the optimization of neural networks that make a single prediction.
This is suitable for scenarios to reconstruct the state of a system at $t_0$ given the state at a $t_e > t_0$ or to estimate an optimal initial state to match certain conditions at $t_e$.

However, our method can also be applied to more complex setups involving multiple objectives at different times and multiple network interactions at different times. 
Such scenarios arise e.g. in control tasks, where a network induces small forces at every time step to reach a certain physical state at $t_e$. It also occurs in correction tasks where a network tries to improve the simulation quality by performing corrections at every time step.

In these scenarios, the process above (Newton step for loss, PG step for physics, GD for the NN) is iteratively repeated, e.g., over the course of different time steps, leading to a series of additive terms in $L$.
This typically makes the learning task more difficult, as we repeatedly backpropagate through the iterations of the physical solver and the NN, but the PG learning algorithm above extends to these case just like a regular GD training.

## Time reversal

The inverse function of a simulator is typically the time-reversed physical process.
In some cases, simply inverting the time axis of the forward simulator, $t \rightarrow -t$, can yield an adequate global inverse simulator.
%
Unless the simulator destroys information in practice, e.g., due to accumulated numerical errors or stiff linear systems, this straightforward approach is often a good starting point for an inverse simulation, or to formulate a _local_ inverse simulation.


---

## A learning toolbox

Taking a step back, what we have here is a flexible "toolbox" for propagating update steps
through different parts of a system to be optimized. An important takeaway message is that
the regular gradients we are working with for training NNs are not the best choice when PDEs are 
involved. In these situations we can get much better information about how to direct the
optimization than the localized first-order information that regular gradients provide.

Above we've motivated a combination of inverse simulations, Newton steps, and regular gradients.
In general, it's a good idea to consider separately for each piece that makes up a learning
task what information we can get out of it for training an NN. The approach explained so far
gives us a _toolbox_ to concatenate update steps coming from the different sources, and due
to the very active research in this area we'll surely discover new and improved ways to compute
these updates.


```{figure} resources/placeholder.png
---
height: 220px
name: pg-toolbox
---
TODO, visual overview of toolbox  , combinations 
```

Details of PGs and additional examples can be found in the corresponding paper {cite}`holl2021pg`.
In the next section's we'll show examples of training physics-based NNs 
with invertible simulations. (These will follow soon, stay tuned.)
