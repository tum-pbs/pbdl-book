Incorporating Physical Constraints
=======================

Despite all powerful capabilities of diffusion- and flow-based networks for generative modeling that we discussed in the previous sections, there is no direct feedback loop between the network, the observation and the sample at training time. This means there is no direct mechanism to include physics-based constraints such as priors from PDEs. As a consequence, it's very difficult to produce highly accurate samples based on learning alone: For scientific applications, we often want to make sure the errors go down to any chosen threshold.

In this chapter, we will outline two strategies to remedy this shortcoming, and building on the content of previous chapters, the central goal of both methods is to get **differentiable simulations** back into the training and inference loop. The previous chapters have shown that they're very capable tools, so the main question is how to bext employ them in the context of diffusion modeling.


![Divider](resources/divider5.jpg)


## Physics-Guided Flow Matching

First, we'll outline a strategy to reintroduce control signals using simulators into the flow matching algorithm {cite}`holzschuh2024fm`. We transform an existing pretrained flow-based network, as outlined in {doc}`probmodels-intro`, with a flexible 
control signal by aggregating the learned flow and control signals into a _controlled flow_. The aggregation network is small compared to the pretrained flow network, and we find that freezing the weights of the pretrained network works very well; thus, refining needs only a minimal amount of additional parameters and compute.


```{figure} resources/probmodels-phys-overview.jpg
---
height: 240px
name: probmodels-phys-overview
---
An overview of the control framework. We will consider a pretrained flow network $v_\phi$ and use the predicted flow for the trajectory point $x_t$ at time $t$ to estimate $\hat{x}_1$.
On the right, we show a gradient-based control signal with a differentiable simulator and cost function $C$ for improving $\hat{x}_1$.
An additional network learns to combine the predicted flow with feedback via the control signal to give a new controlled flow.
By combining learning-based updates with suitable controls, we avoid local optima and obtain high-accuracy samples with low inference times.
```

The control signals can be based on gradients and a cost function, if the simulator is differentiable, but they can also be learned directly from the simulator output.
Below, we'll show that performance gains due to simulator feedback are substantial and cannot be achieved by training on larger datasets alone.
Specifically, we'll show that flow matching with simulator feedback is competitive with MCMC baselines for a problem from gravitational in terms of accuracy, and it beats them significantly regarding inference time. This indicates that it poses a very attractive tool for practical applications.

Note that in the following we'll focus on the inverse problem setting from {doc}`probmodels-intro`. I.e., we have a system $y=f(x)$ and given an observation $y$, we'd like to obtain the posterior distribution for the distributional solution $x \sim p(x|y)$ of the inverse problem. 


**1-step prediction** 

The flow matching networks $v_\phi(t,x)$ from {doc}`probmodels-intro` gradually transform samples from $p_0$ to $p_1$ during inference via integrating a simple ODE step by step. There is no direct feedback loop between the current point on the trajectory $x_t$, the observation $y$, and a physical model that we could bring into the picture. An important first issue is that the current trajectory point $x_t$ is often not be close to a good estimate of a posterior sample $x_1$. 
This is especially severe at the beginning of inference, where $x_0$ is drawn from the source distribution (typically a Gaussian), and hence $x_t$ will be very noisy. Most simulators really don't like very noisy inputs, and trying to compute gradients on top of it is clearly a very bad idea.

This issue is alleviated by extrapolating $x_t$ forward in time to obtain an estimated $\hat{x}_1$ 

$$
\begin{align} 
    \hat{x}_1 = x_t + (1-t) v_\phi(t, x_t, y).
\end{align}
$$ (eq:1_step_prediction)

and then performing subsequent operations for control and guidance on $\hat{x}_1$ instead of the current, potentially noisy $x_1$.


```{note} Direct Gradient Descent.

not great, similar to DPS?
uncond diffusion + gradients

```

Note that this 1-step prediction is also conceptually related to diffusion sampling using [_likelihood-guidance_](http://DBLP:conf/nips/WuTNBC23). For inference in diffusion models, where sampling is based on the conditional score $\nabla_{x_t} \log p(x_t|y)$ and can be decomposed into 

$$
\begin{align}
    \nabla_{x_t} \log p(x_t|y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y|x_t).
\end{align}
$$

The first expression can be estimated using a pretrained diffusion network, whereas the latter is usually intractable, but can be approximated using 
$p(y|x_t) \approx p_{y|x_0}(y|\hat{x}(x_t))$,
where the denoising estimate $\hat{x}(x_t) = \mathbb{E}_q[x_0|x_t]$ is usually obtained via Tweedie's formula $(\mathbb{E}_q[x_0|x_t] - x_t) / t\sigma^2$. In practice, the estimate $\hat{x}(x_t)$ is very poor when $x_t$ is still noisy, making inference difficult in the early stages. In contrast, flows based on linear conditional transportation paths have empirically been shown to have trajectories with less curvature compared to, for example, denoising-based networks. This property of flow matching enables inference in fewer steps and providing better estimates for $\hat{x}_1$.


**Controlled flow $v_\phi^C$** 
 
First, it's a good idea to pretrain a regular flow network $v_\phi(x,y,t)$ without any control signals to make sure that we can realize the best achievable performance possible based on learning alone. 
 
Then, in a second training phase, a control network $v_\phi^C(v, c,t)$ is introduced. It receives the pretrained flow $v$ and control signal $c$ as input. Based on these additional inputs, it can used, e.g., the gradient of a PDE to produce an improved flow matching velocity. At inference time, we integrate
$dx/dt = v^C_\phi(v,c,t)$ just like before, only now this means evaluating $v_\phi(x,y,t)$ and then $c$ beforehand. (We'll focus on the details of $c$ in a moment.)
 
First, the control network is much smaller in size than the regular flow network, making up ca. $10\%$ of the weights $\phi$. The network weights of $v_\phi$ can be frozen, to train with the conditional flow matching loss {eq}`conditional-flow-matching` for a small number of additional steps. This reduces training time and compute since we do not need to backpropagate gradients through $v_\phi(x, y,t)$. We did not observe that freezing the weights of $v_\phi$ affects the performance negatively. We include algorithms for training in appendix \ref{app:algorithms}.

## Physics-based Controls

Now we focus on the content of the control signal $c$ that was already used above. We extend the idea of self-conditioning via physics-based control signals to include an additional feedback loop between the network output and an underlying physics-based prior. We'll distinguish between two types of controls in the following: a gradient-based control from a differentiable simulator, and one from a learned estimator network. 

```{figure} resources/probmodels-learning_based_control_signal.jpg
---
height: 240px
name: learning_based_control_signal
---
TODO, learning_based_control_signal figure
```


**Gradient-based control signal**

In the first case, we make use of a differentiable simulator $S$ to construct a cost function $C$. Naturally, $C$ will likewise be differentiable such that we can compute a gradient for a predicted solution. Also, we will rely on the stochasticity of diffusion/flow matching, and as such the simulator can be deterministic.

Given an observation $y$ and the estimated 1-step prediction $\hat{x}_1$, the control signal computes to how well $\hat{x}_1$ explains $y$ via the cost function $C$. Good choices for the cost are, e.g., an $L^2$ loss or a likelihood $p(y|\hat{x}_1)$. We define the control signal $c$ to consist of two components: the cost itself, and the gradient w.r.t. the cost function:  

$$
    \begin{align}
    c(\hat{x}_1, y) := [C(S(\hat{x}_1), y); \nabla_{\hat{x}_1} C(S(\hat{x}_1), y)].
    \end{align}
$$

As this information is passed to a network, the network can freely make use of the current distance to the target (the value of $C$) and the direction towards lowering it in the form of $\nabla_{\hat{x}_1} C$.

**Learning-based control signal**

When the simulator is non-differentiable, the second variant of using a learned estimaor comes in handy. 
To combine the simulator output with the observation $y$, a learnable encoder network _Enc_ with parameters $\phi_E$ can be introduced to judge the similarity of the simulation and the observation. The output of the encoder is small and of size $O(\mathrm{dim}(x))$.
The control signal is then defined as 

$$
\begin{align}
    c(\hat{x}_1, y) := Enc(S(\hat{x}_1), y).
\end{align}
$$

The gradient backpropagation is stopped at the output of the simulator $S$, as shown in {ref}`learning_based_control_signal`. 
Before showing some examples of the capabilities of these two types of control, we'll discuss some of their properties.



## Additional Considerations 

**Stochastic simulators**

Many Bayesian inference problems have a stochastic simulator. For simplicity, we assume that all stochasticity within such a simulator can be controlled via a variable $z \sim \mathcal{N}(0, I)$, which is an additional input. Motivated by the equivalence of exchanging expectation and gradient  

$$
\begin{align}
    \nabla_{\hat{x}_1} \mathbb{E}_{z\sim \mathcal{N}(0,1)} [ C(S_z(\hat{x}_1), y)] = \mathbb{E}_{z\sim \mathcal{N}(0,1)} [ \nabla_{\hat{x}_1} C(S_z(\hat{x}_1), y)],
\end{align}
$$

when calling the simulator, we draw a random realization of $z$. During training, we randomly draw $z$ for each sample and step while during inference we keep the value of $z$ fixed for each trajectory. 

**Time-dependence**

If the estimate $\hat{x}_1$ is bad and the corresponding cost $C(\hat{x}_1, y)$ is high, gradients and control signals can become unreliable. It turns out that the estimates $\hat{x}_1$ become more reliable for later times in the flow matching process. 

In practice, $t \geq 0.8$ is a good threshold. Therefore, we only train the control network $v_\phi^C$ in this range, which allows for focusing on control signals containing more useful information to, e.g. fine tune the solutions with the accurate gradients of a differentiable simulator. For $t < 0.8$, we directly output the pretrained flow $v_\phi(t, x, y)$.


**Theoretical correctness**

In the formulation above, the approximation $\hat{x}_1$ only influences the control signal, which is an input to the controlled flow network $v_\phi^C$. In the case of a deterministic simulator, this makes the control signal a function of $x_t$. The controlled flow network is trained with the same loss as vanilla flow matching. This has the nice consequence that the theoretical properties are preserved.
This is in contrast to e.g. "likelihood-based guidance", which uses an approximation for $\nabla_{x_t} \log p(y|x_t)$ as a guidance term during inference, which is not covered by the original flow matching theory. 




## An Example from Astrophysics

To demonstrate how these guidance from a physics solver affect the accuracy of samples and the posterior, we show an example from strong gravitational lensing:  an inverse problem in astrophysics that is challenging and requires precise posteriors for accurate modeling of observations. In galaxy-scale strong lenses, light from a source galaxy is deflected by the gravitational potential of a galaxy between the source and observer, causing multiple images of the source to be seen. Traditional computational approaches require several minutes to many hours or days to model a single lens system. Therefore, there is an urgent need to reduce the compute and inference with learning-based methods. In this experiment, we demonstrate that using flow matching and the control signals with feedback from a simulator, we obtain posterior distributions for lens modeling that are competitive with the posteriors obtained by MCMC-based methods but with much faster inference times. 

...



## Score Matching with Differentiable Physics

alternative: equate diffusion time and physics

sbi-sim ...?
