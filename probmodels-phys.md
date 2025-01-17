Incorporating Physical Constraints
=======================

Despite the powerful capabilities of diffusion- and flow-based networks for generative modeling that we discussed in the previous sections, there is no direct feedback loop between the network, the observation and the sample at training time. This means there is no direct mechanism to include **physics-based constraints** such as priors from PDEs. As a consequence, it's very difficult to produce highly accurate samples based on learning alone: For scientific applications, we often want to make sure the errors go down to any chosen threshold.

In this chapter, we will outline strategies to remedy this shortcoming, and building on the content of previous chapters, the central goal of both methods is to get **differentiable simulations** back into the training and inference loop. The previous chapters have shown that they're very capable tools, so the main question is how to best employ them in the context of diffusion modeling.

```{note} 
Below we'll focus on the inverse problem setting from {doc}`probmodels-intro`. I.e., we have a system $y=f(x)$ (with numerical simulator $y=\mathcal P(x)$) and given an observation $y$, we'd like to obtain the posterior distribution for the distributional solution $x \sim p(x|y)$ of the inverse problem. 

```




## Guiding Diffusion Models

Having access to a physical model with a differentiable simulation $\mathcal{P}(x)=y$ means we can obtain gradients $\nabla_x$ through the simulation. As before, we aim for solving _inverse_ problems where, given an output $y$ we'd like to sample from the conditional posterior distribution $p(x|y)$ to obtain samples $x$ that explain $y$. The previous chapter demonstrated learning such distributions with diffusion models, and given a physics prior $\mathcal{P}$, there's a first fundamental choice: should be use the gradient at _training time_, i.e., trying to improve the learned distribution $p_\theta$, or at _inference time_, to improve sampling $x \sim p_\theta(x|y)$.

**Training with physics priors:** The hope of incorporating physics-based signals in the form of gradients at training time would be to improve the state of $p_\theta$ after training. While there's a certain hope this could, e.g., compensate for sparse training data, there is little hope for substantially improving the accuracy of the learned distribution. The training process for diffusion and flow matching models typically yields very capable neural networks, that are excellent at producing approximate samples from the posterior. They're typically limited in terms of their accuracy by model and training data size, but it's difficult to fundamentally improve the capabilities of a model at this stage. Rather, in this context it is more interesting to obtain higher accuracies at inference time.

**Inference with physics priors:** For scientific applications, classic simulations typically yield control knobs that allow for choosing a level of accuracy. E.g., iterative solvers for linear systems provide iteration counts and residual thresholds, and if a solution is not accurate enough, a user can simply reduce the residual threshold to obtain a more accurate output. In contrast, neural networks typically come without such controls, and even the iteration count of denoising or velocity integration (for flow matching) are bounded in terms of final accuracy. More steps typically reduce noise, and correspondingly the error, but will plateau at a level of accuracy given by the capabilities of the trained model. This is exactly where the gradients of physics solver show promise: they provide an external process that can guide and improve the output of a diffusion model. As we'll show below, this makes it possible to push the levels of accuracy beyond those of pure learning, and can yield inverse problem solvers that really outperform traditional solvers.


Recall that for denoising, we train a noise estimator $\epsilon_\theta$, and at inference time iterate denoising steps of the form
$x_{\text{new}} =  x - \hat \alpha_t \epsilon_\theta(x, t) + \hat \sigma_t \mathcal N(0,I)$ , where $\hat \alpha,\hat \sigma$ denote the merged scaling factors for both terms.
The most straight-forward approach for including gradients is to additionally include a step in the direction of the gradient $\nabla_x || \mathcal P(x) - y||_2$. For simplicity, we take an $L^2$ distance towards the observation $y$ here. This was shown to direct sampling even when the posterior is not conditional, i.e., if we only have access to $x \sim p_\theta(x)$, and is known as _diffusion posterior sampling_ {cite}`chung2023dps`.

While this approach manages to includes $\mathcal P$, there are two challenges: $x$ is typically noisy, and the gradient step can distort the distributional sampling of the denoising process. The first point is handled quite easily with an _extrapolation step_ (more details below), while the second one is more difficult to address: the gradient descent steps via $\nabla_x \mathcal P$ are akin to a classic optimization for the inverse problem and could strongly distort the outputs of the diffusion model. E.g., in the worst case they could pull the different points of the posterior distribution towards a single case favored by the simulator $\mathcal P$. Hence, the following paragraphs will outline a strategy that merges simulator and learning, while preserving the distribution of the posterior.
We'll focus on flow matching as a state-of-the-art approach next, and afterwards discuss variant that treats the diffusion steps themselves as a physical process.



![Divider](resources/divider5.jpg)



## Physics-Guided Flow Matching

To reintroduce control signals using simulators into the flow matching algorithm we'll follow {cite}`holzschuh2024fm`. The goal is to transform an existing pretrained flow-based network, as outlined in {doc}`probmodels-intro`, with a flexible control signal by aggregating the learned flow and control signals into a _controlled flow_. This is the task of a second neural network, the _control network_, in  order to make sure that the posterior distribution is not negatively affected by the signals from the simulator. This second network is small compared to the pretrained flow network, and freezing the weights of the pretrained network works very well; thus, the refinement for control needs only a fairly small amount of additional parameters and computing resources.


```{figure} resources/probmodels-phys-overview.jpg
---
height: 240px
name: probmodels-phys-overview
---
An overview of the control framework. We will consider a pretrained flow network $v_\theta$ and use the predicted flow for the trajectory point $x_t$ at time $t$ to estimate $\hat{x}_1$.
On the right, we show a gradient-based control signal with a differentiable simulator and cost function $C$ for improving $\hat{x}_1$.
An additional network learns to combine the predicted flow with feedback via the control signal to give a new controlled flow.
By combining learning-based updates with suitable controls, we avoid local optima and obtain high-accuracy samples with low inference times.
```

The control signals can be based on gradients and a cost function, if the simulator is differentiable, but they can also be learned directly from the simulator output.
Below, we'll show that performance gains due to simulator feedback are substantial and cannot be achieved by training on larger datasets alone.
Specifically, we'll show that flow matching with simulator feedback is competitive with MCMC baselines for a problem from gravitational lensing in terms of accuracy, and it beats them significantly regarding inference time. This indicates that it provides a very attractive tool for practical applications.


**Controlled flow $v_\theta^C$** First, it's a good idea to pretrain a regular, conditional flow network $v_\theta(x,y,t)$ without any control signals to make sure that we can realize the best achievable performance possible based on learning alone. 
 
Then, in a second training phase, a control network $v_\theta^C(v, c,t)$ is introduced. It receives the pretrained flow $v$ and control signal $c$ as input. Based on these additional inputs, it can used, e.g., the gradient of a PDE to produce an improved flow matching velocity. At inference time, we integrate
$dx/dt = v^C_\theta(v,c,t)$ just like before, only now this means evaluating $v_\theta(x,y,t)$ and then $c$ beforehand. (We'll focus on the details of $c$ in a moment.)
 
First, the control network is much smaller in size than the regular flow network, making up ca. $10\%$ of the weights $\theta$. The network weights of $v_\theta$ can be frozen, to train with the conditional flow matching loss {eq}`conditional-flow-matching` for a small number of additional steps. This reduces training time and compute since we do not need to backpropagate gradients through $v_\theta(x, y,t)$. Freezing the weights of $v_\theta$ typically does not negatively affects the performance, although a joint end-to-end training could provide some additional improvements. 


**1-step prediction** The conditional flow matching networks $v_\theta(x,y,t)$ from {doc}`probmodels-intro` gradually transform samples from $p_0$ to $p_1$ during inference via integrating the simple ODE $dx_t/dt = v_\theta(x_t,y,t)$ step by step. There is no direct feedback loop between the current point on the trajectory $x_t$, the observation $y$, and a physical model that we could bring into the picture. An important first issue is that the current trajectory point $x_t$ is often not be close to a good estimate of a posterior sample $x_1$. 
This is especially severe at the beginning of inference, where $x_0$ is drawn from the source distribution (typically a Gaussian), and hence $x_t$ will be very noisy. Most simulators really don't like very noisy inputs, and trying to compute gradients on top of it is clearly a very bad idea.

This issue is alleviated by extrapolating $x_t$ forward in time to obtain an estimated $\hat{x}_1$ 

$$
\begin{align} 
    \hat{x}_1 = x_t + (1-t) v_\theta(x_t, y, t).
\end{align}
$$ (eq:1_step_prediction)

and then performing subsequent operations for control and guidance on $\hat{x}_1$ instead of the current, potentially noisy $x_1$.

Note that this 1-step prediction is also conceptually related to diffusion sampling using [_likelihood-guidance_](http://DBLP:conf/nips/WuTNBC23). For inference in diffusion models, where sampling is based on the conditional score $\nabla_{x_t} \log p(x_t|y)$ and can be decomposed into 

$$
\begin{align}
    \nabla_{x_t} \log p(x_t|y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y|x_t).
\end{align}
$$

The first expression can be estimated using a pretrained diffusion network, whereas the latter is usually intractable, but can be approximated using 
$p(y|x_t) \approx p_{y|x_0}(y|\hat{x}(x_t))$,
where the denoising estimate $\hat{x}(x_t) = \mathbb{E}_q[x_0|x_t]$ is usually obtained via Tweedie's formula $(\mathbb{E}_q[x_0|x_t] - x_t) / t\sigma^2$. In practice, the estimate $\hat{x}(x_t)$ is very poor when $x_t$ is still noisy, making inference difficult in the early stages. In contrast, flows based on linear conditional transportation paths have empirically been shown to have trajectories with less curvature compared to, for example, denoising-based networks. This property of flow matching enables inference in fewer steps and providing better estimates for $\hat{x}_1$.







### Physics-based Controls

Now we focus on the content of the control signal $c$ that was already used above. We extend the idea of self-conditioning via physics-based control signals to include an additional feedback loop between the network output and an underlying physics-based prior. We'll distinguish between two types of controls in the following: a gradient-based control from a differentiable simulator, and one from a learned estimator network. 

```{figure} resources/probphys02-control.jpg
---
height: 240px
name: probphys02-control
---
Types of control signals. (a) From a differentiable simulator, and (b) from a learned encoder.
```


**Gradient-based control signal** In the first case, we make use of a differentiable simulator $\mathcal{P}$ to construct a cost function $C$. Naturally, $C$ will likewise be differentiable such that we can compute a gradient for a predicted solution. Also, we will rely on the stochasticity of diffusion/flow matching, and as such the simulator can be deterministic.

Given an observation $y$ and the estimated 1-step prediction $\hat{x}_1$, the control signal computes to how well $\hat{x}_1$ explains $y$ via the cost function $C$. Good choices for the cost are, e.g., an $L^2$ loss or a likelihood $p(y|\hat{x}_1)$. We define the control signal $c$ to consist of two components: the cost itself, and the gradient w.r.t. the cost function:  

$$
    \begin{align}
    c(\hat{x}_1, y) := [C(\mathcal{P}(\hat{x}_1), y); \nabla_{\hat{x}_1} C(\mathcal{P}(\hat{x}_1), y)].
    \end{align}
$$

As this information is passed to a network, the network can freely make use of the current distance to the target (the value of $C$) and the direction towards lowering it in the form of $\nabla_{\hat{x}_1} C$.

**Learning-based control signal** When the simulator is non-differentiable, the second variant of using a learned estimator comes in handy. 
To combine the simulator output with the observation $y$, a learnable encoder network _Enc_ with parameters $\theta_E$ can be introduced to judge the similarity of the simulation and the observation. The output of the encoder is small and of size $O(\mathrm{dim}(x))$.
The control signal is then defined as 

$$
\begin{align}
    c(\hat{x}_1, y) := Enc(\mathcal{P}(\hat{x}_1), y).
\end{align}
$$

The gradient backpropagation is stopped at the output of the simulator $\mathcal{P}$, as shown in {numref}`figure {number} <probphys02-control>`. 
Before showing some examples of the capabilities of these two types of control, we'll discuss some of their properties.



### Additional Considerations 

**Stochastic simulators** Many Bayesian inference problems have a stochastic simulator. For simplicity, we assume that all stochasticity within such a simulator can be controlled via a variable $z \sim \mathcal{N}(0, I)$, which is an additional input. Motivated by the equivalence of exchanging expectation and gradient  

$$
\begin{align}
    \nabla_{\hat{x}_1} \mathbb{E}_{z\sim \mathcal{N}(0,1)} [ C(\mathcal P_z(\hat{x}_1), y)] = \mathbb{E}_{z\sim \mathcal{N}(0,1)} [ \nabla_{\hat{x}_1} C(\mathcal P_z(\hat{x}_1), y)],
\end{align}
$$

when calling the simulator, we draw a random realization of $z$. During training, we randomly draw $z$ for each sample and step while during inference we keep the value of $z$ fixed for each trajectory. 

**Time-dependence**

If the estimate $\hat{x}_1$ is bad and the corresponding cost $C(\hat{x}_1, y)$ is high, gradients and control signals can become unreliable. It turns out that the estimates $\hat{x}_1$ become more reliable for later times in the flow matching process. 

In practice, $t \geq 0.8$ is a good threshold. Therefore, we only train the control network $v_\theta^C$ in this range, which allows for focusing on control signals containing more useful information to, e.g. fine tune the solutions with the accurate gradients of a differentiable simulator. For $t < 0.8$, we directly output the pretrained flow $v_\theta(t, x, y)$.


**Theoretical correctness**

In the formulation above, the approximation $\hat{x}_1$ only influences the control signal, which is an input to the controlled flow network $v_\theta^C$. In the case of a deterministic simulator, this makes the control signal a function of $x_t$. The controlled flow network is trained with the same loss as vanilla flow matching. This has the nice consequence that the theoretical properties are preserved.
This is in contrast to e.g. "likelihood-based guidance", which uses an approximation for $\nabla_{x_t} \log p(y|x_t)$ as a guidance term during inference, which is not covered by the original flow matching theory. 




### An Example from Astrophysics

To demonstrate how these guidance from a physics solver affect the accuracy of samples and the posterior, we show an example from strong gravitational lensing:  an inverse problem in astrophysics that is challenging and requires precise posteriors for accurate modeling of observations. In galaxy-scale strong lenses, light from a source galaxy is deflected by the gravitational potential of a galaxy between the source and observer, causing multiple images of the source to be seen. Traditional computational approaches require several minutes to many hours or days to model a single lens system. Therefore, there is an urgent need to reduce the compute and inference with learning-based methods. In this experiment, it's shown that using flow matching and the control signals with feedback from a simulator gives posterior distributions for lens modeling that are competitive with the posteriors obtained by MCMC-based methods. At the same time, they are much faster at inference. 


```{figure} resources/probmodels-astro.jpg
---
height: 240px
name: probmodels-astro
---
Results from flow matching for reconstructing gravitational lenses. Left: flow matching with a differentiable simulator (bottom) clearly outperforms pure flow matching (top). Right: comparisons against classic baselines. The FM+simulator variant is more accurate while being faster.
```

The image aboves shows an example reconstruction and the residual errors. While flow matching and the physics-based variant are both very accurate (it's hard to visually make out differences), the FM version is just on par with classic inverse solvers. The version with the simulator, however, provides a substantial boost in terms of accuracy that is very difficult to achieve even for classic solvers. The quantitative results are shown in the table on the right: the best classic baseline is AIES with an average $\chi_2$ statistic of 1.74, while FM with simulator yields 1.48. Provided that the best possible result due to noisy observations is 1.17 for this scenario, the FM+simulation version is really highly accurate. 

At the same time, the performance numbers for _modeling time_ in the right column show that the FM variant clearly outperforms the classic solvers. While the simulator increases inference time compared to only the neural network (10s to 19s), the classic baselines require more than $50\times$ longer reconstruction times. Interestingly, this example also highlights the problems of "simpler" physics combinations in the form of DPS. The DPS version does not manage to keep up with the classic solvers in terms of accuracy. To conclude, the _FM+simulator_ variant is not only substantially more accurate, but also ca. $35\times$ faster than the best classic solver above (AIES). (Source code for this approach will be available soon [in this repository](https://github.com/tum-pbs/sbi-sim).)

---

A summary of the physics-based flow matching is given by the following bullet points:

✅ Pro:
* Improved accuracy over purely learned diffusion models
* Gives control over residual accuracy
* Reduced runtime compared to traditional inverse solvers

❌ Con:
* Requires differentiable physical process
* Increased computational resources




![Divider](resources/divider6.jpg)



## Score Matching with Differentiable Physics

So far we have treated the _diffusion time_ of denoising and flow matching as a process that is purely virtual and orthogonal to the time of the physical process to be represented by the forward and inverse problems. This is the most generic viewpoint, and works nicely, as demonstrated above. However, it's interesting to think about the alternative: merging the two processes, i.e., treating the diffusion process as an inherent component of the physics system.

```{figure} resources/probmodels-smdp-1trainB.jpg
---
height: 240px
name: probmodels-smdp-trainB
---
The physics process (heat diffusion as an example, left) perturbs and "destroys" the initial state. At inference time (right, Buoyancy flow as an example), the solver is used to compute inverse steps and produce solutions by combining steps along the score and the gradient of the solver.
```

The following sections will explain such a combined approach, following the paper "Solving Inverse Physics Problems with Score Matching" {cite}`holzschuh2023smdp`, which which [code is available in this repository](https://github.com/tum-pbs/SMDP).


This approach solves inverse physics problems by leveraging the ideas of score matching. The system’s current state is moved backward in time step by step by combining an approximate inverse physics simulator and a learned correction function. A central insight of this work is that training the learned correction with a single-step loss is equivalent to a score matching objective, while recursively predicting longer parts of the trajectory during training relates to maximum likelihood training of a corresponding probability flow. The resulting inverse solver exhibits good accuracy and temporal stability. In line with diffusion modeling and in contrast to classic learned solvers, it allows for sampling the posterior of the solutions. The method will be called _SMDP_ (for _Score Matching with Differentiable Physics_) in the following.

### Training and Inference with SMDP

For training, SMDP fits a neural ODE, the probability flow, to the set of perturbed training trajectories. The probability flow is comprised of an approximate reverse physics simulator $\tilde{\mathcal{P}}^{-1}$ as well as a correction function $s_\theta$. For inference, we simulate the system backward in time from $\mathbf{x}_T$ to $\mathbf{x}_0$ by combining $\tilde{\mathcal{P}}^{-1}$, the trained $s_\theta$ and Gaussian noise in each step. 
For optimizing $s_\theta$, our approach moves a sliding window of size $S$ along the training trajectories and reconstructs the current window. Gradients for $\theta$ are accumulated and backpropagated through all prediction steps. This process is illustrated in the following figure:

```{figure} resources/probmodels-smdp-1train.jpg
---
height: 240px
name: probmodels-smdp-train
---
Overview of the score matching training process while incorporating a physics solver $\mathcal P$ and it's approximate inverse solver $\matcal{P}^{-1}.
```

A differentiable solver or a learned surrogate model is employed for $\tilde{\mathcal{P}}^{-1}$. 
The neural network $s_\theta(\mathbf{x}, t)$ parameterized by $\theta$ is trained such that

$$
\mathbf{x}_{m} \approx \mathbf{x}_{m+1} + \Delta t \left[ \tilde{\mathcal{P}}^{-1}(\mathbf{x}_{m+1}) + s_\theta(\mathbf{x}_{m+1}, t_{m+1}) \right].
$$

In this equation, the term $s_\theta(\mathbf{x}_{m+1}, t_{m+1})$ corrects approximation errors and resolves uncertainties from the stochastic forcing $F_{t_m}(z_m)$. Potentially, this process can be unrolled over multiple steps at training time to improve accuracy and stability. At inference, time the stochastic differential equation 

$$
d\mathbf{x} = \left[ -\tilde{\mathcal{P}}^{-1}(\mathbf{x}) + C \, s_\theta(\mathbf{x},t) \right] dt + g(t) dW
$$

is integrated via the Euler-Maruyama method to obtain a solution for the inverse problem.
Setting $C=1$ and excluding the noise gives the probability flow ODE: a unique, deterministic solution. This deterministic variant is not probablistic anymore, but has other interesting properties. 

```{figure} resources/probmodels-smdp-2infer.jpg
---
height: 148px
name: probmodels-smdp-infer
---
An overview of SMDP at inference time.
```


### SMDP in Action

This section shows experiments for the stochastic heat equation: $\frac{\partial u}{\partial t} = \alpha \Delta u$, which plays a fundamental role in many physical systems. It slightly perturbs the heat diffusion process and includes an additional term $g(t)\ \xi$, where $\xi$ is space-time white noise. For the experiments, we fix the diffusivity constant to $\alpha = 1$ and sample initial conditions at $t=0$ from Gaussian random fields with $n=4$ at resolution $32 \times 32$. We simulate the heat diffusion with noise from $t=0$ until $t=0.2$ using the Euler-Maruyama method and a spectral solver $\mathcal{P}_h$ with a fixed step size and $g \equiv 0.1$. Given a simulation end state $\mathbf{x}_T$, we want to recover a possible initial state $\mathbf{x}_0$. 

In this experiment, the forward solver cannot be used to infer $\mathbf{x}_0$ directly since high frequencies due to noise are amplified, leading to physically implausible solutions. Instead, the reverse physics step $\tilde{P}^{-1}$ is implemented by using the forward step of the solver $\mathcal{P}_h(\mathbf{x})$, i.e. $\tilde{\mathcal{P}}^{-1}(\mathbf{x}) \approx - \mathcal{P}_h (\mathbf{x})$.

A small ResNet-like architecture is used based on an encoder and decoder part as representation for the score function $s_\theta(\mathbf{x}, t)$. The spectral solver is implemented via differentiable programming in _JAX_. As baseline methods, a supervised training of the same architecture as $s_\theta(\mathbf{x}, t)$, a Bayesian neural network (BNN), as well as a FNO network are considered. An $L_2$ loss is used for all these methods, i.e., the training data consists of pairs of initial state $\mathbf{x}_0$ and end state $\mathbf{x}_T$. Additionally, a variant of the SMDP method is included for which the reverse physics step $\tilde{\mathcal{P}}^{-1}$ is reomved, such that the inversion of the dynamics has to be learned entirely by $s_\theta$, denoted by ''$s_\theta$~only''.


```{figure} resources/probmodels-smdp-3heat.jpg
---
name: probmodels-smdp-heat
---
While the ODE trajectories provide smooth solutions with the lowest reconstruction MSE, the SDE solutions synthesize high-frequency content, significantly improving spectral error.  
The ``$s_\theta$ only'' version without the reverse physics step exhibits a significantly larger spectral error. Metrics (right) are averaged over three runs.
```

SMDP and the baselines are evaluated by considering the _reconstruction MSE_ on a test set of $500$ initial conditions and end states. For the reconstruction MSE, the prediction of the network is simulated forward in time with the solver $\mathcal{P}_h$ to obtain a corresponding end state, which is compared to the ground truth via the $L_2$ distance. This metric has the disadvantage that it does not measure how well the prediction matches the training data manifold. I.e., for this case, whether the prediction resembles the properties of the initial Gaussian random field. For that reason, the power spectral density of the states is shown as a _spectral loss_. An evaluation and visualization of the reconstructions are given in figure \ref{fig:stochastic_heat_eq_overview}, which shows that the ODE inference performs best regarding the reconstruction MSE. However, its solutions are smooth and do not contain the necessary small-scale structures. This is reflected in a high spectral error. The SDE variant, on the other hand, performs very well in terms of spectral error and yields visually convincing solutions with only a slight increase in the reconstruction MSE. 

This highlights the role of noise as a source of entropy in the inference process for diffusion models, such as the SDE in SMDP, which is essential for synthesizing small-scale structures. Note that there is a natural tradeoff between both metrics, and the ODE and SDE inference perform best for each of the cases while using an identical set of weights. This heat diffusion example highlights the advantages and properties of treating the physical process as part of the diffusion process. This, of course, extends to other physics. E.g., [the SMDP repository](https://github.com/tum-pbs/SMDP) additionally shows a case with an inverse Navier-Stokes solve.

## Summary of Physics-based Diffusion Models

Overall, the sections above have explained two methods to incorporate physics-based constraints and models in the form of PDEs into diffusion modeling. Interestingly, the inclusion is largely in line with {doc}`diffphys`, i.e. gradients of the physics solver are a central quantity, and concepts like unrolling play an important role. On the other hand, the probabilistic modeling introduces additional complexity on the training and inference sides. It provides powerful tools and access to distribiutions of solutions (we haven't even touched follow up applications such as uncertainty quantification above), but this comes at a cost. 

As a rule of thumb 👍, diffusion modeling should only be used if the solution is a distribution that is _not_ well represented by the mean of the solutions. If the mean is accetable, "regular" neural networks offer substantial advantages in terms of reduced complexity for training and inference.

However, if the solutions are a distribution 🌦️, diffusion models are powerful tools to work with complex and varied solutions. Given its capabilties, deep learning with diffusion models arguably introduces surprisingly _little_ additional complexity. E.g., training flow matching models is suprisingly robust, can be build on top of deterministic training, and introduces only a mild computational overhead.

To show how the combination of physics solvers and diffusion models turns out in terms of an implementation, the next section shows source code for an SMDP use case.
