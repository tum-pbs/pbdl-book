Unconditional Stablility
=======================

The results of the previous section, for time predictions with diffusion models, and earilier ones ({doc}`diffphys-discuss`)
make it clear that unconditionally stable networks are definitely possible. 
This has also been reported various other works. However, there's still a fair amount of approaches that seem to have trouble with long term stability.
This poses a very interesting question: which ingredients are necessary to obtain _unconditional stability_? 
Unconditional stability here means obtaining trained networks that are stable for arbitrarily long rollouts. Are inductive biases or special training methodologies necessary, or is it simply a matter of training enough different initializations? Our setup provides a very good starting point to shed light on this topic.

The "success stories" from earlier chapters, some with fairly simple setups, indicate that unconditional stability is “nothing special” for neural network based predictors. I.e., it does not require special loss functions or tricks beyond a properly chosen set of hyperparamters for training. As errors will accumulate over time, we can expect that network size and the total number of update steps in training are important. Interestingly, it seems that the neural network architecture doesn’t really matter: we can obtain stable rollouts with pretty much “any” architecture once it’s sufficiently large.

## Neural Network Architectures

As shown in the previuos chapter, diffusion models perform extremely well. This can be attribute to the underlying task of working with noisy distributions (e.g. for denoising or flow matching). Likewise, the network architecture has only a minor influence: the network simply needs to be large enough to provide a converging iteration. For supervised  or unrolled training, we can leverage a variety of discrete and continuous neural operators. CNNs, Unets, FNOs and Transformers are popular approaches here.
Interestingly, FNOs, due to their architecture _project_ the solution onto a subspace of the frequencies in the discretization. This inherently removes high frequencies that primarily drive isntabilities. As such, they're less strongly influenced by unrolling [(details can be found, e.g., here)](https://tum-pbs.github.io/apebench-paper/).
Operators that better preserve small-scale details, such as convolutions, can strongly benefit from unrolling. This will be a focus of the following ablations.

Interestingly, it turns out that the batch size and the length of the unrolling horizon play a crucial but conflicting role: small batches are preferable, but in the worst case under-utilize the hardware and require long training runs. Unrolling on the other hand significantly stabilizes the rollout, but leads to increased resource usage due to the longer computational graph for each NN update. Thus, our experiements show that a “sweet spot” along the Pareto-front of batch size vs unrolling horizon can be obtained by aiming for as-long-as-possible rollouts at training time in combination with a batch size that sufficiently utilizes the available GPU memory.

Learning Task: To analyze the temporal stability of autoregressive networks on long rollouts, two flow prediction tasks from the [ACDM benchmark](https://github.com/tum-pbs/autoreg-pde-diffusion) are considered: an easier incompressible cylinder flow (denoted by _Inc_), and a complex transonic wake flow (denoted as _Tra_) at Reynolds number 10 000. For Inc, the networks are trained on flows with Reynolds number 200 – 900 and required to extrapolate to Reynolds numbers of 960, 980, and 1000 during inference (_Inc-high_). For Tra, the training data consists of flows with Mach numbers between 0.53 and 0.9, and networks are tested on the Mach numbers 0.50, 0.51, and 0.52 (denoted by _Tra-ext_). This Mach number is tough as 
For each sequences in both data sets, three training runs of each architecture are unrolled over 200.000 steps. This unrolling length is no proof that these networks yield infinitely long stable rollouts, but they feature an extremely small probability for blowups.

## Architecture Comparison

As a first comparison, we'll train three network architectures with an identical U-Net architecture, that use different stabilization techniques. This comparison shows that it is possible to successfully achieve the task "unconditional stability" in different ways:
- Unrolled training (_U-Net-ut_) where gradients are backpropagated through multiple time steps during training.
- Networks trained on a single prediction step with added training noise (_U-Net-tn_). This technique is known to improve stability by reducing data shift, as the added noise emulates errors that accumulate during inference.
- Autoregressive conditional diffusion models (ACDM). A denoising diffusion model is conditioned on the previous time step and iteratively refines noise to create a prediction for the next step, as shown in {doc}`probmodels-time`. 

NT_DEBUG, todo, more ACDM discussion below!
    images from : 2024-08-05-long-rollout-www/Long Rollouts/imgs/


```{figure} resources/probmodels-uncond01.png
---
height: 240px
name: probmodels-uncond-inc
---
Vorticity predictions for an incompressible flow with a Reynolds number of 1000 over 200 000 time steps (Inc-high).
```

The figure above illustrates the resulting predictions. All methods and training runs remain unconditionally stable over the entire rollout on Inc-high. Since this flow is unsteady but fully periodic, the results of all networks are simple, periodic trajectories that prevent error accumulation. This example serves to show that for simpler tasks, long term stability is less of an issue. Networks have a relatively easy time to keep their predictions within the manifold of the solutions. Let's consider a tougher example: the transonic flows with shock waves in Tra.

```{figure} resources/probmodels-uncond02.png
---
height: 240px
name: probmodels-uncond-tra
---
Vorticity predictions for transonic flows with a Mach number 0.52 (Tra-ext, outside the trainig data range) over 200 000 time steps.
```

For the test sequences from Tra-ext, one from the three trained U-Net-tn networks has stability issues within the first few thousand steps. This network deteriorates to a simple, mean flow prediction without vortices. Unrolled training (U-Net-ut) and diffusion models (ACDM), on the other hand, are fully stable across sequences and training runs for this case, indicating a higher resistance to rollout errors which normally cause instabilities. The autoregressive diffusion models turn out to be unconditionally stable across the board [(details here)](https://arxiv.org/abs/2309.01745), so we'll drop them in the following evaluations and focus on models where stability is more difficult to achieve: the U-Nets, as representatives of convolutional, discrete neural operators.

## Stability Criteria

Focusing on the U-Net networks with unrolled training, we will next focus on training multiple models (3 each time), and measure the percentage of stable runs they achieve. This provides more thorough statistics compared to the single, qualitative examples above.
We'll investigate the first key criterium rollout length, to show how it influences fully stable rollouts over extremely long horizons.
Figure 2 lists the percentage of stable runs for a range of ablation networks on the Tra-ext data set with rollouts over 200 000 time steps. Results on the indiviual Mach numbers, as well as an average (top row) are shown.

```{figure} resources/probmodels-uncond03-ma.png
---
height: 210px
name: probmodels-uncond03-ma
---
Percentage of stable runs on the Tra-ext data set for different ablations of unrolled training.
```

The different generalization test over Mach numbers make no difference.
The most important criterion for stability is the number of unrolling steps m: while networks with m <= 4 consistently do not achieve stable rollouts, using m >= 8 is sufficient for stability across different Mach numbers. 

**Negligible Aspects:** 
Three factors that did not substantially impact rollout stability in experiments are the prediction strategy, the amount of training data, and the backbone architecture. We'll only briefly summarize the results here. First, using residual predictions, i.e., predicting the difference to the previous time step instead of the full time steps itself, did not impact stability. Second, the stability is not affected when reducing the amount of available training data by a factor of 8, from 1000 time steps per Mach number to 125 steps (while training with 8× more epochs to ensure a fair comparison). This training data reduction still retains the full physical behavior, i.e., complete vortex shedding periods. Third, it possible to train other backbone architectures with unrolling to achieve fully stable rollouts as well, such as dilated ResNets. For ResNets without dilations only one trained network is stable, most likely due to the reduced receptive field. However, we expect achieving full stability is also possible with longer training rollout horizons.

------

## Batch Size vs Rollout

Interestingly, the batch size turns out to be an important factor:
it can substantially impact the stability of autoregressive networks. This is similar to the image domain, where smaller batches are know to improve generalization (this is the motivation for using mini-batching instead of gradients over the full data set). The impact of the batch size on the stability and training time is shown in the figure below, for both investigated data sets. Networks that only come close to the ideal rollout lenght at a large batch size, can be stabilized with smaller batches. However, this effect does not completely remove the need for unrolled training, as networks without unrolling were unstable across all tested batch sizes. For the Inc case, the U-Net width was reduced by a factor of 8 across layers (in comparison to above), to artifically increase the difficulty of this task. Otherwise all parameter configurations would already be stable and show the effect of varying the batchsize.

```{figure} resources/probmodels-uncond04a.png
---
height: 210px
name: probmodels-uncond04a
---
Percentage of stable runs and training time for different combinations of rollout length and batch size for the Tra-ext data set. Grey configurations are omitted due to memory limitations (mem) or due to high computational demands (-).
```

```{figure} resources/probmodels-uncond04b.png
---
height: 210px
name: probmodels-uncond04b
---
Percentage of stable runs and training time for rollout length and batch size for the Inc-high data set. Grey again indicates out-of-memory (mem) or overly high computations (-).
```

This shows that increasing the batch size is more expensive in terms of training time on both data sets, due to less memory efficient computations. Using longer rollouts during training does not necessarily induce longer training times, as we compensate for longer rollouts with a smaller number of updates per epoch. E.g., we use either 250 batches with a rollout of 4, or 125 batches with a rollout of 8. Thus the number of simulation states that each network sees over the course of training remains constant. However, we did in practice observe additional computational costs for training the larger U-Net network on Tra-ext. This leads to the "central" question in these ablations: which combination of rollout length and batch size is most efficient?

```{figure} resources/probmodels-uncond05.png
---
height: 180px
name: probmodels-uncond05
---
Training time for different combinations of rollout length and batch size to on the Tra-ext data set (left) and the Inc-high data set (right). Only configurations that to lead to highly stable networks (stable run percentage >= 89%) are shown.
```

This figure answers this question by showing the central tradeoff between rollout length and batch size (only stable versions are included here). 
To achieve _unconditionally stable_ networks and neural operators, it is consistently beneficial to choose configurations where large rollout lengths are paired with a batch size that is big enough the sufficiently utilize the available GPU memory. This means, improved stability is achieved more efficiently with longer training rollouts rather than smaller batches, as indicated by the green dots with the lowest training times.

## Summary 

To conclude the results above: With a suitable training setup, unconditionally stable predictions with extremely long rollout are clearly possible, even for complex flows. According to the experiments, the most important factors that impact stability are the decision for or against diffusion-based training

Without diffusion, several factors need to be considered:
- Long rollouts at training time
- Small batch sizes
- Comparing these two factors: longer rollouts are preferable, and result in faster training times than smaller batch sizes
- At the same time, sufficiently large networks are necessary (this depends on the complexity of the learning task).

Factors that did not substantially impact long-term stability are:

- Prediction paradigm during training, i.e., residual and direct prediction are viable
- Additional training data without new physical behavior
- Different network architectures, although the ideal number of unrolling steps might vary for each architecture

This concludes the topic of "unconditional stability".
Further details of these experiments can be found in the [ACDM paper](https://arxiv.org/abs/2309.01745) 

