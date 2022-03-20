Discussion
=======================

At this point it's a good time to take another step back, and assess the different methods of the previous chapters. For deep learning applications, we can broadly distinguish three approaches: the _regular_ differentiable physics (DP) training, the training with half-inverse gradients (HIGs), and using the physical gradients (PGs). Unfortunately, we can't simply discard two of them, and focus on a single approach for all future endeavours. However, discussing the pros and cons sheds light on some fundamental aspects of physics-based deep learning, so here we go...

![Divider](resources/divider7.jpg)

## Addressing scaling issues

First and foremost, a central motivation for improved updates is the need to address the scaling issues of the learning problems. This is not a completely new problem: numerous deep learning algorithms were proposed to address these for training NNs. However, the combination of NNs with physical simulations brings new challenges that provide new angles to tackle this problem. On the negative side, we have additional, highly non-linear operators from the PDE models. On the positive side, these operators typically do not have free parameters during learning, and thus can be treated with different, tailored methods.

This is exactly where HIGs and PGs come in: instead of treating the physical simulation like the rest of the NNs (this is the DP approach), they show how much can be achieved with custom inverse solvers (PGs) or a custom numerical inversion (HIGs).

## Computational Resources

Both cases usually lead to more complicated and resource intensive training. However, assuming that we can re-use a trained model many times after the training has been completed, there are many areas of applications where this can quickly pay off: the trained NNs, despite being identical in runtime to those obtained from other training methods, often achieve significantly improved accuracies. Achieving similar levels of accuracy with regular Adam and DP-based training can be infeasible. 

When such a trained NN is used, e.g., as a surrogate model for an inverse problem, it might be executed a large number of times, and the improved accuracy can save correspondingly large amounts of computational resources in such a follow up stage. 
A good potential example are shape optimizations for the drag reduction of bodies immersed in a fluid {cite}`chen2021numerical`.





## A learning toolbox

***re-integrate?***

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

***re-integrate?***



![Divider](resources/divider1.jpg)


## Summary 

To summarize, the physical gradients showed the importance of the inversion. Even when it is only done for the physics simulation component, it can substantially improve the learning process. When we can employ a custom inverse solver, we can often do even better. These methods employed higher-order information.

✅ Pro PG: 
- Very accurate "gradient" information for physical simulations.
- Often strongly improved convergence and model performance.

❌ Con PG: 
- Requires inverse simulators (at least local ones).
- Less wide-spread availability than, e.g., differentiable physics simulators.

---

The HIGs on the other hand, go back to first order information in the form of Jacobians. They showed how useful the inversion can be even without any higher order terms. At the same time, they make use of a combined inversion of NN and physics, taking into account all samples of a mini-batch.

✅ Pro HIG: 
- Robustly addresses scaling issues, jointly for physical models and NN.
- Improved convergence and model performance.

❌ Con HIG: 
- Requires an SVD for potentially large Jacobian matrix.
- This can also lead to significant memory requirements.

---

In both cases, the resulting models can give a performance that we simply can't obtain by, e.g., training longer with a simpler DP or supervised approach. So, if we plan to evaluate these models often, e.g., shipping them in an application, this increased one-time cost can pay off in the long run.


xxx TODO, connect to uncert. chapter xxx


% DP basic, generic, 
% PGs higher order, custom inverse , chain PDE & NN together
% HIG more generic, numerical inversion , joint physics & NN

%In a way, the learning via physical gradients provide the tightest possible coupling of physics and NNs: the full non-linear process of the PDE model directly steers the optimization of the NN.

%PG old: Naturally, this comes at a cost - invertible simulators are more difficult to build (and less common) than the first-order gradients from deep learning and adjoint optimizations. Nonetheless, if they're available, invertible simulators can speed up convergence, and yield models that have an inherently better performance. 




