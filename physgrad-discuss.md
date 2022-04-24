Discussion of Improved Gradients
=======================

At this point it's a good time to take another step back, and assess the different methods introduced so far. For deep learning applications, we can broadly distinguish three approaches: the _regular_ differentiable physics (DP) training, the training with half-inverse gradients (HIGs), and using the scale-invariant physics updates (SIPs). Unfortunately, we can't simply discard two of them, and focus on a single approach for all future endeavours. However, discussing the pros and cons sheds light on some fundamental aspects of physics-based deep learning.

![Divider](resources/divider7.jpg)

## Addressing scaling issues

First and foremost, a central motivation for improved updates is the need to address the scaling issues of the learning problems. This is not a completely new problem: numerous deep learning algorithms were proposed to address these for training NNs. However, the combination of NNs with physical simulations brings new challenges that at the same time provide new angles to tackle this problem. On the negative side, we have additional, highly non-linear operators from the PDE models. On the positive side, these operators typically do not have free parameters during learning, and thus can be treated with different, tailored methods.

This is exactly where HIGs and SIPs come in: instead of treating the physical simulation like the rest of the NNs (this is the DP approach), they show how much can be achieved with custom inverse solvers (SIPs) or a custom numerical inversion (HIGs). Both methods make important steps towards _scale-invariant_ training.

## Computational Resources

Both cases usually lead to more complicated and resource intensive training. However, assuming that we can re-use a trained model many times after the training has been completed, there are many areas of application where this can quickly pay off: the trained NNs, despite being identical in runtime to those obtained from other training methods, often achieve significantly improved accuracies. Achieving similar levels of accuracy with regular Adam and DP-based training can be completely infeasible. 

When such a trained NN is used, e.g., as a surrogate model for an inverse problem, it might be executed a large number of times, and the improved accuracy can save correspondingly large amounts of computational resources in such a follow up stage. 
A good potential example are shape optimizations for the drag reduction of bodies immersed in a fluid {cite}`chen2021numerical`.




![Divider](resources/divider1.jpg)


## Summary 

To summarize, this chapter demonstrated the importance of the inversion. 
An important takeaway message is that
the regular gradients from NN training are not the best choice when PDEs are 
involved. In these situations we can get much better information about how to direct the
optimization than the localized first-order information that regular gradients provide.

Even when the inversion is only done for the physics simulation component (as with SIPs), it can substantially improve the learning process. The custom inverse solvers allow us to employ higher-order information in the training.

✅ Pro SIP: 
- Very accurate "gradient" information for physical simulations.
- Often strongly improved convergence and model performance.

❌ Con SIP: 
- Require inverse simulators (at least local ones).
- Only makes the physics component scale-invariant.

---

The HIGs on the other hand, go back to first order information in the form of Jacobians. They show how useful the inversion can be even without any higher order terms. At the same time, they make use of a combined inversion of NN and physics, taking into account all samples of a mini-batch to compute an optimal first-order direction.

✅ Pro HIG: 
- Robustly addresses scaling issues, jointly for physical models and NN.
- Improved convergence and model performance.

❌ Con HIG: 
- Requires an SVD for a potentially large Jacobian matrix.
- This can be costly in terms of runtime and memory.

---

In both cases, the resulting neural networks can yield a performance that we simply can't obtain by, e.g., training longer with a simpler DP or supervised approach. So, if we plan to evaluate these models often, e.g., shipping them in an application, this increased one-time cost will pay off in the long run.

This concludes the chapter on improved learning methods for physics-based NNs. 
It's clearly an active topic of research, with plenty of room for new methods, but the algorithms here already
indicate the potential of tailored learning algorithms for physical problems. 
This also concludes the focus on numerical simulations as DL components. In the next chapter, we'll instead
focus on a different statistical viewpoint, namely the inclusion of uncertainty.
