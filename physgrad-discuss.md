Discussion
=======================


xxx TODO update , include HIG discussion xxx
... discarded supervised, and PIs

PGs higher order, custom inverse , chain PDE & NN together

HIG more generic, numerical inversion , joint physics & NN



In a way, the learning via physical gradients provide the tightest possible coupling
of physics and NNs: the full non-linear process of the PDE model directly steers
the optimization of the NN.

Naturally, this comes at a cost - invertible simulators are more difficult to build
(and less common) than the first-order gradients from
deep learning and adjoint optimizations. Nonetheless, if they're available,
invertible simulators can speed up convergence, and yield models that have an inherently better performance.
Thus, once trained, these models can give a performance that we simply can't obtain
by, e.g., training longer with a simpler approach. So, if we plan to evaluate these
models often (e.g., ship them in an application), this increased one-time cost
can pay off in the long run.

![Divider](resources/divider1.jpg)

## Summary

✅ Pro: 
- Very accurate "gradient" information for learning and optimization.
- Improved convergence and model performance.
- Tightest possible coupling of model PDEs and learning.

❌ Con: 
- Requires inverse simulators (at least local ones).
- Less wide-spread availability than, e.g., differentiable physics simulators.
