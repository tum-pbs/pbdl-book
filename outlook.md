Outlook
=======================

Despite the lengthy discussions and numerous examples, 
we've really just barely scratched the surface regarding the possibilities that arise in the context 
of physics-based deep learning.

The examples with Burgers equation and Navier-Stokes solvers are non-trivial, and good examples for advection-diffusion-type PDEs. However, there's a wide variety of other potential combinations. To name just a few promising examples from other fields:

* PDEs for chemical reactions often show complex behavior due to the interactions of multiple species. Here, and especially interesting direction is to train models that quickly learn to predict the evolution of an experiment or machine, and adjust control knobs to stabilize it, i.e., an online _control_ setting.

* Plasma simulations share a lot with vorticity-based formulations for fluids, but additionally introduce terms to handle electric and magnetic interactions within the material. Likewise, controllers for plasma fusion experiments and generators are an excellent topic with plenty of potential for DL with differentiable physics.

* Finally, weather and climate are crucial topics for humanity, and highly complex systems of fluid flows interacting with a multitude of phenomena on the surface of our planet. Accurately modeling all these interacting systems and predicting their long-term behavior shows a lot of promise to benefit from DL approaches that can interface with numerical simulations.

So overall, there's lots of exciting research work left to do - the next years and decades definitely won't be boring 👍
