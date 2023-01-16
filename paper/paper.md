---
title: 'KnetNLPModels.jl: a bridge to apply smooth optimization solvers to neural networks training'
tags:
  - Julia
  - Machine learning
  - Smooth-optimization
authors:
  - name: Paul Raynaud
    # orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Farhad Rahbarnia
    corresponding: true
    affiliation: 1
  - name: Nathan Allaire
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: GERAD, Montréal, Canada
   index: 1
 - name: GSCOP, Grenoble, France
   index: 2
date: 6 December 2022
bibliography: paper.bib
---

# Summary

Deep neural network(DNN) modules, similar to the Julia module (ref) [Knet.jl](https://github.com/denizyuret/Knet.jl), are generally standalone modules whose provide:
- deep neural networks modelling;
- support standard training and test datasets (from MLDatasets.jl in Julia);
- several loss-functions, which may be evaluated from a mini-batch of a dataset;
- evaluate the accuracy of a neural network from a test dataset;
- GPU support of any operation performed by a neural network;
- state-of-the-art optimizers: SGD, Nesterov, Adagrad, Adam (refs), which are sophisticate stochastic line-search around first order derivatives of the loss-function.


Due to their design focused on machine learning, those modules lack interfaces with pure optimization frameworks such as JSOSolver (ref).

KnetNLPModels.jl tackles this issue by enabling wrapping DNN into unconstrained models. It inhernat and implement most, if not all, Knet's interfaces, such as:
- standard training and test datasets
- its lost functions
- ability to divide datasets into user-defined-size minibatches
- support GPU/CPU interface

<!-- KnetNLPModels.jl tackles this issue by implementing a KnetNLPModel, an unconstrained smooth optimization model. -->

<!-- KnetNLPModel gather a neural network modelled with Knet, a loss function, a dataset and implement interface's methods related to unconstrained models with Knet's functionnalities. -->
KnetNLPModel benefits from the JuliaSmoothOptimizers ecosystem and is not limitted to the Knet solvers
It has access to:
- [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) optimizers, which train the neural network by considering the weights as variables;
- augmented optimization models such as quasi-Newton models (LBFGS or LSR1).


# Working example -- name to change 

# Statement of need


# Acknowledgements

This work has been supported by the NSERC Alliance grant 544900-19 in collaboration with Huawei-Canada



# References