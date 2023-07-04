---
title: 'FluxNLPModels.jl and KnetNLPModels.jl: Connecting Deep Learning Models with Optimization Solvers'
tags:
  - Julia
  - Machine learning
  - Smooth optimization
authors:
  - name: Paul Raynaud^[corresponding author]
    orcid: 0000-0000-0000-0001
    equal-contrib: true
    affiliation: "1, 2"
  - name: Farhad Rahbarnia
    orcid: 0000-0000-0000-0002
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: GERAD and Department of Mathematics and Industrial Engineering, Polytechnique Montréal, QC, Canada.
   index: 1
 - name: GSCOP, Grenoble, France
   index: 2
date: 10 June 2023
bibliography: paper.bib
---

# Summary

`FluxNLPModels.jl` and `KnetNLPModels.jl` are Julia [@bezanson2017julia] modules, part of the JuliaSmoothOptimizers (JSO) ecosystem [@jso].
They are designed to bridge the gap between JSO optimization solvers and deep neural network modeling frameworks, specifically Flux.jl [@Flux.jl-2018] and Knet.jl [@Knet2020].

Both Flux.jl and Knet.jl allow users to model deep neural network architectures and combine them with a loss function and a dataset from MLDataset [@MLDataset2016].
These frameworks support various usual stochastic optimizers, such as stochastic gradient [@lecun-bouttou-bengio-haffner1998], Nesterov acceleration [@Nesterov1983], Adagrad [@duchi-hazan-singer2011], and Adam [@kingma-ba2017].

FluxNLPModels.jl and KnetNLPModels.jl adopt the triptych of architecture, dataset, and loss function to model a neural network training problem as an unconstrained smooth optimization problem conforming to the NLPModels.jl API [@orban-siqueira-nlpmodels-2020].
Consequently, these models can be solved using solvers from, e.g., JSOSolvers [@orban-siqueira-jsosolvers-2021], which include gradient-based first and second-order methods.
Limited-memory quasi-Newton methods [@byrd-nocedal-schnabel-1994; @lu-1996; @liu-nocedal1989] can be used transparently by way of NLPModelModifiers [@orban-siqueira-nlpmodelsmodifiers-2021].
Contrary to usual stochastic optimizers, all methods in JSOSolvers enforce decrease of a certain merit function.

<!-- By utilizing these solvers, the loss function can be minimized to train the deep neural network using optimization methods different from those embedded in Knet.jl or Flux.jl.
The optimization frameworks as JSOSolvers.jl include solvers in which descent of a certain objective is enforced. -->
<!-- This approach allows users to leverage the interfaces provided by the deep learning libraries, including standard training and test datasets, predefined or user-defined loss functions, the ability to partition datasets into user-defined minibatches, GPU/CPU support, use of various floating-point systems, weight initialization routines, and data preprocessing capabilities. 
PR : already in the Statement of need, this section focus how what FluxNLPModel and KnetNLPModel do -->

While it is possible to write and integrate solvers directly into Flux.jl or Knet.jl, separating them into standalone packages offers advantages in terms of modularity, flexibility and ecosystem interoperability.
Leveraging existing packages within the Julia ecosystem allows developers to tap into a broader range of optimization solvers.

We hope the decoupling of the modeling tool from the optimization solvers will allow users and researchers to employ a wide variety of optimization solvers, including a range of existing solvers not traditionally applied to deep network training such as R2 [@birgin2017worst ; @birgin-gardenghi-martinez-santos-toint-2017].
<!-- , quasi-Newton trust-region methods [@ranganath-deguchy-singhal-marcia2021], or quasi-Newton linesearch [@byrd-hansen-nocedal-singer2016], which are not available in Flux.jl or Knet.jl, and have shown promising results [@ranganath-deguchy-singhal-marcia2021 ; @byrd-hansen-nocedal-singer2016]. -->

# Statement of Need

Flux.jl and Knet.jl, as standalone frameworks, do not have built-in interfaces with general optimization frameworks like JSO.
However, they offer convenient modeling features for defining neural network architectures.
These frameworks provide pre-defined neural layers, such as dense layers, convolutional layers, and other complex layers.
Additionally, they allow users to initialize the weights using various methods, such as uniform distribution.

Both offer a wide range of loss functions, e.g., negative log likelihood, and provide the flexibility for users to define their own loss functions according to their specific needs.
These frameworks enable efficient evaluation of the sampled loss and its derivatives, as well as the neural network output, on both CPU and GPU. This flexibility allows the weights to be represented as either a Vector (for CPU) or a CUVector (for GPU), with support for multiple floating-point systems. They facilitate the definition of minibatches as iterators over the dataset, enabling efficient batch processing during training.

The solvers in JSOSolvers are deterministic.
However, the integrated callback mechanism allows the user to change the training minibatch and its size at each iteration, which effectively produces stochastic solvers.
Finally, Flux.jl and Knet.jl offer convenient methods for evaluating the accuracy of the trained neural network on the test dataset.

The FluxNLPModels.jl and KnetNLPModels.jl modules have been developed to expand the range of optimization methods available for training neural networks defined with Flux.jl and Knet.jl.

They leverage the tools provided by JSO to enable the use of a broader set of optimization techniques without the need for users to reimplement them specifically for Flux.jl or Knet.jl.
This integration allows researchers and users from the deep learning community of Julia to benefit from advances in optimization.
On the other side, researchers in optimization will benefit from advances in modeling network developed by either of the Flux.jl and Knet.jl communities.

# Training a neural network with JuliaSmoothOptimizers solvers

In the following section, we illustrate how to train a LeNet architecture [@lecun-bouttou-bengio-haffner1998] using JSO solvers.
The example is divided in two, depending on whether one chooses to specify the neural network architecture with Flux.jl or Knet.jl.

## FluxNLPModels.jl

We assume that the MNIST dataset [@lecun-bouttou-bengio-haffner1998] from MLDataset has been downloaded, loaded, and a minibatch loader has been created. Additionally, we presume that a LeNet model is defined in Flux.jl. To view an example, please refer to the 'example' folder of FluxNLPModels.jl or check out this [link](https://github.com/Farhad-phd/FluxNLPModels.jl/blob/main/example/MNIST_cnn.jl). <!-- TODO change the link to JSO ? -->

To cast the LeNet model as an FluxNLPModel, one needs to pass the model that was defined in Flux, the loss function, as well as the train and test data loaders.
Flux.jl allows flexibility to define any loss function we need.
We will use the built-in `Flux.logitcrossentropy`.

```julia
using FluxNLPModels

LeNetNLPModel = FluxNLPModel(LeNet, train_loader, test_loader; loss_f = Flux.logitcrossentropy)
```

After completing the necessary steps, one can utilize a solver from JSOSolvers to minimize the loss of LeNetNLPModel. These solvers have been primarily designed for deterministic optimization. In the case of FluxNLPModel.jl (and KnetNLPModels.jl), the loss function is managed to ensure its application to sampled data. However, it is essential to modify the training minibatch between iterations. This can be accomplished by leveraging the callback mechanism incorporated in JSOSolvers. This mechanism executes a pre-defined callback at the conclusion of each iteration. For more comprehensive information, we refere the readers to the JSOSolvers documentation.

In the following code snippet, we demonstrate the execution of the R2 solver with a `callback` that changes the training minibatch at each iteration:
```julia
using JSOSolvers

max_time = 300. # run at most 5min
callback = (LeNetNLPModel, solver, stats) -> FluxNLPModels.minibatch_next_train!(LeNetNLPModel)

solver_stats = R2(LeNetNLPModel; callback, max_time) # We collect the status of Solver run
test_accuracy = FluxNLPModels.accuracy(LeNetNLPModel)
```

Another choice to train `LeNetNLPModel` is the LBFGS solver with linesearch:
```julia
solver_stats = lbfgs(LeNetNLPModel; callback, max_time)
```

To exploit any non-convexity present in `LeNetNLPModel`, an `LSR1` approximation of the Hessian which can be employed and fed into the `trunk` solver, utilizes a trust-region method with a backtracking linesearch.
To integrate the LSR1 approximation and trunk into the training process, the code can be modified as:

```julia
using NLPModelsModifiers # defines LSR1Model

lsr1_LeNet = NLPModelsModifiers.LSR1Model(LeNetNLPModel)
callback_lsr1 = 
  (lsr1_LeNet, solver, stats) -> FluxNLPModels.minibatch_next_train!(
                                                 lsr1_LeNet.model
                                               )
```

## KnetNLPModels.jl

The following code differs from the FluxNLPModel.jl example in the way it defines the neural network architecture.
To build a Knet.jl architecture, you must define the layers you need: convolutional and dense layers and link them properly.
Both topics, LeNet architecture and the MNIST dataset loading, were covered similarly during the Flux.jl example, thus, we gathered everything in the `LeNet` [training documentation](https://jso.dev/KnetNLPModels.jl/stable/LeNet_Training/) from KnetNLPModels.jl.
After defining `LeNet` and the datasets, we instantiate `KnetNLPModel`:
```julia 
using KnetNLPModels

LeNetNLPModel = KnetNLPModel(
        LeNet;
        data_train,
        data_test,
    )
```
which will instantiate automatically the mandatory data-structures as `Vector` or `CuVector` if the code is launched either on a CPU or on a GPU.

The KnetNLPModels framework allows the execution of the R2 solver with a `callback` that modifies the training minibatch at each iteration, similar to FluxNLPModels. The only difference is that the user needs to load the DNN model into KnetNLPModels.

# Acknowledgements

This work has been supported by the NSERC Alliance grant 544900-19 in collaboration with Huawei-Canada and NSERC PGS-D.

# References
