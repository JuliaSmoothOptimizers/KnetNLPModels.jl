---
title: 'FluxNLPModels.jl and KnetNLPModels.jl: Wrappers and Connectors for Deep Learning Models in JSOSolvers.jl'
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

`KnetNLPModels.jl` and `FluxNLPModels.jl` are Julia [@bezanson2017julia] modules design to bridges the gap between 
deep neural networks modules: Flux.jl [@Flux.jl-2018] and Knet.jl [@Knet2020], and the JuliaSmoothOptimizers ecosystem.
Both Flux.jl and Knet.jl allow a user to construct the architecture of a deep neural network, which can then be paired with a loss function and a dataset from MLDataset [@MLDataset2016] to apply state-of-the-art optimizers such as the: stochastic gradient descent [@lecun-bouttou-bengio-haffner1998], Nesterov acceleration [@Nesterov1983], Adagrad [@duchi-hazan-singer2011], or Adam [@kingma-ba2017].

The triptych: architecture, dataset and loss function are gathered to model neural network training problem as a standard smooth optimization problem who's fulfilling the NLPModels.jl API [@orban-siqueira-nlpmodels-2020].
As an NLPModel, it benefits from the tools developed in JuliaSmoothOptimizers, such as limited-memory quasi-Newton approximation of Hessian (LBFGS or LSR1) or the solvers from JSOSolvers [@orban-siqueira-jsosolvers-2021].
The minimization of the loss function by a solver trains the deep neural network on the dataset specified.
Additionally, it furnishes methods related specifically to neural network to ease the manipulation of neural network NLPModel, for example: evaluate the accuracy of the neural network or switch the sampled data considered.

With these modules, we hope that users and researchers will be able to employ a wide variety of optimization solvers and tools not traditionally applied to deep neural network training (neither available in Knet.jl nor in Flux.jl).

# Statement of need

Knet.jl and Flux.jl, given their standalone nature, lack interfaces with general optimization frameworks such as JuliaSmoothOptimizers.
However, they ease the architecture definition by providing definition of neural layers: dense layers, convolutional layers or more complex layers, initialized with a uniform distribution, which can be difficult and error-prone to implement itself.
In addition, they propose several loss functions, like the negative log likelihood, or allow the user to define its own loss function.
Knet.jl and Flux.jl may be run on both CPU and GPU, providing weights respectfully as a Vector or a CUVector (only CUDA supported?) supporting various floating-point systems.
Moreover, they both support the training and test datasets from MLDatasets, they facilitate the definition of their minibatches as an iterator of a dataset, and they provide a way to evaluate the neural network accuracy on the test dataset.
The optimizers provided by Knet.jl and Flux.jl all rely solely on first derivatives of the sampled loss.
We are interested in using the tools of JuliaSmoothOptimizers to enable a line search or a trust-region using a quasi-Newton approximation second derivatives or methods dedicated to smooth problems.

# Training a neural network with R2

```julia
using Knet
# Define a convolutional layer
struct Conv
  weight
  bias
  activation_function
end
(c::Conv)(x) = c.activation_function.(pool(conv4(c.weight, x) .+ c.bias))
Conv(w1, w2, cx, cy, activation_function = relu) = Conv(param(w1, w2, cx, cy), param0(1, 1, cy, 1), activation_function)

# Define a dense layer
struct Dense
  weight
  bias
  activation_function
end
(d::Dense)(x) = d.activation_function.(d.weight * mat(x) .+ d.bias)
Dense(input::Int, output::Int, activation_function = sigm) = Dense(param(output, input), param0(output), activation_function)

using KnetNLPModels
# A chain of layers ended by a negative log likelihood loss function
struct Chainnll <: KnetNLPModels.Chain
  layers
  Chainnll(layers...) = new(layers)
end
(c::Chainnll)(x) = (for l in c.layers
  x = l(x)
end;
x)
(c::Chainnll)(x, y) = Knet.nll(c(x), y)
(c::Chainnll)(data::Tuple{T1, T2}) where {T1, T2} = c(first(data, 2)...)
(c::Chainnll)(d::Knet.Data) = Knet.nll(c; data = d, average = true)

LeNet = Chainnll(Conv(4,4,1,6), Conv(4,4,6,8), Dense(128, 84), Dense(84,10))
```
This LeNet architecture is designed to distinguish 10 classes.
Therefore, we chose to use the MNIST dataset [@lecun-bouttou-bengio-haffner1998] from MLDataset:
```julia
using MLDatasets 

# download datasets without user intervention
ENV["DATADEPS_ALWAYS_ACCEPT"] = true  

xtrn, ytrn = MNIST.traindata(Float32) # MNIST training dataset
ytrn[ytrn.==0] .= 10 # re-arrange indices
xtst, ytst = MNIST.testdata(Float32) # MNIST test dataset
ytst[ytst.==0] .= 10 # re-arrange indices
```

From these elements, we transfer the Knet model to a `KnetNLPModel`:
```julia 
minibatchSize = 100
LeNetNLPModel = KnetNLPModel(
        LeNet;
        data_train=(xtrn, ytrn),
        data_test=(xtst, ytst),
        size_minibatch = minibatchSize,
    )
```
which will instantiate automatically the mandatory data-structures as `Vector` or `CuVector` if the code is launched either on a CPU or on a GPU.

Once these steps are completed, a solver from JSOSolver can be used, for example R2 (je n'ai pas trouvé de ref) :
```julia
using JSOSolvers

solver_stats = R2(
        LeNetNLPModel;
        # change the minibatch at each iteration
        callback = (nlp, solver, stats) -> (KnetNLPModels.minibatch_next_train!(nlp)), 
        max_time=300.
    )

final_accuracy = KnetNLPModels.accuracy(LeNetNLPModel)
```
Being implemented as a deterministic optimization method, the mini-batch data or sophisticate stopping criteria are managed through a callback method passed to the R2 solver.

Why not being implemented to train neural network, the LBFGS linesearch of JSOSolvers.jl may also be used to train `LeNetNLPModel`
```julia
LeNet = Chainnll(Conv(4,4,1,6), Conv(4,4,6,8), Dense(128, 84), Dense(84,10))
LeNetNLPModel = KnetNLPModel(
        LeNet;
        data_train=(xtrn, ytrn),
        data_test=(xtst, ytst),
        size_minibatch = minibatchSize,
    )

solver_stats = lbfgs(
        LeNetNLPModel;
        # change the minibatch at each iteration
        callback = (nlp, solver, stats) -> (KnetNLPModels.minibatch_next_train!(nlp)), 
        max_time=300.
    )

final_accuracy = KnetNLPModels.accuracy(LeNetNLPModel)
```

`LeNetNLPModel` may be enhanced by a LBFGS approximation of the Hessian which can be fed to `trunk`, a quadratic trust-region with a back-tracking line-search.
```julia
using NLPModelsModifiers

LeNet = Chainnll(Conv(4,4,1,6), Conv(4,4,6,8), Dense(128, 84), Dense(84,10))
LeNetNLPModel = KnetNLPModel(
        LeNet;
        data_train=(xtrn, ytrn),
        data_test=(xtst, ytst),
        size_minibatch = minibatchSize,
    )

lbfgs_LeNet = NLPModelsModifiers.LBFGSModel(LeNetNLPModel)

solver_stats = trunk(
        lbfgs_LeNet;
        # change the minibatch at each iteration
        callback = (nlp, solver, stats) -> (KnetNLPModels.minibatch_next_train!(nlp.model)), 
        max_time=300.
        # It may retrieve the accuracy if needed
    )

final_accuracy = KnetNLPModels.accuracy(LeNetNLPModel)
```

To do: Add numerical graph (maybe).

# Acknowledgements

This work has been supported by the NSERC Alliance grant 544900-19 in collaboration with Huawei-Canada.

# References
