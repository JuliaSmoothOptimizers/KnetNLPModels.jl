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

`KnetNLPModels.jl` and `FluxNLPModels.jl` are Julia [@bezanson2017julia] modules, from the JuliaSmoothOptimizers ecosystem [@jso], design to bridge the gap between JuliaSmoothOptimizers and deep neural networks modules: Flux.jl [@Flux.jl-2018] and Knet.jl [@Knet2020].
Both Flux.jl and Knet.jl allow a user to construct the architecture of a deep neural network, which can then be paired with a loss function and a dataset from MLDataset [@MLDataset2016] to apply optimizers such as: the stochastic gradient descent [@lecun-bouttou-bengio-haffner1998], the Nesterov acceleration [@Nesterov1983], Adagrad [@duchi-hazan-singer2011], or Adam [@kingma-ba2017].

KnetNLPModels.jl and FluxNLPModels.jl consider the triptych: architecture, dataset and loss function to model a neural network training problem as an unconstrained smooth optimization problem who's fulfilling the NLPModels.jl API [@orban-siqueira-nlpmodels-2020].
Therefore, they may be minimized by the solvers from JSOSolvers [@orban-siqueira-jsosolvers-2021], which can use, among other things, limited-memory quasi-Newton approximations of the Hessian (LBFGS or LSR1) [@byrd-nocedal-schnabel-1994; @lu-1996; @liu-nocedal1989].
The minimization of the loss function by those solvers trains the deep neural network using different optimization methods than those embedded either in Knet.jl or in Flux.jl.
Additionally, KnetNLPModels.jl and FluxNLPModels.jl furnish methods to ease the manipulation of neural network NLPModel, for example: evaluate the accuracy of the neural network or switch the sampled data considered by the loss function.

With these modules, users and researchers will be able to employ a wide variety of optimization solvers and tools not traditionally applied to deep neural network training : R2, quasi-Newton trust-region methods or a quasi-Newton line search (neither available in Knet.jl nor in Flux.jl).

# Statement of need

Knet.jl and Flux.jl, given their standalone nature, lack interfaces with general optimization frameworks such as JuliaSmoothOptimizers.
However, they ease the architecture definition by providing definition of neural layers: dense layers, convolutional layers or more complex layers, initialized with a uniform distribution, which can be difficult and error-prone to implement itself.
In addition, they propose several loss functions, like the negative log likelihood, or allow the user to define its own loss function.
Knet.jl and Flux.jl architectures may be run on both CPU and GPU, providing weights respectfully as a Vector or a CUVector supporting various floating-point systems.
Moreover, they both support the training and test datasets from MLDatasets, they facilitate the definition of their minibatches as an iterator of a dataset, and they provide a way to evaluate the neural network accuracy on the test dataset.
Knet.jl and Flux.jl slightly differ on how architecture are defined, which floating-point systems are supported and how active their respective community are.
But, the optimizers provided by Knet.jl and Flux.jl all rely solely on the first derivative of the sampled loss.

Both KnetNLPModels.jl and FluxNLPModels.jl, through the mean of the tools of JuliaSmoothOptimizers, seek to broaden the scope of optimization methods that can train the neural networks defined by Knet.jl and Flux.jl.

# Training a neural network with JuliaSmoothOptimizers solvers

The first step is to define the neural network architecture using either Knet.jl or Flux.jl.
In the following example, we build a simplified LeNet architecture [@lecun-bouttou-bengio-haffner1998], which is designed to distinguish 10 picture classes.

```julia
using Knet
# Define a convolutional layer
struct Conv
  weight # AbstractMatrix of weight
  bias # AbstractVector of bias
  activation_function
end
# evaluation of a Conv layer given an input x
(c::Conv)(x) = c.activation_function.(pool(conv4(c.weight, x) .+ c.bias))
# Constructor of a Conv structure/layer
Conv(kernel_width, kernel_height, channel_input, 
  channel_output, activation_function = relu) = 
  Conv(
    param(kernel_width, kernel_height, channel_input, channel_output), 
    param0(1, 1, channel_output, 1), 
    activation_function
  )

# Define a dense layer
struct Dense
  weight # AbstractMatrix of weight
  bias # AbstractVector of bias
  activation_function
end
# evaluation of a Dense layer given an input x
(d::Dense)(x) = d.activation_function.(d.weight * mat(x) .+ d.bias)
# Constructor of a Dense structure/layer
Dense(input::Int, output::Int, activation_function = sigm) =
  Dense(param(output, input), param0(output), activation_function)

using KnetNLPModels
# A chain of layers ended by a negative log likelihood loss function
struct Chainnll <: KnetNLPModels.Chain
  layers
  Chainnll(layers...) = new(layers) # Chainnll constructor
end
# Evaluate successively each layer
# A layer's input is the precedent layer's output
(c::Chainnll)(x) = (for l in c.layers
  x = l(x)
end;
x)
# Apply the negative log likelihood function 
# on the result of the neural network forward pass
(c::Chainnll)(x, y) = Knet.nll(c(x), y)
(c::Chainnll)(data::Tuple{T1, T2}) where {T1, T2} = c(first(data, 2)...)
(c::Chainnll)(d::Knet.Data) = Knet.nll(c; data = d, average = true)

output_classes = 10
LeNet = Chainnll(Conv(4,4,1,6), Conv(4,4,6,8), Dense(128, 84), Dense(84,output_classes))
```

Accordingly to the architecture, we chose the MNIST dataset [@lecun-bouttou-bengio-haffner1998] from MLDataset:
```julia
using MLDatasets 

# download the dataset without the user intervention
ENV["DATADEPS_ALWAYS_ACCEPT"] = true  

xtrn, ytrn = MNIST.traindata(Float32) # MNIST training dataset
ytrn[ytrn.==0] .= output_classes # re-arrange indices
xtst, ytst = MNIST.testdata(Float32) # MNIST test dataset
ytst[ytst.==0] .= output_classes # re-arrange indices

data_train = (xtrn, ytrn)
data_test = (xtst, ytst)
```

From these elements, we transfer the Knet.jl architecture to a `KnetNLPModel`:
```julia 
size_minibatch = 100
LeNetNLPModel = KnetNLPModel(
        LeNet;
        data_train,
        data_test,
        size_minibatch,
    )
```
which will instantiate automatically the mandatory data-structures as `Vector` or `CuVector` if the code is launched either on a CPU or on a GPU.

Once these steps are completed, a solver from JSOSolver can minimize `LeNetNLPModel`.
These solvers are originally designed for deterministic optimization.
Even if a KnetNLPModel or a FluxNLPModel manage the loss function to apply it on a sampled data, the training minibatch must be changed between iterates.
This is done with the callback mechanism integrated in solvers from JSOSolvers, which executes a predefined function `callback` at the end of each iteration (more detail [JSOSolvers documentation](https://juliasmoothoptimizers.github.io/JSOSolvers.jl/stable/solvers/)).
In the next piece of code, we execute the solver R2 with a `callback` that only changes the training minibatch:
```julia
using JSOSolvers

max_time = 300.

# callback change the 
callback = (nlp, solver, stats) -> (KnetNLPModels.minibatch_next_train!(nlp))

solver_stats = R2(
        LeNetNLPModel;
        # change the minibatch at each iteration
        callback, 
        max_time
    )

final_accuracy = KnetNLPModels.accuracy(LeNetNLPModel)
```
In addition of changing the minibatch, the callback function may be used for more complex tasks, such as defining stopping criteria (not developped here).

Next, the LBFGS linesearch of JSOSolvers.jl may also train `LeNetNLPModel`:
```julia
LeNet = Chainnll(Conv(4,4,1,6), Conv(4,4,6,8), Dense(128, 84), Dense(84,10))
LeNetNLPModel = KnetNLPModel(
        LeNet;
        data_train,
        data_test,
        size_minibatch,
    )

solver_stats = lbfgs(
        LeNetNLPModel;
        # change the minibatch at each iteration
        callback, 
        max_time
    )

final_accuracy = KnetNLPModels.accuracy(LeNetNLPModel)
```

Finally, `LeNetNLPModel` may be enhanced by a LBFGS approximation of the Hessian which can be fed to `trunk`, a quadratic trust-region with a back-tracking line-search.
```julia
using NLPModelsModifiers

LeNet = Chainnll(Conv(4,4,1,6), Conv(4,4,6,8), Dense(128, 84), Dense(84,10))
LeNetNLPModel = KnetNLPModel(
        LeNet;
        data_train,
        data_test,
        size_minibatch,
    )

lbfgs_LeNet = NLPModelsModifiers.LBFGSModel(LeNetNLPModel)

solver_stats = trunk(
        lbfgs_LeNet;
        # change the minibatch at each iteration
        callback,
        max_time
    )

final_accuracy = KnetNLPModels.accuracy(LeNetNLPModel)
```

# Acknowledgements

This work has been supported by the NSERC Alliance grant 544900-19 in collaboration with Huawei-Canada.

# References
