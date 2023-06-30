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

`FluxNLPModels.jl` and `KnetNLPModels.jl` adopt the triptych of architecture, dataset, and loss function to model a neural network training problem as an unconstrained smooth optimization problem conforming to the NLPModels.jl API [@orban-siqueira-nlpmodels-2020].
Consequently, these models can be solved using solvers from, e.g., JSOSolvers [@orban-siqueira-jsosolvers-2021], which include gradient-based first and second-order methods.
Limited-memory quasi-Newton methods [@byrd-nocedal-schnabel-1994; @lu-1996; @liu-nocedal1989] can be used transparently by way of NLPModelModifiers [@orban-siqueira-nlpmodelsmodifiers-2021].
Contrary to usual stochastic optimizers, all methods in JSOSolvers enforce decrease of a certain merit function.

<!-- By utilizing these solvers, the loss function can be minimized to train the deep neural network using optimization methods different from those embedded in Knet.jl or Flux.jl.
The optimization frameworks as JSOSolvers.jl include solvers in which descent of a certain objective is enforced. -->
<!-- This approach allows users to leverage the interfaces provided by the deep learning libraries, including standard training and test datasets, predefined or user-defined loss functions, the ability to partition datasets into user-defined minibatches, GPU/CPU support, use of various floating-point systems, weight initialization routines, and data preprocessing capabilities. 
PR : already in the Statement of need, this section focus how what FluxNLPModel and KnetNLPModel do -->

While it is possible to write and integrate solvers directly into Flux.jl or Knet.jl, separating them into standalone packages offers advantages in terms of modularity, flexibility and ecosystem interoperability.
Leveraging existing packages within the Julia ecosystem allows developers to tap into a broader range of optimization solvers.

We hope the decoupling of the modeling tool from the optimization solvers will allow users and researchers to employ a wide variety of optimization solvers, including a range of existing solvers not traditionally applied to deep network training such as R2 [@birgin2017worst ; @birgin-gardenghi-martinez-santos-toint-2017], quasi-Newton trust-region methods [@ranganath-deguchy-singhal-marcia2021], or quasi-Newton line search [@byrd-hansen-nocedal-singer2016], which are not available in Flux.jl or Knet.jl, and have shown promising results [@ranganath-deguchy-singhal-marcia2021 ; @byrd-hansen-nocedal-singer2016].

# Statement of Need

Flux.jl and Knet.jl, as standalone frameworks, do not have built-in interfaces with general optimization frameworks like JSO [@jso].
However, they offer convenient modeling features for defining neural network architectures.
These frameworks provide pre-defined neural layers, such as dense layers, convolutional layers, and other complex layers.
Additionally, they allow users to initialize the weights using various methods, such as uniform distribution.

Both offer a wide range of loss functions, e.g., negative log likelihood, and provide the flexibility for users to define their own loss functions according to their specific needs.
These frameworks support efficient evaluation the sampled loss and its derivatives and of neural network output on both CPU and GPU, which allows the weights to be represented as either a Vector (for CPU) or a CUVector (for GPU) with support for various floating-point systems.
They facilitate the definition of minibatches as iterators over the dataset, enabling efficient batch processing during training.

The solvers in JSOSolvers are deterministic.
However, the integrated callback mechanism allows the user to change the training minibatch and its size at each iteration, which effectively produces stochastic solvers.
Finally, Flux.jl and Knet.jl offer convenient methods for evaluating the accuracy of the trained neural network on the test dataset, which allows users to assess the model's capability.

<!-- While there are differences between Flux.jl and Knet.jl in terms of how architectures are defined, the floating-point systems supported, and the level of community activity, both frameworks rely on the first derivative of the sampled loss function for their optimization algorithms. -->

The FluxNLPModels.jl and KnetNLPModels.jl modules have been developed to expand the range of optimization methods available for training neural networks defined with Flux.jl and Knet.jl.
These modules leverage the tools provided by JSO to enable the use of a broader set of optimization techniques without the need for users to reimplement them specifically for Flux.jl or Knet.jl.
This integration allows researchers and users from the deep learning community to benefit from advances in optimization, and researchers in optimization to benefit from advances in modeling network structures.

# Training a neural network with JuliaSmoothOptimizers solvers

In the following section, we illustrate how to train a LeNet architecture [@lecun-bouttou-bengio-haffner1998] using JSO solvers.
The example is divided in two, depending on whether one chooses to specify the neural network architecture with Flux.jl or Knet.jl.

## FluxNLPModels.jl

```julia 
using CUDA
using Flux
using Statistics

device = cpu # or gpu
LeNet =
  Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
  ) |> device
```

We will discuss the process of loading datasets and defining minibatches for model training using the Flux framework. To download and load the MNIST dataset [@lecun-bouttou-bengio-haffner1998] from MLDataset, follow these steps:

```julia
using MLDatasets
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs

function get_MNIST(;T = Float32)
  xtrain, ytrain = MLDatasets.MNIST(Tx = T, split = :train)[:] 
  xtest, ytest = MLDatasets.MNIST(Tx = T, split = :test)[:] 
  return xtrain, ytrain, xtest, ytest
end

function get_data(; T = Float32)
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
  xtrain, ytrain, xtest, ytest = get_MNIST(;T=T)
  # Reshape Data in order to flatten each image into a linear array
  xtrain = Flux.flatten(xtrain)
  xtest = Flux.flatten(xtest)
  # One-hot-encode the labels
  ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)
  return xtrain, ytrain, xtest, ytest
end

function create_batch(; batchsize = 128)
  # Create DataLoaders (mini-batch iterators)
  xtrain, ytrain, xtest, ytest = get_data()
  xtrain = reshape(xtrain, 28,28,1,:)
  xtest = reshape(xtest, 28,28,1,:)
  train_loader = DataLoader((xtrain, ytrain), batchsize = batchsize, shuffle = true)
  test_loader = DataLoader((xtest, ytest), batchsize = batchsize)
  return train_loader, test_loader
end

train_loader, test_loader = create_batch()
```

To cast the LeNet model as an FluxNLPModel, one needs to pass the model that was defined in Flux, the loss function, as well as the train and test data loaders.
Flux.jl allows flexibility to define any loss function we need.
We will use the built-in `Flux.logitcrossentropy`.

```julia
using FluxNLPModels
using Flux.Losses: logitcrossentropy
const loss = logitcrossentropy

LeNetNLPModel = FluxNLPModel(LeNet, train_loader, test_loader; loss_f = loss)

```

After completing the necessary steps, one can utilize a solver from JSOSolvers to minimize the loss of LeNetNLPModel. These solvers have been primarily designed for deterministic optimization. In the case of FluxNLPModel.jl (and KnetNLPModels.jl), the loss function is managed to ensure its application to sampled data. However, it is essential to modify the training minibatch between iterations. This can be accomplished by leveraging the callback mechanism incorporated in JSOSolvers. This mechanism executes a pre-defined function, known as a callback, at the conclusion of each iteration. For more comprehensive information, please consult the JSOSolvers [@jso] documentation.

In the following code snippet, we demonstrate the execution of the R2 solver with a `callback` that changes the training minibatch at each iteration:
```julia
using JSOSolvers

max_time = 300. # run at most 5min
callback = (LeNetNLPModel, solver, stats) -> FluxNLPModels.minibatch_next_train!(LeNetNLPModel)

solver_stats = R2(LeNetNLPModel; callback, max_time)

## Report on test data
test_accuracy = FluxNLPModels.accuracy(LeNetNLPModel)
```

Another choice to train `LeNetNLPModel` is the LBFGS linesearch solver:
```julia
solver_stats = lbfgs(LeNetNLPModel; callback, max_time)
test_accuracy = FluxNLPModels.accuracy(LeNetNLPModel)
```

To exploit any non-convexity present in `LeNetNLPModel`, an `LSR1` approximation of the Hessian can be employed and fed into the `trunk` solver.
The trunk solver utilizes a trust-region method with a backtracking line search.
To integrate the LSR1 approximation and trunk into the training process, the code can be modified as:

```julia
using NLPModelsModifiers # define also LSR1Model

lsr1_LeNet = NLPModelsModifiers.LSR1Model(LeNetNLPModel)
callback_lsr1 = 
  (lsr1_LeNet, solver, stats) -> FluxNLPModels.minibatch_next_train!(
                                                 lsr1_LeNet.model
                                               )

solver_stats = trunk(lsr1_LeNet; callback = callback_lsr1, max_time)

test_accuracy = FluxNLPModels.accuracy(LeNetNLPModel)
```

## KnetNLPModels.jl

The following code differs from the FluxNLPModel.jl example in the way it defines the neural network architecture.
To define the same neural network architecture, you must define the layers you need: convolutionnal and dense layers (facilitated by Knet.jl):
```julia
using Knet

struct ConvolutionnalLayer
  weight
  bias
  activation_function
end
# evaluation of a ConvolutionnalLayer layer given an input x
(c::ConvolutionnalLayer)(x) = c.activation_function.(pool(conv4(c.weight, x) .+ c.bias))
# Constructor of a ConvolutionnalLayer structure
ConvolutionnalLayer(kernel_width, kernel_height, channel_input, 
  channel_output, activation_function = relu) = 
  ConvolutionnalLayer(
    param(kernel_width, kernel_height, channel_input, channel_output), 
    param0(1, 1, channel_output, 1), 
    activation_function
  )

struct DenseLayer
  weight
  bias
  activation_function
end
# evaluation of a DenseLayer given an input x
(d::DenseLayer)(x) = d.activation_function.(d.weight * mat(x) .+ d.bias)
# Constructor of a DenseLayer structure
DenseLayer(input::Int, output::Int, activation_function = sigm) =
  DenseLayer(param(output, input), param0(output), activation_function)

# A chain of layers ended by a negative log likelihood loss function
struct Chainnll
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

LeNet = Chainnll(ConvolutionnalLayer(5,5,1,6), 
                 ConvolutionnalLayer(5,5,6,16),
                 DenseLayer(256, 120),
                 DenseLayer(120, 84),
                 DenseLayer(84,output_classes)
                )
```

Accordingly to LeNet architecture, we chose the MNIST dataset [@lecun-bouttou-bengio-haffner1998] from MLDataset:
```julia
xtrain, ytrain, xtest, ytest = get_MNIST(; T=Float32)
ytrain[ytrain.==0] .= output_classes # re-arrange indices
ytest[ytest.==0] .= output_classes # re-arrange indices
data_train = (xtrain, ytrain)
data_test = (xtest, ytest)
```

From these elements, we transfer the Knet.jl architecture to a `KnetNLPModel`:
```julia 
using KnetNLPModels

size_minibatch = 100
LeNetNLPModel = KnetNLPModel(
        LeNet;
        data_train,
        data_test,
        size_minibatch,
    )
```
which will instantiate automatically the mandatory data-structures as `Vector` or `CuVector` if the code is launched either on a CPU or on a GPU.

The following code snippet executes the R2 solver with a `callback` that changes the training minibatch at each iteration:
```julia
using JSOSolvers

callback = (nlp, solver, stats) -> KnetNLPModels.minibatch_next_train!(nlp)

solver_stats = R2(LeNetNLPModel; callback, max_time)

test_accuracy = KnetNLPModels.accuracy(LeNetNLPModel)
```
The other solvers defined for FluxNLPModel may also be applied for any KnetNLPModel:
```julia
# lbfgs
solver_stats = lbfgs(LeNetNLPModel; callback, max_time)

test_accuracy = FluxNLPModels.accuracy(LeNetNLPModel)

# trunk(LSR1Model(LetNLPModel))
lsr1_LeNet = NLPModelsModifiers.LSR1Model(LeNetNLPModel)
solver_stats = trunk(lsr1_LeNet; callback = callback_lsr1, max_time)

test_accuracy = FluxNLPModels.accuracy(LeNetNLPModel)
```

# Acknowledgements

This work has been supported by the NSERC Alliance grant 544900-19 in collaboration with Huawei-Canada and NSERC PGS-D.

# References
