---
title: 'KnetNLPModels.jl and FluxNLPModels.jl: Connecting Deep Learning Models with Optimization Solvers'
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

`KnetNLPModels.jl` and `FluxNLPModels.jl` are Julia [@bezanson2017julia] modules, part of the JuliaSmoothOptimizers (JSO) ecosystem [@jso].
They are designed to bridge the gap between JSO optimization solvers and deep neural network modules, specifically Flux.jl [@Flux.jl-2018] and Knet.jl [@Knet2020].

Both Flux.jl and Knet.jl allow users to construct the architecture of a deep neural network, which can then be combined with a loss function and a dataset from MLDataset [@MLDataset2016].
These frameworks support various usual stochastic optimizers, such as stochastic gradient descent [@lecun-bouttou-bengio-haffner1998], Nesterov acceleration [@Nesterov1983], Adagrad [@duchi-hazan-singer2011], and Adam [@kingma-ba2017].

`KnetNLPModels.jl` and `FluxNLPModels.jl` adopt the triptych of architecture, dataset, and loss function to model a neural network training problem as an unconstrained smooth optimization problem, following the NLPModels.jl API [@orban-siqueira-nlpmodels-2020].
Consequently, these modules can be solved using the solvers from JSOSolvers [@orban-siqueira-jsosolvers-2021], which offer various optimization methods, including limited-memory quasi-Newton approximations of the Hessian such as LBFGS or LSR1 [@byrd-nocedal-schnabel-1994; @lu-1996; @liu-nocedal1989].

<!-- By utilizing these solvers, the loss function can be minimized to train the deep neural network using optimization methods different from those embedded in Knet.jl or Flux.jl.
The optimization frameworks as JSOSolvers.jl include solvers in which descent of a certain objective is enforced. -->

The packages KnetNLPModels.jl and FluxNLPModels.jl expose DL models as optimization problems conforming to the NLPModels.jl API. 

This approach allows users to leverage the interfaces provided by the DL libraries, including standard training and test datasets, predefined or user-defined loss functions, the ability to partition datasets into user-defined minibatches, GPU/CPU support, use of various floating-point systems, weight initialization routines, and data preprocessing capabilities.

While it is possible to integrate solvers directly into deep learning frameworks like Flux or Knet, separating them into standalone packages offers several advantages in terms of modularity, flexibility, ecosystem interoperability, and performance optimization. Leveraging existing solver packages within the Julia ecosystem allows developers to tap into a broader range of optimization techniques, while deep learning frameworks can focus on their core purpose of building and training neural networks.

We hope the decoupling of the modeling tool from the optimization solvers will allow users and researchers to employ a wide variety of optimization solvers, including a range of existing solvers not traditionally applied to deep network training, such as R2 [@birgin2017worst ; @birgin-gardenghi-martinez-santos-toint-2017], quasi-Newton trust-region methods, or quasi-Newton line search, which are not available in Knet.jl or Flux.jl.

# Statement of Need

Knet.jl and Flux.jl, as standalone frameworks, do not have built-in interfaces with general optimization frameworks like JSO [@jso]. However, they offer convenient features for defining neural network architectures. These frameworks provide pre-defined neural layers, such as dense layers, convolutional layers, and other complex layers. Additionally, they allow users to initialize the weights using various methods, including uniform distribution.

By providing these pre-defined layers and weight initialization options, Knet.jl and Flux.jl simplify the process of defining neural network architectures. This eliminates the need for users to manually implement these layers, which can be error-prone and time-consuming.

Both Knet.jl and Flux.jl offer a wide range of loss functions, including the negative log likelihood, and provide the flexibility for users to define their own loss functions according to their specific needs. These frameworks support efficient evaluation of neural network architectures on both CPU and GPU, allowing the weights to be represented as either a Vector (for CPU) or a CUVector (for GPU) with support for various floating-point systems.

In addition, Knet.jl and Flux.jl simplify the handling of training and testing datasets by providing support for MLDatasets. They facilitate the definition of minibatches as iterators over the dataset, enabling efficient batch processing during training. Moreover, these frameworks offer convenient methods for evaluating the accuracy of the trained neural network on the test dataset, allowing users to assess the model's performance.

While there are some differences between Knet.jl and Flux.jl in terms of how architectures are defined, the supported floating-point systems, and the level of community activity, both frameworks rely on the first derivative of the sampled loss function for their optimization algorithms.

To expand the range of optimization methods available for training neural networks defined in Knet.jl and Flux.jl, the KnetNLPModels.jl and FluxNLPModels.jl modules have been developed. These modules leverage the tools provided by JSO (JuliaSmoothOptimizers) to enable the use of a broader set of optimization techniques without the need for users to reimplement them specifically for Knet or Flux architectures. This integration allows researchers and users to explore and apply advanced optimization methods to train their neural networks effectively.

# Training a neural network with JuliaSmoothOptimizers solvers

In the following example, we build a simplified LeNet architecture [@lecun-bouttou-bengio-haffner1998], which is designed to distinguish 10 picture classes.

## FluxNLPModels.jl
We will define the neural network architecture using Flux.jl, specifically the LeNet architecture [@lecun1998gradient]. First, we define some methods to assist with calculations of accuracy, data loading, and training arguments such as the number of epochs and batch size.
These methods are inspired by Flux-zoo [@flux_model_zoo].
```julia
using FluxNLPModels
using CUDA, Flux, NLPModels
using Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using MLDatasets
using JSOSolvers
```


#### Loading Data
In this section, we will cover the process of loading datasets and defining minibatches for training your model using Flux.

To download and load the MNIST dataset [@lecun-bouttou-bengio-haffner1998] from MLDataset, you can use the following steps:
```julia
function getdata(; T = Float32) #T for types
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
  # Loading Dataset	
  xtrain, ytrain = MLDatasets.MNIST(Tx = T, split = :train)[:]
  xtest, ytest = MLDatasets.MNIST(Tx = T, split = :test)[:]
  # Reshape Data in order to flatten each image into a linear array
  xtrain = Flux.flatten(xtrain)
  xtest = Flux.flatten(xtest)
  # One-hot-encode the labels
  ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)
  return xtrain, ytrain, xtest, ytest
end

function create_batch(; batchsize = 128)
  # Create DataLoaders (mini-batch iterators)
  xtrain, ytrain, xtest, ytest = getdata()
  xtrain = reshape(xtrain, 28,28,1,:)
  xtest = reshape(xtest, 28,28,1,:)
  train_loader = DataLoader((xtrain, ytrain), batchsize = batchsize, shuffle = true)
  test_loader = DataLoader((xtest, ytest), batchsize = batchsize)
  return train_loader, test_loader
end

train_loader, test_loader = create_batch()

```
#### Loss Function 
We have the flexibility to define any loss function we need. In this example, we will use the built-in `Flux.logitcrossentropy` function as our loss function. 

```julia
const loss = logitcrossentropy
```
### Neural Network - Lenet
The model architecture is defined as LeNet [@lecun1998gradient] in Flux.jl using the build-in methods such as Dense and MaxPool.
```julia 
## Construct Nural Network model
model =
  Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
  ) |> cpu
```

#### Transfering to FluxNLPModels
To transfer the LeNet model using the FluxNLPModel function, you need to pass the model that was defined in Flux, as well as the train and test data loaders. The loss function can also be customized according to your needs. 
```julia
  nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss)
```

#### Train with R2 

To leverage JSO Solvers, such as R2, with the FluxNLPModel object, you simply need to pass the FluxNLPModel object to the solver. Additionally, we can demonstrate how to use the callback method to change the minibatch during training. Finally, we can compute the accuracy of the trained model.

```julia

callback = (nlp, solver, stats) -> FluxNLPModels.minibatch_next_train!(nlp)
solver_stats = JSOSolvers.R2(nlp; callback = callback)

## Report on train and test
train_acc = FluxNLPModels.accuracy(nlp; data_loader = train_loader)
test_acc = FluxNLPModels.accuracy(nlp) #on the test data
```




## KnetNLPModels.jl

The first step is to define the neural network architecture using Knet.jl.
```julia
using Knet

struct ConvolutionnalLayer
  weight
  bias
  activation_function
end
# evaluation of a Conv layer given an input x
(c::ConvolutionnalLayer)(x) = c.activation_function.(pool(conv4(c.weight, x) .+ c.bias))
# Constructor of a Conv structure/layer
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
# evaluation of a Dense layer given an input x
(d::DenseLayer)(x) = d.activation_function.(d.weight * mat(x) .+ d.bias)
# Constructor of a Dense structure/layer
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
Accordingly to the architecture, we chose the MNIST dataset [@lecun-bouttou-bengio-haffner1998] from MLDataset:
```julia
using MLDatasets 
function get_MNIST(;T = Float32)
  xtrain, ytrain = MLDatasets.MNIST(Tx = T, split = :train)[:] 
  xtest, ytest = MLDatasets.MNIST(Tx = T, split = :test)[:] 
  return xtrain, ytrain, xtest, ytest
end

xtrain, ytrain, xtest, ytest = get_MNIST()
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

Once these steps are completed, you can use a solver from JSOSolvers to minimize the loss of `LeNetNLPModel`.
These solvers are originally designed for deterministic optimization.
While KnetNLPModel manages the loss function to apply it to sampled data, the training minibatch needs to be changed between iterations.
This can be achieved using the callback mechanism integrated in JSOSolvers, which executes a predefined function, `callback`, at the end of each iteration.
For more details, refer to the JSOSolvers documentation [@jso].

In the following code snippet, we demonstrate the execution of the R2 solver with a `callback` that changes the training minibatch at each iteration:

```julia
using JSOSolvers

max_time = 300.

callback = (nlp, solver, stats) -> KnetNLPModels.minibatch_next_train!(nlp)

solver_stats = R2(LeNetNLPModel; callback, max_time)

final_accuracy = KnetNLPModels.accuracy(LeNetNLPModel)
```
In addition of changing the minibatch, the callback function may be used for more complex tasks, such as defining stopping criteria (not illustrated here).

Next, the LBFGS linesearch solver of JSOSolvers.jl may also train `LeNetNLPModel`:
```julia
solver_stats = lbfgs(LeNetNLPModel; callback, max_time)

final_accuracy = KnetNLPModels.accuracy(LeNetNLPModel)
```
Finally, the `LeNetNLPModel` can be enhanced by using an LSR1 (or LBFGS) approximation of the Hessian, which can be fed to the `trunk` solver, a quadratic trust-region method with a backtracking line search.

To incorporate the LSR1 approximation and the `trunk` solver into the training process, you can modify the code as follows:
```julia
using NLPModelsModifiers # define also LBFGSModel

lsr1_LeNet = NLPModelsModifiers.LSR1Model(LeNetNLPModel)

callbacklsr1 = (nlp, solver, stats) -> KnetNLPModels.minibatch_next_train!(nlp.model)

solver_stats = trunk(lsr1_LeNet; callback = callbacklsr1, max_time)

final_accuracy = KnetNLPModels.accuracy(LeNetNLPModel)
```

# Acknowledgements

This work has been supported by the NSERC Alliance grant 544900-19 in collaboration with Huawei-Canada and NSERC PGS-D.

# References
