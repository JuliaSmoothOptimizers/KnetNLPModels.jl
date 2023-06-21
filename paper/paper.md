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

`KnetNLPModels.jl` and `FluxNLPModels.jl` are Julia modules [@bezanson2017julia], part of the JuliaSmoothOptimizers ecosystem [@jso]. They are designed to bridge the gap between JuliaSmoothOptimizers and deep neural network modules, specifically Flux.jl [@Flux.jl-2018] and Knet.jl [@Knet2020].

Both Flux.jl and Knet.jl allow users to construct the architecture of a deep neural network, which can then be combined with a loss function and a dataset from MLDataset [@MLDataset2016]. These frameworks support various optimizers such as stochastic gradient descent [@lecun-bouttou-bengio-haffner1998], Nesterov acceleration [@Nesterov1983], Adagrad [@duchi-hazan-singer2011], and Adam [@kingma-ba2017].

`KnetNLPModels.jl` and `FluxNLPModels.jl` adopt the triptych of architecture, dataset, and loss function to model a neural network training problem as an unconstrained smooth optimization problem, following the NLPModels.jl API [@orban-siqueira-nlpmodels-2020]. Consequently, these modules can be minimized using the solvers from JSOSolvers [@orban-siqueira-jsosolvers-2021], which offer various optimization methods, including limited-memory quasi-Newton approximations of the Hessian such as LBFGS or LSR1 [@byrd-nocedal-schnabel-1994; @lu-1996; @liu-nocedal1989].

By utilizing these solvers, the loss function can be minimized to train the deep neural network using optimization methods different from those embedded in Knet.jl or Flux.jl. Additionally, KnetNLPModels.jl and FluxNLPModels.jl provide methods to facilitate the manipulation of neural network NLPModels. For instance, users can evaluate the accuracy of the neural network or switch the sampled data considered by the loss function.

These modules enable users and researchers to leverage a wide variety of optimization solvers and tools that are not traditionally employed in deep neural network training, such as R2[@birgin2017worst]; [@birgin-gardenghi-martinez-santos-toint-2017], quasi-Newton trust-region methods, or quasi-Newton line search, which are not available in Knet.jl or Flux.jl.

# Statement of Need

Knet.jl and Flux.jl, as standalone frameworks, lack interfaces with general optimization frameworks like JuliaSmoothOptimizers[@jso]. However, they simplify the process of defining neural network architectures by providing pre-defined neural layers such as dense layers, convolutional layers, or other complex layers. These layers are initialized with a uniform distribution, saving users from implementing them manually, which can be error-prone.

Additionally, Knet.jl and Flux.jl offer a variety of loss functions, including negative log likelihood, and allow users to define their own loss functions. Both frameworks support running architectures on both CPU and GPU, providing weights as either a Vector or a CUVector, accommodating various floating-point systems.

Moreover, Knet.jl and Flux.jl provide support for training and testing datasets from MLDatasets. They simplify the definition of minibatches as an iterator of a dataset and offer a convenient way to evaluate the accuracy of the neural network on the test dataset.

While Knet.jl and Flux.jl have some differences in how architectures are defined, the floating-point systems they support, and the activity of their respective communities, the optimizers provided by both frameworks rely solely on the first derivative of the sampled loss.

KnetNLPModels.jl and FluxNLPModels.jl aim to expand the range of optimization methods that can be used to train neural networks defined by Knet.jl and Flux.jl. They achieve this through the tools provided by JuliaSmoothOptimizers, enabling users to explore a broader set of optimization techniques.

# Training a neural network with JuliaSmoothOptimizers solvers

In the following example, we build a simplified LeNet architecture [@lecun-bouttou-bengio-haffner1998], which is designed to distinguish 10 picture classes.

## KnetNLPModels.jl

The first step is to define the neural network architecture using Knet.jl.


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

function getdata(args; T = Float32) #T for types
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

xtrn, ytrn, xtst, ytst = getdata(args)
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

Once these steps are completed, you can use a solver from JSOSolvers to minimize the `LeNetNLPModel`. These solvers are originally designed for deterministic optimization. While KnetNLPModel manage the loss function to apply it to sampled data, the training minibatch needs to be changed between iterations. This can be achieved using the callback mechanism integrated in JSOSolvers, which executes a predefined function, `callback`, at the end of each iteration. For more details, refer to the [JSOSolvers documentation](https://juliasmoothoptimizers.github.io/JSOSolvers.jl/stable/solvers/).

In the following code snippet, we demonstrate the execution of the R2 solver with a `callback` that only changes the training minibatch:

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

Finally, the `LeNetNLPModel` can be enhanced by using an LBFGS approximation of the Hessian, which can be fed to the `trunk` solver, a quadratic trust-region method with a backtracking line search.

To incorporate the LBFGS approximation and the `trunk` solver into the training process, you can modify the code as follows:



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
## FluxNLPModels.jl
We will define the neural network architecture using Flux.jl, specifically the LeNet architecture [@lecun1998gradient]. First, we define some methods to assist with calculations of accuracy, data loading, and training arguments such as the number of epochs and batch size. These methods are inspired by Flux-zoo [@flux_model_zoo].
```julia
using FluxNLPModels
using CUDA, Flux, NLPModels
using Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using MLDatasets
using JSOSolvers
```


#### Loading Data
In this section, we will cover the process of loading datasets and defining minibatches for training your model using Flux. 
To donwnload and load MNIST dataset [@lecun-bouttou-bengio-haffner1998] from MLDataset:
```julia

function create_batch(args)
  # Create DataLoaders (mini-batch iterators)
  xtrain, ytrain, xtest, ytest = getdata(args) #getdata is from KnetNlPModels section
  train_loader = DataLoader((xtrain, ytrain), batchsize = args.batchsize, shuffle = true)
  test_loader = DataLoader((xtest, ytest), batchsize = args.batchsize)
  return train_loader, test_loader
end

device = cpu
train_loader, test_loader = create_batch(args)
```
#### Loss Function 
We can define any loss function that we need, here we use Flux build-in logitcrossentropy function. 

```julia
const loss = logitcrossentropy
```
We also definethe loss function `loss_and_accuracy`. It expects the following arguments:
* ADataLoader object.
* The `build_model` function we defined above.
* A device object (in case we have a GPU available).
  
```julia
function loss_and_accuracy(data_loader, model, device; T=Float32)
  acc = T(0)
  ls = T(0.0f0)
  num = T(0)
  for (x, y) in data_loader
    x, y = device(x), device(y)
    ŷ = model(x)
    ls += loss(ŷ, y, agg = sum)
    acc += sum(onecold(ŷ) .== onecold(y)) ## Decode the output of the model
    num += size(x)[end]
  end
  return ls / num, acc / num
end
```
#### Parameters 
Here are some of the parameters that we use.

   ```julia 
      @kwdef mutable struct Args
        batchsize::Int = 128    # batch size
        epochs::Int = 10        # number of epochs
        use_cuda::Bool = true   # use gpu (if cuda and gpu available)
        verbose_freq = 10       # logging for every verbose_freq iterations
      end
      args = Args() # collect options in a struct for convenience
      
  ``` 
#### Neural Network - Lenet
```julia 
## Construct Nural Network model
device = cpu # or GPU
model =
  Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10)
  ) |> device

```

#### Transfering to FluxNLPModels

```julia
  # now we set the model to FluxNLPModel
  nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss)
```

#### Train with R2 
First we need to define a callback function that changes the mini-batch and we need a data struct to keep track of the epochs and changes:
```julia

# used in the callback of R2 for training deep learning model 
mutable struct StochasticR2Data
  i::Int
  ϵ::Float64 
  state
end

function cb(nlp, stats, train_loader, device, data::StochasticR2Data;)

iter = train_loader
  if data.i == 0
    next = iterate(iter)
  else
    next = iterate(iter, data.state)
  end
  data.i += 1 

  if next === nothing 
    return
  end
  (item, data.state) = next
  nlp.current_training_minibatch = device(item) 
end
```
Now we can train using R2 and change the mini-batch according to our callback

```julia
stochastic_data = StochasticR2Data(0, atol, nothing) 

solver_stats = JSOSolvers.R2(
  nlp;
  callback = (nlp, solver, stats) 
  -> cb(nlp, stats, train_loader, device, stochastic_data),
)

## Report on train and test
train_loss, train_acc = loss_and_accuracy(train_loader, nlp.chain, device)
test_loss, test_acc = loss_and_accuracy(test_loader, nlp.chain, device)

```

# Acknowledgements

This work has been supported by the NSERC Alliance grant 544900-19 in collaboration with Huawei-Canada and NSERC PGS-D.

# References
