# KnetNLPModels : An iterface to NLPModels

| **Documentation** | **Coverage** | **DOI** |
|:-----------------:|:------------:|:-------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] [![build-cirrus][build-cirrus-img]][build-cirrus-url] | [![codecov][codecov-img]][codecov-url] | [![doi][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://paraynaud.github.io/KnetNLPModels.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://paraynaud.github.io/KnetNLPModels.jl/dev
[build-gh-img]: https://github.com/paraynaud/PartitionedStructures.jl/workflows/CI/badge.svg?branch=main
[build-gh-url]: https://github.com/paraynaud/PartitionedStructures.jl/actions
[build-cirrus-img]: https://img.shields.io/cirrus/github/paraynaud/PartitionedStructures.jl?logo=Cirrus%20CI
[build-cirrus-url]: https://cirrus-ci.com/github/paraynaud/PartitionedStructures.jl
[codecov-img]: https://codecov.io/gh/paraynaud/KnetNLPModels.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/gh/paraynaud/KnetNLPModels.jl
[doi-img]: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.822073-blue.svg
[doi-url]: https://doi.org/10.5281/zenodo.822073

## How to install
This module can be installed with the following command:
```julia
julia> ] add https://github.com/paraynaud/KnetNLPModels.jl.git
pkg> test
```

This step by step example suppose prior knowledge [julia](https://julialang.org/) and [Knet.jl](https://github.com/denizyuret/Knet.jl.git).
See the [Julia tutorial](https://julialang.org/learning/) and the [Knet.jl tutorial](https://github.com/denizyuret/Knet.jl/tree/master/tutorial) for more details.

KnetNLPModels is an interface between [Knet.jl](https://github.com/denizyuret/Knet.jl.git)'s classification neural networks and [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl.git).

A KnetNLPModel gives the user access to:
- the values of the neural network variables/weights `w`;
- the objective/loss function `L(X, Y; w)` of the loss function L at the point `w` for a given minibatch `(X,Y)`
- the gradient `∇L(X, Y; w)` of the objective/loss function at the point `w` for a given mini-batch `(X,Y)`

In addition, it provides tools to:
- Switch the minibatch used to evaluate the neural network;
- Measure the neural network's accuracy at the current point for a given testing mini-batch.

## Preliminaries

### Define the layers of interest
The following code defines a dense layer as a callable julia structure for use on a GPU via [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl):
```julia
using Knet

struct Dense{T}
  w :: Param{CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}} # parameters of the layers
  b :: Param{CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}} # bias of the layer
  f # activation function
end
(d :: Dense)(x) = d.f.(d.w * mat(x) .+ d.b) # evaluates the layer for a given input x
Dense(i :: Int, o :: Int, f=sigm) = Dense(param(o, i), param0(o), f) # define a dense layer with input size i and output size o
```
More layer structures can be defined.
Once again, see the [Knet.jl tutorial](https://github.com/denizyuret/Knet.jl/tree/master/tutorial) for more details.

### Definition of the chained structure that evaluates the network and the loss function 
```julia
using KnetNLPModels

struct Chainnll <: Chain # must derive from KnetNLPModels.Chain
  layers
  Chainnll(layers...) = new(layers)
end
(c :: Chainnll)(x) = (for l in c.layers; x = l(x); end; x) # evaluate the network for a given input x
(c :: Chainnll)(x, y) = Knet.nll(c(x), y) # compute the loss function given input x and expected output y
(c :: Chainnll)(data :: Tuple{T1,T2}) where {T1,T2} = c(first(data,2)...) # evaluate loss given data inputs (x,y)
# This lines is mandatory to compute single minibatch (ex : (x,y) = rand(dtrn) or (x,y) = first(dtrn)).
(c :: Chainnll)(d :: Knet.Data) = Knet.nll(c; data=d, average=true) # evaluate negative log likelihood loss using a minibatch iterator d
```
The chained structure that defines the neural network **must be a subtype** of `KnetNLPModels.Chain`.

### Load datasets and mini-batch
Load a dataset from [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl.git).
In this example, we use the [MNIST](https://juliaml.github.io/MLDatasets.jl/stable/datasets/MNIST/) dataset.
```julia
using MLDatasets

xtrn, ytrn = MNIST.traindata(Float32) # MNIST training dataset
ytrn[ytrn.==0] .= 10 # re-arrange indices
xtst, ytst = MNIST.testdata(Float32) # MNIST testing dataset
ytst[ytst.==0] .= 10 # re-arrange indices

dtrn = minibatch(xtrn, ytrn, 100; xsize=(size(xtrn, 1), size(xtrn, 2), 1, :)) # training mini-batch
dtst = minibatch(xtst, ytst, 100; xsize=(size(xtst, 1), size(xtst, 2), 1, :)) # testing mini-batch
```

## Definition of the neural network and KnetNLPModel
The following code defines `DenseNet`, a neural network composed of 3 dense layers.
The loss function applied is the negative likelihood function define by `Chainnll`.
```julia
DenseNet = Chainnll(Dense(784, 200), Dense(200, 50), Dense(50, 10))
```
Next, we define the `KnetNLPModel` from the neural network.
By default, the size of the minibatch is a hundredth of the dataset and the dataset used is MNIST.
```julia
DenseNetNLPModel = KnetNLPModel(DenseNet; size_minibatch=100, data_train=(xtrn, ytrn), data_test=(xtst, ytst))
```

## Tools associated to a KnetNLPModel
The dimension of the problem n:
```julia
DenseNetNLPModel.meta.nvar
```

### Get the current variables of the network:
```julia
w = vector_params(DenseNetNLPModel)
```

### Evaluate the loss function (i.e. the objective function) at the point $w$:
```julia
NLPModels.obj(DenseNetNLPModel, w)
```
The length of the vector w must be `DenseNetNLPModel.meta.nvar`.

### Evaluate the loss function gradient at the point w (ie the gradient):
```julia
NLPModels.grad!(DenseNetNLPModel, w, g)
```
The result is stored in `g ::  Vector{T}`(of size `DenseNetNLPModel.meta.nvar`).

The accuracy of the network can be evaluated with:
```julia
accuracy(DenseNetNLPModel)
```
`accuracy` use the full training dataset.
That way, the accuracy will not fluctuate depending on the testing minibatch used.

## Default behavior
By default, the training minibatch that evaluates the neural network doesn't change between evaluations.
To change the training minibatch use:
```julia
reset_minibatch_train!(DenseNetNLPModel)
```
The size of the new minibatch is the size define previously.

The size of the training and testing minibatch can be changed with:
```julia
set_size_minibatch!(DenseNetNLPModel, size)
```
The size of the new training and testing minibatch is 