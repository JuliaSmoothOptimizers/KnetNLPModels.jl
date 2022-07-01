# KnetNLPModels : An interface to NLPModels

| **Documentation** | **Linux/macOS/Windows/FreeBSD** | **Coverage** | **DOI** |
|:-----------------:|:-------------------------------:|:------------:|:-------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] [![build-cirrus][build-cirrus-img]][build-cirrus-url] | [![codecov][codecov-img]][codecov-url] | [![doi][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaSmoothOptimizers.github.io/KnetNLPModels.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://JuliaSmoothOptimizers.github.io/KnetNLPModels.jl/dev
[build-gh-img]: https://github.com/JuliaSmoothOptimizers/KnetNLPModels.jl/workflows/CI/badge.svg?branch=main
[build-gh-url]: https://github.com/JuliaSmoothOptimizers/KnetNLPModels.jl/actions
[build-cirrus-img]: https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/KnetNLPModels.jl?logo=Cirrus%20CI
[build-cirrus-url]: https://cirrus-ci.com/github/JuliaSmoothOptimizers/KnetNLPModels.jl
[codecov-img]: https://codecov.io/gh/JuliaSmoothOptimizers/KnetNLPModels.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/gh/JuliaSmoothOptimizers/KnetNLPModels.jl
[doi-img]: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.822073-blue.svg
[doi-url]: https://doi.org/10.5281/zenodo.822073

## How to install
This module can be installed with the following command:
```julia
pkg> add KnetNLPModels
pkg> test KnetNLPModels
```

## Synopsis

A `KnetNLPModel` gives the user access to:
- the values of the neural network variables/weights `w`;
- the value of the objective/loss function `L(X, Y; w)` at `w` for a given minibatch `(X,Y)`;
- the gradient `∇L(X, Y; w)` of the objective/loss function at `w` for a given mini-batch `(X,Y)`.

In addition, it provides tools to:
- switch the minibatch used to evaluate the neural network;
- change the minibatch size;
- measure the neural network's accuracy at the current `w`.

## Example

This step-by-step example assume prior knowledge of [julia](https://julialang.org/) and [Knet.jl](https://github.com/denizyuret/Knet.jl.git).
See the [Julia tutorial](https://julialang.org/learning/) and the [Knet.jl tutorial](https://github.com/denizyuret/Knet.jl/tree/master/tutorial) for more details.

KnetNLPModels is an interface between [Knet.jl](https://github.com/denizyuret/Knet.jl.git)'s classification neural networks and [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl.git).

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
(d :: Dense)(x) = d.f.(d.w * mat(x) .+ d.b) # evaluate the layer for a given input x

# define a dense layer with input size i and output size o
Dense(i :: Int, o :: Int, f=sigm) = Dense(param(o, i), param0(o), f)
```
More layer types can be defined.
Once again, see the [Knet.jl tutorial](https://github.com/denizyuret/Knet.jl/tree/master/tutorial) for more details.

### Definition of the chained structure that evaluates the network and the loss function (negative log likelihood)
```julia
using KnetNLPModels

struct ChainNLL <: Chain # must derive from KnetNLPModels.Chain
  layers
  ChainNLL(layers...) = new(layers)
end
(c :: ChainNLL)(x) = (for l in c.layers; x = l(x); end; x) # evaluate the network for a given input x
(c :: ChainNLL)(x, y) = Knet.nll(c(x), y) # compute the loss function given input x and expected output y
(c :: ChainNLL)(data :: Tuple{T1,T2}) where {T1,T2} = c(first(data,2)...) # evaluate loss given data inputs (x,y)
(c :: ChainNLL)(d :: Knet.Data) = Knet.nll(c; data=d, average=true) # evaluate loss using a minibatch iterator d
```
The chained structure that defines the neural network must be a subtype of `KnetNLPModels.Chain`.

### Load datasets and define mini-batch
In this example, we use the [MNIST](https://juliaml.github.io/MLDatasets.jl/stable/datasets/MNIST/) dataset from [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl.git).
```julia
using MLDatasets

xtrn, ytrn = MNIST.traindata(Float32) # MNIST training dataset
ytrn[ytrn.==0] .= 10 # re-arrange indices
xtst, ytst = MNIST.testdata(Float32) # MNIST test dataset
ytst[ytst.==0] .= 10 # re-arrange indices

dtrn = minibatch(xtrn, ytrn, 100; xsize=(size(xtrn, 1), size(xtrn, 2), 1, :)) # training mini-batch
dtst = minibatch(xtst, ytst, 100; xsize=(size(xtst, 1), size(xtst, 2), 1, :)) # test mini-batch
```

## Definition of the neural network and KnetNLPModel
The following code defines `DenseNet`, a neural network composed of 3 dense layers, embedded in a `ChainNLL`.
```julia
DenseNet = ChainNLL(Dense(784, 200), Dense(200, 50), Dense(50, 10))
```
Next, we define the `KnetNLPModel` from the neural network.
By default, the size of each minibatch is 1% of the corresponding dataset offered by MNIST.
```julia
DenseNetNLPModel = _init_KnetNLPModel(DenseNet; size_minibatch=100, data_train=(xtrn, ytrn), data_test=(xtst, ytst))
```

`DenseNetNLPModel` will be either a `KnetNLPModelCPU` if the code runs on a CPU or a `KnetNLPModelGPU` if it runs on a GPU.
All the methods are defined for both `KnetNLPModelCPU` and `KnetNLPModelGPU`.

## Tools associated to a KnetNLPModel
The problem dimension `n`, where `w` ∈ ℝⁿ:
```julia
n = DenseNetNLPModel.meta.nvar
```

### Get the current network weights:
```julia
w = vector_params(DenseNetNLPModel)
```

### Evaluate the loss function (i.e. the objective function) at `w`:
```julia
NLPModels.obj(DenseNetNLPModel, w)
```
The length of `w` must be `DenseNetNLPModel.meta.nvar`.

### Evaluate the gradient at `w`:
```julia
NLPModels.grad!(DenseNetNLPModel, w, g)
```
The result is stored in `g :: Vector{T}`, `g` is similar to `v` (of size `DenseNetNLPModel.meta.nvar`).

### Evaluate the network accuracy:
The accuracy of the network can be evaluated with:
```julia
accuracy(DenseNetNLPModel)
```
`accuracy()` uses the full training dataset.
That way, the accuracy will not fluctuate with the minibatch.

## Default behavior
By default, the training minibatch that evaluates the neural network doesn't change between evaluations.
To change the training minibatch, use:
```julia
reset_minibatch_train!(DenseNetNLPModel)
```
The size of the new minibatch is the size define earlier.

The size of the training and test minibatch can be set to `1/p` the size of the dataset with:
```julia
set_size_minibatch!(DenseNetNLPModel, p) # p::Int > 1
```

## How to Cite

If you use KnetNLPModels.jl in your work, please cite using the format given in [`CITATION.bib`](https://github.com/JuliaSmoothOptimizers/KnetNLPModels.jl/blob/main/CITATION.bib).