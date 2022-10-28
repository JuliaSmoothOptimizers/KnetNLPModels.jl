# KnetNLPModels.jl Tutorial

## Preliminaries
This step-by-step example assume prior knowledge of [julia](https://julialang.org/) and [Knet.jl](https://github.com/denizyuret/Knet.jl.git).
See the [Julia tutorial](https://julialang.org/learning/) and the [Knet.jl tutorial](https://github.com/denizyuret/Knet.jl/tree/master/tutorial) for more details.

### Define the layers of interest
The following code defines a dense layer as a callable julia structure for use on a CPU, via `Matrix` and `Vector` or on a GPU via [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and `CuArray`:
```@example KnetNLPModel
using Knet, CUDA

mutable struct Dense{T, Y}
  w :: Param{T}
  b :: Param{Y}
  f # activation function
end
(d :: Dense)(x) = d.f.(d.w * mat(x) .+ d.b) # evaluate the layer for a given input x

# define a dense layer with input size i and output size o
Dense(i :: Int, o :: Int, f=sigm) = Dense(param(o, i), param0(o), f)
```
More layer types can be defined.
Once again, see the [Knet.jl tutorial](https://github.com/denizyuret/Knet.jl/tree/master/tutorial) for more details.

### Definition of the chained structure that evaluates the network and the loss function (negative log likelihood)
```@example KnetNLPModel
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
```@example KnetNLPModel
using MLDatasets

# download datasets without user intervention
ENV["DATADEPS_ALWAYS_ACCEPT"] = true 

xtrn, ytrn = MNIST.traindata(Float32) # MNIST training dataset
ytrn[ytrn.==0] .= 10 # re-arrange indices
xtst, ytst = MNIST.testdata(Float32) # MNIST test dataset
ytst[ytst.==0] .= 10 # re-arrange indices

dtrn = minibatch(xtrn, ytrn, 100; xsize=(size(xtrn, 1), size(xtrn, 2), 1, :)) # training mini-batch
dtst = minibatch(xtst, ytst, 100; xsize=(size(xtst, 1), size(xtst, 2), 1, :)) # test mini-batch
```

## Definition of the neural network and KnetNLPModel
The following code defines `DenseNet`, a neural network composed of 3 dense layers, embedded in a `ChainNLL`.
```@example KnetNLPModel
DenseNet = ChainNLL(Dense(784, 200), Dense(200, 50), Dense(50, 10))
```
Next, we define the `KnetNLPModel` from the neural network.
By default, the size of each minibatch is 1% of the corresponding dataset offered by MNIST.
```@example KnetNLPModel
DenseNetNLPModel = KnetNLPModel(DenseNet; size_minibatch=100, data_train=(xtrn, ytrn), data_test=(xtst, ytst))
```

`DenseNetNLPModel` will be either a `KnetNLPModelCPU` if the code runs on a CPU or a `KnetNLPModelGPU` if it runs on a GPU.
All the methods are defined for both `KnetNLPModelCPU` and `KnetNLPModelGPU`.

## Tools associated with a KnetNLPModel
The problem dimension `n`, where `w` ∈ ℝⁿ:
```@example KnetNLPModel
n = DenseNetNLPModel.meta.nvar
```

### Get the current network weights:
```@example KnetNLPModel
w = vector_params(DenseNetNLPModel)
```

### Evaluate the loss function (i.e. the objective function) at `w`:
```@example KnetNLPModel
using NLPModels
NLPModels.obj(DenseNetNLPModel, w)
```
The length of `w` must be `DenseNetNLPModel.meta.nvar`.

### Evaluate the gradient at `w`:
```@example KnetNLPModel
g = similar(w)
NLPModels.grad!(DenseNetNLPModel, w, g)
```
The result is stored in `g :: Vector{T}`, `g` is similar to `v` (of size `DenseNetNLPModel.meta.nvar`).

### Evaluate the network accuracy:
The accuracy of the network can be evaluated with:
```@example KnetNLPModel
KnetNLPModels.accuracy(DenseNetNLPModel)
```
`accuracy()` uses the full training dataset.
That way, the accuracy will not fluctuate with the minibatch.

## Default behavior
By default, the training minibatch that evaluates the neural network doesn't change between evaluations.
To change the training minibatch, use one of the following methods:
* To select randomly a mini-batch 
```@example KnetNLPModel
rand_minibatch_train!(DenseNetNLPModel)
```
* To select the next mini-batch from current mini-batch iterator (Can be used in a loop to go over the entire dataset)
```@example KnetNLPModel
minibatch_next_train!(DenseNetNLPModel)
```
* To reset the first mini-batch
```@example KnetNLPModel
reset_minibatch_train!(DenseNetNLPModel)
```

The size of the new minibatch is the size define earlier.

The size of the training and test minibatch can be set to `1/p` the size of the dataset with:
```@example KnetNLPModel
p = 120
set_size_minibatch!(DenseNetNLPModel, p) # p::Int > 1
```