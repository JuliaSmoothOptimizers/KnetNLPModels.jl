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
More sophisticated layers may be defined.
Once again, see the [Knet.jl tutorial](https://github.com/denizyuret/Knet.jl/tree/master/tutorial) for more details.

### Definition of the loss function (negative log likelihood)
Next, we define a chain that stacks layer, to evaluate them successively.
```@example KnetNLPModel
using KnetNLPModels

struct ChainNLL
  layers
  ChainNLL(layers...) = new(layers)
end
(c :: ChainNLL)(x) = (for l in c.layers; x = l(x); end; x) # evaluate the network for a given input x
(c :: ChainNLL)(x, y) = Knet.nll(c(x), y) # compute the loss function given input x and expected output y
(c :: ChainNLL)(data :: Tuple{T1,T2}) where {T1,T2} = c(first(data,2)...) # evaluate loss given data inputs (x,y)
(c :: ChainNLL)(d :: Knet.Data) = Knet.nll(c; data=d, average=true) # evaluate loss using a minibatch iterator d

DenseNet = ChainNLL(Dense(784, 200), Dense(200, 50), Dense(50, 10))
```

### Load datasets and define minibatch
In this example, and to fit the architecture proposed, we use the [MNIST](https://juliaml.github.io/MLDatasets.jl/stable/datasets/MNIST/) dataset from [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl.git).
```@example KnetNLPModel
using MLDatasets

# download datasets without user intervention
ENV["DATADEPS_ALWAYS_ACCEPT"] = true 

T = Float32
xtrain, ytrain = MLDatasets.MNIST(Tx = T, split = :train)[:] 
xtest, ytest = MLDatasets.MNIST(Tx = T, split = :test)[:] 
ytrain[ytrain.==0] .= 10 # re-arrange indices
xtest, ytest = MNIST.testdata(Float32) # MNIST test dataset
ytest[ytest.==0] .= 10 # re-arrange indices

dtrn = minibatch(xtrain, ytrain, 100; xsize=(size(xtrain, 1), size(xtrain, 2), 1, :)) # training minibatch
dtst = minibatch(xtest, ytest, 100; xsize=(size(xtest, 1), size(xtest, 2), 1, :)) # test minibatch
```

## Definition of the neural network and KnetNLPModel
Finally, we define the `KnetNLPModel` from the neural network.
By default, the size of each minibatch is 1% of the corresponding dataset offered by MNIST.
```@example KnetNLPModel
DenseNetNLPModel = KnetNLPModel(DenseNet; size_minibatch=100, data_train=(xtrain, ytrain), data_test=(xtest, ytest))
```
All the methods provided by KnetNLPModels.jl support both `CPU` and `GPU`.

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
* To select a minibatch randomly
```@example KnetNLPModel
rand_minibatch_train!(DenseNetNLPModel)
```
* To select the next minibatch from the current minibatch iterator (can be used in a loop to go over the whole dataset)
```@example KnetNLPModel
minibatch_next_train!(DenseNetNLPModel)
```
* Reset to the first minibatch
```@example KnetNLPModel
reset_minibatch_train!(DenseNetNLPModel)
```

The size of the new minibatch is the size defined earlier.

The size of the training and test minibatch can be set to `1/p` the size of the dataset with:
```@example KnetNLPModel
p = 120
set_size_minibatch!(DenseNetNLPModel, p) # p::Int > 1
```