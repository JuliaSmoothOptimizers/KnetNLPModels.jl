# KnetNLPModels.jl Tutorial

This tutoriel suppose a prior knowledge about julia and [Knet.jl](https://github.com/denizyuret/Knet.jl.git).
See [Julia tutorial](https://julialang.org/learning/) and [Knet.jl tutorial](https://github.com/denizyuret/Knet.jl/tree/master/tutorial) for more details.

KnetNLPModels is an interface between [Knet.jl](https://github.com/denizyuret/Knet.jl.git)'s classification neural networks and [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl.git).

A KnetNLPModel allow acces to:
- the values of the neural network variable $w$;
- the objective function $\mathcal{L}(X,Y;w)$ of the loss function $\mathcal{L}$ at the point $w$ for a given minibatch $X,Y$
- the gradient $\nabla \mathcal{L}(X,Y;w)$ of the loss function at the point $w$ for a given mini-batch $X,Y$

In addition it provides tools to:
- Switch the minibatch used to evaluate the neural network
- Measure the neural network's accuracy at the current point for a given testing mini-batch

## Define the layers of interest
The following code define a dense layer as an evaluable julia structure.
```julia
  using Knet

  struct Dense{T}
    w :: Param{Knet.KnetArrays.KnetMatrix{T}} # parameters of the layers
    b :: Param{Knet.KnetArrays.KnetVector{T}} # bias of the layer
    f # activation function
  end
  (d :: Dense)(x) = d.f.(d.w * mat(x) .+ d.b) # evaluates the layer for a given input `x`
  Dense(i :: Int, o :: Int, f=sigm) = Dense(param(o, i), param0(o), f) # define a dense layer whith an input size of `i` and an output of size `o`
```
More layers can be defined, once again see [Knet.jl tutorial](https://github.com/denizyuret/Knet.jl/tree/master/tutorial) for more details.

## Definition of the chained structure that evaluates the network and the loss function 
```julia
  using KnetNLPModels

  struct Chainnll <: Chain # KnetNLPModels.Chain
    layers
    Chainnll(layers...) = new(layers)
  end
  (c :: Chainnll)(x) = (for l in c.layers; x = l(x); end; x) # evaluates the network for a given input `x`
  (c :: Chainnll)(x, y) = Knet.nll(c(x), y) # computes the loss function given the input `x` and the expected result `y`
	(c :: Chainnll)(data :: Tuple{T1,T2}) where {T1,T2} = c(first(data,2)...) # compute the loss function given the data inputs as a tuple `(x,y),. This lines is mandatory to compute single minibatch (ex : `(x,y) = rand(dtrn)` or `(x,y) = first(dtrn)`).
  (c :: Chainnll)(d :: Knet.Data) = Knet.nll(c; data=d, average=true) # computes the loss function negative log likelihood using a minibatch iterator `d`
```

The chained structure that defines the neural network **must be a subtype** (`<: Chain`) of `KnetNLPModels.Chain`.
If it is not the case an **error** will be raise when the KnetNLPModel is instantiated.

## Load datasets and mini-batch
Load a dataset from the package [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl.git).
In this example, the dataset used is [MNIST](https://juliaml.github.io/MLDatasets.jl/stable/datasets/MNIST/).
```julia
  using MLDatasets

  xtrn, ytrn = MNIST.traindata(Float32) # MNIST's training dataset
  ytrn[ytrn.==0] .= 10 # re-arrange the indices
  xtst, ytst = MNIST.testdata(Float32) # MNIST's testing dataset
  ytst[ytst.==0] .= 10 # re-arrange the indices

	dtrn = minibatch(xtrn, ytrn, 100; xsize=(size(xtrn, 1), size(xtrn, 2), 1, :)) # training mini-batch
	dtst = minibatch(xtst, ytst, 100; xsize=(size(xtst, 1), size(xtst, 2), 1, :)) # testing mini-batch
```

## Definition of the neural network and KnetNLPModel
The following code define `DenseNet`, a neural network composed by 3 dense layers.
The loss function applied est negative likelihood function define in the evaluation of `Chainnll`.
```julia
  DenseNet = Chainnll(Dense(784, 200), Dense(200, 50), Dense(50, 10)) 
```
Then you can define the KnetNLPModel from the neural network.
By default the size of the minibatch is 100 and the dataset used is MNIST.
```julia
  DenseNetNLPModel = KnetNLPModel(DenseNet; size_minibatch=100, data_train=(xtrn, ytrn), data_test=(xtst, ytst)) # define the KnetNLPModel
```



## Tools associated to a KnetNLPModel
The dimension of the problem $n$:
```julia
DenseNetNLPModel.meta.nvar
```
or in a costly way:
```julia
length(vector_params(DenseNetNLPModel))
```

### Get the current variables of the network:
```julia
w = vector_params(DenseNetNLPModel)
```

### Evaluate the loss function (i.e. the objective function) at the point $w$:
```julia
NLPModels.obj(DenseNetNLPModel, w)
```
The length of the vector w must be `DenseNetNLPModel.meta.nvar`

### Evaluate the loss function gradient at the point w (ie the gradient):
```julia
NLPModels.grad!(DenseNetNLPModel, w, g)
```
The result is stored in `g ::  Vector{T}`(of size `DenseNetNLPModel.meta.nvar`)

The accuracy of the network can be evaluate with:
```julia
accuracy(DenseNetNLPModel)
```

## Default behaviour
By default neither the training or the testing minibatch that evaluates the neural network change between evaluations.
To change the training/testing minibatch use:

```julia
reset_minibatch_train!(DenseNetNLPModel)
reset_minibatch_test!(DenseNetNLPModel)
```
The size of the minibatch will be about the size define previously (may be improved in the future to be dynamic).
