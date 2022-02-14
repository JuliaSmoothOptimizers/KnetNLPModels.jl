# KnetNLPModels.jl Tutorial

This tutoriel suppose a prior knowledge about julia and Knet.jl.
The tutorial about [Knet.jl](https://github.com/denizyuret/Knet.jl/tree/master/tutorial)

## Define the layers of interest
```julia
  using Knet

  # Define dense layer
  struct Dense{T}
    w :: Param{Knet.KnetArrays.KnetMatrix{T}} # parameters of the layers
    b :: Param{Knet.KnetArrays.KnetVector{T}} # bias of the layer
    f # activation function
  end
  (d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b) # evaluates the layer for a given input x
  Dense(i :: Int, o :: Int, f=sigm) = Dense(param(o, i), param0(o), f) # define a dense layer whith an input size of i and an output size of o
```

## Definition of the chained structure that evaluates the network and the loss function 
```julia
  using KnetNLPModels

  struct Chainnll <: Chain # KnetNLPModels.Chain
    layers
    Chainnll(layers...) = new(layers)
  end
  (c::Chainnll)(x) = (for l in c.layers; x = l(x); end; x) # evaluates the network for a given input
  (c::Chainnll)(x, y) = Knet.nll(c(x), y) # computes the loss function given the input x and the expected result y
  (c::Chainnll)(d::Knet.Data) = Knet.nll(c; data=d, average=true) # computes the loss function for a minibatch
```

The chained structure that defines the neural network **must be a subtype** of KnetNLPModels.Chain otherwise there will be an **error** the KnetNLPModel is instantiated. 

## Load the dataset required (MNIST is this example)
```julia
  using MLDatasets
  xtrn, ytrn = MNIST.traindata(Float32) # MNIST's training database of type T
  ytrn[ytrn.==0] .= 10 # re-arrange the indices
  xtst, ytst = MNIST.testdata(Float32) # MNIST's testing database of type T
  ytst[ytst.==0] .= 10 # re-arrange the indices
```

## Neural network definition and KnetNLPModel
```julia
  DenseNet = Chainnll(Dense(784, 200), Dense(200, 50), Dense(50, 10)) 

  DenseNetNLPModel = KnetNLPModel(DenseNet; size_minibatch=100, data_train=(xtrn, ytrn), data_test=(xtst, ytst)) # define the KnetNLPModel
```
Define the neural network from the chained structure defined previously.
Then you can define the KnetNLPModel from the neural network.
By default the size of the minibatch is 100 and the dataset used is MNIST.

## Uses of a KnetNLPModel
Get the dimension of the problem:
```julia
DenseNetNLPModel.meta.nvar
```
or 
```julia
length(vector_params(DenseNetNLPModel))
```

### Get the current variables of the network:
```julia
w = vector_params(DenseNetNLPModel) :: Vector{T}
```

### Evaluate the network and the loss function (ie the objective):
```julia
NLPModels.obj(DenseNetNLPModel, w)
```
The length of the vector w must be DenseNetNLPModel.meta.nvar

### Evaluate the loss function gradient at the point w (ie the gradient):
```julia
NLPModels.grad!(DenseNetNLPModel, w, g)
```
The result is stored in g ```:: Vector{T}```(of size DenseNetNLPModel.meta.nvar)

The accuracy of the network can be evaluate with:
```julia
accuracy(DenseNetNLPModel)
```

## Default behaviour
By default neither the training or testing minibatch that evaluates the neural network change between evaluations.
To change the training/testing minibatch use:

```julia
reset_minibatch_train!(DenseNetNLPModel)
reset_minibatch_test!(DenseNetNLPModel)
```
The size of the minibatch will be about the size define previously (may be improved in the future to be dynamic).
