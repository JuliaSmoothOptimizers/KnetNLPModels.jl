# KnetNLPModels.jl Tutorial

## Define the layers of interest (using Knet.jl)
```julia
	using Knet

	# Define convolution layer, see Knet.jl for more details
	struct Conv; w; b; f; end
	(c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
	Conv(w1,w2,cx,cy,f=relu) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f)

	# Define dense layer, see Knet.jl for more details
	struct Dense; w; b; f; p; end
	(d::Dense)(x) = d.f.(d.w * mat(dropout(x,d.p)) .+ d.b)
	Dense(i::Int,o::Int,f=sigm;pdrop=0.) = Dense(param(o,i), param0(o), f, pdrop)
```

## Define the chain structure that evaluates the network and the loss function 
```julia
	using KnetNLPModels

	struct Chainnll <: Chain # KnetNLPModels.Chain
		layers
		Chainnll(layers...) = new(layers)
	end
	(c::Chainnll)(x) = (for l in c.layers; x = l(x); end; x) # evaluates the network for a given input
	(c::Chainnll)(x,y) = Knet.nll(c(x),y) # compute the loss function given the input x and the expected result y
	(c::Chainnll)(d::Knet.Data) = Knet.nll(c; data=d, average=true) # compute the loss function for a minibatch
```
The chained structure that defines the neural network must be a subtype of KnetNLPModels.Chain otherwise there will be an error when the KnetNLPModel is instantiated. 

## Load the dataset required (MNIST is this example)
```julia
	xtrn,ytrn = MNIST.traindata(Float32) # training data of MNIST
	ytrn[ytrn.==0] .= 10 # re-arrange the indices
	xtst,ytst = MNIST.testdata(Float32) # test data of MNIST
	ytst[ytst.==0] .= 10 # re-arrange the indices
```

## Neural network definition and KnetNLPModel
```julia
	LeNet = Chainnll(Conv(5,5,1,20), Conv(5,5,20,50), Dense(800,500), Dense(500,10,identity)) # The network is defined from two concolution layers followed by two dense layers

	LeNetNLPModel = KnetNLPModel(LeNet; size_minibatch=100, data_train=(xtrn,ytrn), data_test=(xtst,ytst)) # define the KnetNLPModel
```
Define the neural network from the chained structure defined previously.
Then you can define the KnetNLPModel from the neural network.
By default the training and testing datasets are those of MNIST and the size of the minibatch is 100.


## Uses of a KnetNLPModel
Get the dimension of the problem:
```julia
LeNetNLPModel.meta.nvar
```
or 
```julia
length(vector_params(LeNetNLPModel))
```

Get the current variables of the network:
```julia
w = vector_params(LeNetNLPModel) # w is a Vector
```

Evaluate the network and the loss function (ie the objective):
```julia
NLPModels.obj(LeNetNLPModel, w)
```
The length of the vector w must be LeNetNLPModel.meta.nvar

Evaluate the loss function gradient at the point w (ie the gradient):
```julia
NLPModels.grad!(LeNetNLPModel, w, g)
```
The result is stored in g (of size LeNetNLPModel.meta.nvar)

The accuracy of the network can be evaluate with:
```julia
accuracy(LeNetNLPModel)
```



## Default behaviour
By default neither the training or testing minibatch that evaluates the neural network change between evaluations.
To change the training/testing minibatch use:

```julia
reset_minibatch_train!(LeNetNLPModel)
reset_minibatch_test!(LeNetNLPModel)
```
The size of the minibatch will be about the size define previously (may be improve in the future).
