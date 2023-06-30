# Training a LeNet architecture with JSO optimizers

## Neural network architecture
Define the layer Julia structures, how these structures are linked and evaluated.
We choose the negative log likelihood loss function.

```@example LeNetTraining
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

## MNIST dataset loading
Accordingly to LeNet architecture, we chose the MNIST dataset [@lecun-bouttou-bengio-haffner1998] from MLDataset:
```@example LeNetTraining
using MLDatasets
ENV["DATADEPS_ALWAYS_ACCEPT"] = true 

T = Float32
xtrain, ytrain = MLDatasets.MNIST(Tx = T, split = :train)[:] 
xtest, ytest = MLDatasets.MNIST(Tx = T, split = :test)[:] 

ytrain[ytrain.==0] .= output_classes # re-arrange indices
ytest[ytest.==0] .= output_classes # re-arrange indices

data_train = (xtrain, ytrain)
data_test = (xtest, ytest)
```

## KnetNLPModel instantiation
From these elements, we transfer the Knet.jl architecture to a `KnetNLPModel`:
```@example LeNetTraining 
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
```@example LeNetTraining 
using JSOSolvers

max_time = 30.
callback = (nlp, solver, stats) -> KnetNLPModels.minibatch_next_train!(nlp)
solver_stats = R2(LeNetNLPModel; callback, max_time)

test_accuracy = KnetNLPModels.accuracy(LeNetNLPModel)
```
Other solvers may also be applied for any KnetNLPModel:
```@example LeNetTraining 
solver_stats = lbfgs(LeNetNLPModel; callback, max_time)

test_accuracy = KnetNLPModels.accuracy(LeNetNLPModel)
```
In the last case, `LeNetNLPModel` is wrapped with a LSR1 approximation of loss Hessian to define a `lsr1_LeNet` NLPModel.
The callback must be adapted to work on `LeNetNLPModel` which is accessible from `lsr1_LeNet.model`.
```@example LeNetTraining
using NLPModelsModifiers

lsr1_LeNet = NLPModelsModifiers.LSR1Model(LeNetNLPModel)

callback_lsr1 = (lsr1_nlpmodel, solver, stats) -> KnetNLPModels.minibatch_next_train!(lsr1_nlpmodel.model)
solver_stats = trunk(lsr1_LeNet; callback = callback_lsr1, max_time)

test_accuracy = KnetNLPModels.accuracy(LeNetNLPModel)
```