using MLDatasets
using IterTools: ncycle, takenth, takewhile
using StatsBase

using Knet
using KnetNLPModels
import Base.size

# include("layers.jl")
# include("losses.jl")
# include("script_results.jl")
include("AAAI24/layers.jl")
include("AAAI24/losses.jl")
include("AAAI24/script_results.jl")
include("AAAI24/PS_detection.jl")

create_minibatch = KnetNLPModels.create_minibatch


#= 
MNIST
=#

(xtrn, ytrn) = MNIST(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
(xtst, ytst) = MNIST(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10

C = 10 # number of classes
layer_PS = [24,15,1] # individual score neurons composing the successive searable layers
# in total, it contains respectively : 240, 150 and 10 neurons (e.g. layer_Ps .* C).

LeNet_NLL = () -> Chain_NLL(Conv(5,5,1,6; pool_option=1), Conv(5,5,6,16; pool_option=1), Dense(256,120), Dense(120,84), Dense(84,10,identity))
LeNet_PSLDP = () -> Chain_PSLDP(Conv(5,5,1,6; pool_option=1), Conv(5,5,6,16; pool_option=1), Dense(256,120), Dense(120,84), Dense(84,10,identity))
PSNet_NLL = () -> Chain_NLL(Conv(5,5,1,40; pool_option=1), Conv(5,5,40,30; pool_option=1), SL(480, C, layer_PS[1]), SL(C*layer_PS[1], C, layer_PS[2]), SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
PSNet_PSLDP = () -> Chain_PSLDP(Conv(5,5,1,40; pool_option=1), Conv(5,5,40,30; pool_option=1), SL(480, C, layer_PS[1]), SL(C*layer_PS[1], C, layer_PS[2]), SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
# LeNet 44426 # length(vector_params(LeNet_NLL()))
# PSNet 53780 # length(vector_params(PSNet_PSLDP()))
# every raw score depends only on 6340 weights # length(PS(PSNet_PSLDP())[1])
# every element function depends on 11588 weights # length(build_listes_indices(PSNet_PSLDP())[1,2]) (or [i,j] with 1 ≤ i != j ≤ C)
# 1092 common weights for every score

iter_max = 50
max_seed = 10
size_minibatch = 20
train_architecture_dataset_for_a_minibatchsize(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed, "MNIST")

size_minibatch = 100
train_architecture_dataset_for_a_minibatchsize(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed, "MNIST")


#= 
CIFAR10
=#

(xtrn, ytrn) = CIFAR10(Tx=Float32, split=:train)[:]; ytrn[ytrn.==0] .= 10
(xtst, ytst) = CIFAR10(Tx=Float32, split=:test)[:]; ytst[ytst.==0] .= 10
create_minibatch = KnetNLPModels.create_minibatch

C = 10 # number of classes
layer_PS = [35,15,1] 

element_function_indices[1,1]
LeNet_NLL = () -> Chain_NLL(Conv(5,5,3,6; pool_option=1), Conv(5,5,6,16; pool_option=1), Dense(400,200), Dense(200,100), Dense(100,10,identity))
LeNet_PSLDP = () -> Chain_PSLDP(Conv(5,5,1,6; pool_option=1), Conv(5,5,6,16; pool_option=1), Dense(400,200), Dense(200,100), Dense(100,10,identity))
PSNet_NLL = () -> Chain_NLL(Conv(5,5,3,60; pool_option=1), Conv(5,5,60,30; pool_option=1), SL(750, C, layer_PS[1]), SL(C*layer_PS[1], C, layer_PS[2]), SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
PSNet_PSLDP = () -> Chain_PSLDP(Conv(5,5,3,60; pool_option=1), Conv(5,5,60,30; pool_option=1), SL(750, C, layer_PS[1]), SL(C*layer_PS[1], C, layer_PS[2]), SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
# LeNet 103882 # length(vector_params(LeNet_NLL()))
# PSNet 81750 # length(vector_params(PSNet_PSLDP()))
# every raw score depends only on 12279 weights # length(PS(PSNet_PSLDP())[1])
# every element function depends on 19998 weights # length(build_listes_indices(PSNet_PSLDP())[1,2]) (or [i,j] with 1 ≤ i != j ≤ C)
# 4560 common weights for every score

iter_max = 50
max_seed = 10
size_minibatch = 20
train_architecture_dataset_for_a_minibatchsize(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed, "CIFAR10")

size_minibatch = 100
train_architecture_dataset_for_a_minibatchsize(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed, "CIFAR10")

# weights = vector_params(LeNet_NLL) # return the weights as a (Cu)Vector
# size(weights) #  return the size of the neural network