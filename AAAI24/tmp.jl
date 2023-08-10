using MLDatasets
using IterTools: ncycle, takenth, takewhile
using StatsBase

using Knet
using KnetNLPModels
import Base.size

using Revise

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
dtrn = create_minibatch(xtrn, ytrn, size_minibatch)	 	 
dtst = create_minibatch(xtst, ytst, size_minibatch)

C = 10 # number of classes
layer_PS = [24,15,1] # individual score neurons composing the successive searable layers
# in total, it contains respectively : 240, 150 and 10 neurons (e.g. layer_Ps .* C).

LeNet_NLL = () -> Chain_NLL(Conv(5,5,1,6; pool_option=1), Conv(5,5,6,16; pool_option=1), Dense(256,120), Dense(120,84), Dense(84,10,identity))
LeNet_PSLDP = () -> Chain_PSLDP(Conv(5,5,1,6; pool_option=1), Conv(5,5,6,16; pool_option=1), Dense(256,120), Dense(120,84), Dense(84,10,identity))
LeNet_PSLDP3 = () -> Chain_PSLDP3(Conv(5,5,1,6; pool_option=1), Conv(5,5,6,16; pool_option=1), Dense(256,120), Dense(120,84), Dense(84,10,identity))
PSNet_NLL = () -> Chain_NLL(Conv(5,5,1,40; pool_option=1), Conv(5,5,40,30; pool_option=1), SL(480, C, layer_PS[1]), SL(C*layer_PS[1], C, layer_PS[2]), SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
PSNet_PSLDP = () -> Chain_PSLDP(Conv(5,5,1,40; pool_option=1), Conv(5,5,40,30; pool_option=1), SL(480, C, layer_PS[1]), SL(C*layer_PS[1], C, layer_PS[2]), SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
PSNet_PSLDP3 = () -> Chain_PSLDP3(Conv(5,5,1,40; pool_option=1), Conv(5,5,40,30; pool_option=1), SL(480, C, layer_PS[1]), SL(C*layer_PS[1], C, layer_PS[2]), SL(C*layer_PS[2], C, layer_PS[3]; f=identity))
# LeNet 44426 # length(vector_params(LeNet_NLL()))
# PSNet 53780 # length(vector_params(PSNet_PSLDP()))
# every raw score depends only on 6340 weights # length(PS(PSNet_PSLDP())[1])
# every element function depends on 11588 weights # length(build_listes_indices(PSNet_PSLDP())[1,2]) (or [i,j] with 1 ≤ i != j ≤ C)
# 1092 common weights for every score

iter_max = 10
max_seed = 1
size_minibatch = 20

seeded_adam_trains(LeNet_PSLDP3, dtrn, dtst; iter_max, size_minibatch, name_architecture="LeNet_PSLDP3", name_dataset = "MNIST", max_seed)
seeded_adam_trains(PSNet_PSLDP3, dtrn, dtst; iter_max, size_minibatch, name_architecture="PSNet_PSLDP3", name_dataset = "MNIST", max_seed)

seeded_adam_trains(LeNet_NLL, dtrn, dtst; iter_max, size_minibatch, name_architecture="LeNet_NLL", name_dataset = "MNIST", max_seed)
seeded_adam_trains(LeNet_PSLDP, dtrn, dtst; iter_max, size_minibatch, name_architecture="LeNet_PSLDP", name_dataset = "MNIST", max_seed)
seeded_adam_trains(LeNet_PSLDP3, dtrn, dtst; iter_max, size_minibatch, name_architecture="LeNet_PSLDP3", name_dataset = "MNIST", max_seed)
seeded_adam_trains(PSNet_NLL, dtrn, dtst; iter_max, size_minibatch, name_architecture="PSNet_NLL", name_dataset = "MNIST", max_seed)
seeded_adam_trains(PSNet_PSLDP, dtrn, dtst; iter_max, size_minibatch, name_architecture="PSNet_PSLDP", name_dataset = "MNIST", max_seed)
seeded_adam_trains(PSNet_PSLDP3, dtrn, dtst; iter_max, size_minibatch, name_architecture="PSNet_PSLDP3", name_dataset = "MNIST", max_seed)