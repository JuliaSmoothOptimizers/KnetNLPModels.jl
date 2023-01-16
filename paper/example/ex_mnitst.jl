# This is a simple example of training KnetNLPModels with Lenet5 and MNIST on CPU/GPU
# For more tools and indetailed example check KnetNLPModelsProblems.jl
#

using CUDA, IterTools
using JSOSolvers
using LinearAlgebra
using Random
using Printf
using NLPModels
using SolverCore
using Plots
using Knet, Images, MLDatasets
using Statistics: mean
using KnetNLPModels
using Knet:
    Knet,
    conv4,
    pool,
    mat,
    nll,
    accuracy,
    progress,
    sgd,
    param,
    param0,
    dropout,
    relu,
    minibatch,
    Data

# ####################################################
# # used in the callback of R2 for training deep learning model 
# ####################################################

mutable struct StochasticR2Data
    epoch::Int
    i::Int
    # other fields as needed...
    max_epoch::Int
    acc_arr::Vector{Float64}
    train_acc_arr::Vector{Float64}
    epoch_arr::Vector{Float64}
    grads_arr::Vector{Float64} #TODO fix this type to be dynamic
    ϵ::Float64 #TODO Fix with type T
end

# ####################################################
# # different Layers used
# ####################################################
struct Conv
    w::Any
    b::Any
    f::Any
end
(c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
Conv(w1, w2, cx, cy, f = relu) = Conv(param(w1, w2, cx, cy), param0(1, 1, cy, 1), f)

struct Dense
    w::Any
    b::Any
    f::Any
    p::Any
end
# (d::Dense)(x) = d.f.(mat_zero_mean(d.w, mat(dropout(x, d.p))) .+ d.b)
(d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b) # todo change * in future 

Dense(i::Int, o::Int, f = sigm; pdrop = 0.0) = Dense(param(o, i), param0(o), f, pdrop)

struct Chainnll <: KnetNLPModels.Chain
    layers::Any
    Chainnll(layers...) = new(layers)
end
(c::Chainnll)(x) = (for l in c.layers
    x = l(x)
end;
x)
(c::Chainnll)(x, y) = Knet.nll(c(x), y)  # nécessaire
(c::Chainnll)(data::Tuple{T1,T2}) where {T1,T2} = c(first(data, 2)...)
(c::Chainnll)(d::Knet.Data) = Knet.nll(c; data = d, average = true)


"""
Lenet model
Data need to be MNIST (#todo test with CIFAR10))
Returns Knet model and KnetNLPModel model
"""
function lenet_prob(xtrn, ytrn, xtst, ytst; minibatchSize = 100)

    LeNet = Chainnll(
        Conv(5, 5, 1, 20), # Conv(filter_length, filter_width, number of input channels from previous layer :n filters, 3 for RGB, 1 for Grey..., number of filters)
        Conv(5, 5, 20, 50),
        Dense(800, 500),
        Dense(500, 10, identity),
    )

    LeNetNLPModel = KnetNLPModel(
        LeNet;
        data_train = (xtrn, ytrn),
        data_test = (xtst, ytst),
        size_minibatch = minibatchSize,
    )


    return LeNet, LeNetNLPModel

end

####################################################
# R2 supports callback function so we don't need to 
# init every epoch 
# currently we have Cb stopping the DNN model after
# some iterations, but TBD other stopping conditions
####################################################

function cb(
    nlp,
    solver,
    stats,
    data::StochasticR2Data,
)
    # Max epoch
    if data.epoch == data.max_epoch
        stats.status = :user
        return
    end

    data.i = KnetNLPModels.minibatch_next_train!(nlp)
    if data.i == 1   # once one epoch is finished     
        # reset
        data.grads_arr = []
        data.epoch += 1
        acc = KnetNLPModels.accuracy(nlp) # accracy of the minibatch on the test Data
        train_acc = Knet.accuracy(nlp.chain; data = nlp.training_minibatch_iterator) #TODO minibatch acc.
        @info @sprintf "Current epoch:  %d out of max epoch: %d, \t train_acc: %f \t test_acc: %f" data.epoch data.max_epoch train_acc acc
        append!(data.train_acc_arr, train_acc) #TODO fix this to save the acc
        append!(data.acc_arr, acc) #TODO fix this to save the acc
        append!(data.epoch_arr, data.epoch)
    end

end
####################################################
# Training using R2
####################################################


function train_knetNLPmodel!(
    modelNLP,
    solver,
    xtrn,
    ytrn;
    mbatch = 64, #128     #TODO see if we need this , in future we can update the number of batch size in different epoch
    mepoch = 10, # 100
    verbose = -1,
    atol::T = √eps(T),
    rtol::T = √eps(T),
    η1 = eps(T)^(1 / 4),
    η2 = T(0.95),
    γ1 = T(1 / 2),
    γ2 = 1 / γ1,
    σmin = zero(T),# change this
    β::T = T(0),
    max_time = Inf, 
) where {T}
    stochastic_data = StochasticR2Data(0, 0, mepoch, [], [], [], [], atol)
    solver_stats = solver(
        modelNLP;
        atol = atol,
        rtol = rtol,
        η1 = η1,
        η2 = η2,
        γ1 = γ1,
        γ2 = γ2,
        σmin = σmin,
        β = β,
        max_time = max_time,
        verbose = verbose,
        callback = (nlp, solver, stats, nlp_param) ->
            cb(nlp, solver, stats, stochastic_data),
    )
    return stochastic_data

end


####################################################
# utilities
####################################################
"""
    all_accuracy(nlp::AbstractKnetNLPModel)
Compute the accuracy of the network `nlp.chain` given the data in `nlp.tests`.
uses the whole test data sets"""
all_accuracy(nlp::AbstractKnetNLPModel) = Knet.accuracy(nlp.chain; data = nlp.data_test)

# this functions load MNIST data or CIFAR10
function loaddata(data_flag, T)
    if (data_flag == 1)
        @info("Loading MNIST...")
        xtrn, ytrn = MNIST.traindata(T)
        ytrn[ytrn.==0] .= 10
        xtst, ytst = MNIST.testdata(T)
        ytst[ytst.==0] .= 10
        @info("Loaded MNIST")
    else # CIFAR data
        @info("Loading CIFAR 10...")
        xtrn, ytrn = CIFAR10.traindata(T)
        xtst, ytst = CIFAR10.testdata(T)
        xtrn = convert(Knet.array_type[], xtrn)
        xtst = convert(Knet.array_type[], xtst)

        mn = mean(xtrn, dims = (1, 2, 4))
        xtrn = xtrn .- mn
        xtst = xtst .- mn
        @info("Loaded CIFAR 10")
    end
    return (xtrn, ytrn), (xtst, ytst)
end



# ####################################################
# # Main
# ####################################################

T = Float32 # we can select the precision we want 
# Knet.atype() = Array{T}
# if GPU is avaiable do it on GPU
if CUDA.functional()
    Knet.array_type[] = CUDA.CuArray{T}
else
    Knet.array_type[] = Array{T}
end


(xtrn, ytrn), (xtst, ytst) = loaddata(1, T)

# size of minibatch 
m = 125
max_epochs = 50

knetModel, myModel = lenet_prob(xtrn, ytrn, xtst, ytst, minibatchSize = m)



println("Training R2 with KNET")
trained_model = train_knetNLPmodel!(
    myModel,
    JSOSolvers.R2,
    xtrn,
    ytrn;
    mbatch = m,
    mepoch = max_epochs,
    verbose = 1,
    # β = T(0.3),
    # atol = T(0.05),
    # rtol = T(0.05),
    # η1 = eps(T)
)


res = trained_model
epochs = res.epoch_arr
acc = res.acc_arr
train_acc = res.train_acc_arr


# println("Training SGD with KNET")
# # Train Knet
# trained_model_knet =
#     train_knet(knetModel, xtrn, ytrn, xtst, ytst; mbatch = m, mepoch = max_epochs) #TODO some reason when mepoch=max_epochs, will give error , maybe Int(max_epochs)
# res_knet = trained_model_knet[2]
# epochs_knet = res_knet[:, 1]
# acc_knet = res_knet[:, 2]
# train_acc_knet = res_knet[:, 3]


# fig = plot(
#     epochs,
#     # title = " test accuracy vs Epoch",
#     markershape = :star4,
#     acc,
#     label = "test accuracy R2",
#     legend = :bottomright,
#     xlabel = "epoch",
#     ylabel = "accuracy",
# )
# plot!(fig, epochs, markershape = :star1, acc_knet, label = "test accuracy SGD")

# # plotSamples(myModel, xtrn, ytrn, MNIST; samples=10)

# fig = plot(
#     epochs,
#     title = "train accuracy vs Epoch on Float32",
#     markershape = :star4,
#     train_acc,
#     label = "train accuracy R2",
#     legend = :bottomright,
#     xlabel = "epoch",
#     ylabel = "accuracy",
# )
# plot!(fig, epochs, markershape = :star1, train_acc_knet, label = "train accuracy SGD")


# # plotSamples(myModel, xtrn, ytrn, MNIST; samples=10)


#Plot all

fig = plot(
    epochs,
    # title = "accuracy vs Epoch on Float32",
    markershape = :star1,
    acc,
    label = "test accuracy R2",
    legend = :bottomright,
    xlabel = "epoch",
    ylabel = "accuracy",
)



plot!(
    fig,
    epochs,
    markershape = :star1,
    train_acc,
    label = "train accuracy R2",
    legend = :bottomright,
    linestyle = :dash,
)


# plot!(fig, epochs, markershape = :star4, acc_knet, label = "test accuracy SGD")

# plot!(
#     fig,
#     epochs,
#     markershape = :star4,
#     train_acc_knet,
#     label = "train accuracy SGD",
#     linestyle = :dot,
# )

savefig("run_LENET_MNIST.png")