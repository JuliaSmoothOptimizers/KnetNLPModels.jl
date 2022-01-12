using Knet, MLDatasets, IterTools
using Statistics: mean
using NLPModels

include("utils.jl")
include("ChainedNLPModel.jl")

export ChainNLPModel, Chain