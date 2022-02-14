module KnetNLPModels
  using Knet, MLDatasets, IterTools
  using Statistics: mean
  using NLPModels

  export KnetNLPModel, Chain
  export vector_params, accuracy, reset_minibatch_test!, reset_minibatch_train!

  abstract type Chain end 

  """ 
  KnetNLPModel{T, S, C <: Chain} <: AbstractNLPModel{T, S}

  Data structure that interfaces neural networks defined with Knet.jl as an NLPModel.
  
  """
  mutable struct KnetNLPModel{T, S, C <: Chain} <: AbstractNLPModel{T, S}
    meta :: NLPModelMeta{T, S}
    chain :: C
    counters :: Counters
    data_train
    data_test
    size_minibatch :: Int
    minibatch_train
    minibatch_test
    w :: S # == Vector{T}
    layers_g :: Vector{Param}
    nested_knet_array :: Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}
  end

  """
      KnetNLPModel(chain_ANN, size_minibatch; data_train=data_train, data_test=data_test)

  Build a KnetNLPModel from the neural network represented by `chain_ANN`.
  `chain` is build Knet, see the [tutorial](https://paraynaud.github.io/KnetNLPModels.jl/dev/tutorial/) for more details.

  """
  function KnetNLPModel(chain_ANN :: T;
            size_minibatch :: Int=100,
            data_train = begin (xtrn, ytrn) = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10; (xtrn, ytrn) end,
            data_test = begin (xtst, ytst) = MNIST.testdata(Float32); ytst[ytst.==0] .= 10; (xtst, ytst) end
            ) where T <: Chain
    x0 = vector_params(chain_ANN)
    n = length(x0)
    meta = NLPModelMeta(n, x0=x0)
    
    xtrn = data_train[1]
    ytrn = data_train[2]
    xtst = data_test[1]
    ytst = data_test[2]
    minibatch_train = create_minibatch(xtrn, ytrn, size_minibatch)	 	 	
    minibatch_test = create_minibatch(xtst, ytst, size_minibatch)

    nested_array = build_nested_array_from_vec(chain_ANN, x0)
    layers_g = similar(params(chain_ANN)) # create a Vector of layer variables

    return KnetNLPModel(meta, chain_ANN, Counters(), data_train, data_test, size_minibatch, minibatch_train, minibatch_test, x0, layers_g, nested_array)
  end

  include("utils.jl")
  include("KnetNLPModels_methods.jl")
end 