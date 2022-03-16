module KnetNLPModels
  using Statistics: mean
  using CUDA, IterTools, Knet, MLDatasets, NLPModels

  export KnetNLPModel, Chain
  export vector_params, accuracy, reset_minibatch_test!, reset_minibatch_train!, set_size_minibatch!
  export build_nested_array_from_vec, build_nested_array_from_vec!

  abstract type Chain end 

  """ 
      KnetNLPModel{T, S, C <: Chain} <: AbstractNLPModel{T, S}

  Data structure that makes the interfaces between neural networks defined with Knet.jl and NLPModels.
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
		current_minibatch_training
		current_minibatch_testing
    w :: S # == Vector{T}
    layers_g :: Vector{Param}
    nested_cuArray :: Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}
  end

  """
      KnetNLPModel(chain_ANN, size_minibatch; data_train=data_train, data_test=data_test)

  Build a KnetNLPModel from the neural network represented by `chain_ANN`.
  `chain` is build by Knet.jl, see the [tutorial](https://paraynaud.github.io/KnetNLPModels.jl/dev/tutorial/) for more details.
  The other mandatory data are: `data_train`, `data_test` and the size of the minibatch `size_minibatch`.
	By default they are set to `MNIST` dataset with minibatchs of size 100.
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
		current_minibatch_training = rand(minibatch_train)
		current_minibatch_testing = rand(minibatch_test)

    nested_array = build_nested_array_from_vec(chain_ANN, x0)
    layers_g = similar(params(chain_ANN)) # create a Vector of layer variables

    return KnetNLPModel(meta, chain_ANN, Counters(), data_train, data_test, size_minibatch, minibatch_train, minibatch_test, current_minibatch_training, current_minibatch_testing, x0, layers_g, nested_array)
  end

	"""
			set_size_minibatch!(knetnlp, size_minibatch)
	
	Change the size of the minibatchs of training and testing of the `knetnlp`.
	After a call of `set_size_minibatch!`, if one want to use a minibatch of size `size_minibatch` it must use beforehand `reset_minibatch_train!`.
	"""
	function set_size_minibatch!(knetnlp :: KnetNLPModel, size_minibatch :: Int) 
		knetnlp.size_minibatch = size_minibatch
		knetnlp.minibatch_train = create_minibatch(knetnlp.data_train[1], knetnlp.data_train[2], knetnlp.size_minibatch)
		knetnlp.minibatch_test = create_minibatch(knetnlp.data_test[1], knetnlp.data_test[2], knetnlp.size_minibatch)
	end

  include("utils.jl")
  include("KnetNLPModels_methods.jl")
end 