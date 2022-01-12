module ChainedNLPModel
	using Knet, MLDatasets, IterTools
	using Statistics: mean
	using NLPModels


	abstract type Chain end 

	""" 
	ChainNLPModel
	Data structure that interface neural network define with Knet.jl feature to a NLPModel.
	"""
	mutable struct ChainNLPModel{T,S,C<:Chain} <: NLPModels.AbstractNLPModel{T,S}
		meta :: NLPModelMeta{T,S}
		chain :: C
		counters :: Counters
		data_train
		data_test
		size_minibatch :: Int
		minbatch_train
		minbatch_test
		w :: Vector
		layers_g :: Vector{Param}
		nested_knet_array :: Vector{KnetArray{T}}
	end

	# The Data base is MNIST by default
	function ChainNLPModel(chain::Chain;
						size_minbatch=100,
						data_train = begin (xtrn,ytrn)=MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10 end,
						data_test = begin (xtst,ytst)=MNIST.testdata(Float32);  ytst[ytst.==0] .= 10 end
						)
		x0 = vector_params(chain)
		n = length(x0)
		meta = NLPModelMeta(n,x0=x0) #32Lio les 3 lignes
		
		minbatch_train = create_minibatch(xtrn, ytrn, size_minbatch)	 	 	
		minbatch_test = create_minibatch(xtst, ytst, size_minbatch)

		w = vector_params(chain)
		nested_array = build_nested_array_from_vec(chain,w)
		layers_g = similar(params(chain)) # create un Vector of layer variables

		return ChainNLPModel(meta, chain, Counters(), data_train, data_test, size_minbatch, minbatch_train, minbatch_test, w, layers_g, nested_array)
	end

	export ChainNLPModel, Chain

	# Base.MainInclude.include("utils.jl")
	# Base.MainInclude.include("ChainedNLPModel_methods.jl")
	include("utils.jl")
	include("ChainedNLPModel_methods.jl")
end 