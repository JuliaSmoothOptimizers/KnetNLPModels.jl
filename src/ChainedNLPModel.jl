abstract type Chain end 


""" 
		ChainNLPModel
Data structure that interface neural network define with Knet.jl feature to a NLPModel.
"""
mutable struct ChainNLPModel{T,S,C<:Chain} <: AbstractNLPModel{T,S}
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


"""
		set_vars!(model, new_w)
Set the variables of the model to new_w.
There are variants based on model.chain and new_w, or the nested_array
"""
set_vars!(vars, new_w :: Vector) = map(i -> vars[i].value .= new_w[i], [1:length(vars);] )
# set_vars!(vars, new_w :: Vector) = map(i -> ((x,y)-> x.=y).(vars[i].value,new_w[i]), [1:length(vars);] )
set_vars!(vars, new_w :: Vector{KnetArray}) = map(i -> vars[i].value .= new_w[i], [1:length(vars);] )
# set_vars!(vars, new_w :: Vector{KnetArray}) = map(i -> ((x,y)-> x=y).(vars[i].value,new_w[i]), [1:length(vars);] )
set_vars!(chain::T, new_w :: Vector) where T <: Chain = set_vars!(params(chain), build_nested_array_from_vec!(chain, new_w) ) 
function set_vars!(model::ChainNLPModel, new_w :: Vector) 
	param = build_nested_array_from_vec!(model, new_w)	
	set_vars!(param, model.nested_knet_array)
	model.w .= new_w
end 
((x,y) -> x+y).(ones(20),ones(20))
"""
		build_nested_array_from_vec(model, v)
Builds a vector of KnetArrays corresponding to the chain's shape of model.
It calls iteratively build_array to build each intermediary KnetArrays.
This methods is not optimised, it consumes memory.
"""
build_nested_array_from_vec(model::ChainNLPModel, v::Vector{T}) where T <: Number = begin build_nested_array_from_vec!(model, v); model.nested_knet_array end 
function build_nested_array_from_vec!(model::ChainNLPModel, v::Vector{T}) where T <: Number
	param = params(model.chain)
	flatten_params = mapreduce_cat1d_vcat(param)
	size(flatten_params) == size(v) || error("Le vecteur n'est pas à la bonne dimension, function rebuild_nested_array $(size(flatten_params)) != $(size(v))")
	index = 0
	
	for (i,variable_layer) in enumerate(param)
		vl = variable_layer.value #passe de Param{KnetArray} à KnetArray
		consumed_index = build_array!(v, vl, index, model.nested_knet_array[i])		
		index += consumed_index
	end
	
	return param
end

# The Data structure is MNIST by default
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


function NLPModels.obj(nlp :: ChainNLPModel, w :: AbstractVector{Y}) where Y <: Number
	increment!(nlp, :neval_obj)
	w != nlp.w && set_vars!(nlp, w)
	f_w = nlp.chain(nlp.minbatch_train)
	return f_w
end


NLPModels.grad(nlp :: ChainNLPModel, w :: AbstractVector{Y}) where Y <: Number = begin g = similar(w); grad!(nlp, w, g); return g end
function NLPModels.grad!(nlp :: ChainNLPModel, w :: AbstractVector{Y}, g :: AbstractVector{Y}) where Y <: Number
	length(w) != length(g) && error("different size vector: w, g: NLPModels.grad!")
	increment!(nlp, :neval_grad)
	w != nlp.w && set_vars!(nlp, w)
	L = Knet.@diff nlp.chain(nlp.minbatch_train)	
	vars = Knet.params(nlp.chain)	
	for (index,wᵢ) in enumerate(vars)
		nlp.layers_g[index] = Param(Knet.grad(L,wᵢ))
	end
	g .= mapreduce_cat1d_vcat(nlp.layers_g)
end

""" 
		reset_minibatch_train!(nlp)
Take a new minibatch for the ChainedNLPModel. Usually use before a new evaluation.
"""
reset_minibatch_train!(nlp :: ChainNLPModel) = nlp.minbatch_train = create_minibatch(nlp.data_train[1], nlp.data_train[2], nlp.size_minibatch)

"""
		reset_minibatch_test!(nlp)	
Take a new minibatch for the ChainedNLPModel. Usually use before a new accuracy test.
"""
reset_minibatch_test!(nlp :: ChainNLPModel) = nlp.minbatch_test = create_minibatch(nlp.data_test[1], nlp.data_test[2], nlp.size_minibatch)

""" 
		accuracy(nlp)
Computes the accuracy of the network nlp.chain given the data in nlp.minbatch_test.
"""
accuracy(nlp :: ChainNLPModel{T,S,C}) where {T,S,C} = Knet.accuracy(nlp.chain; data=nlp.minbatch_test)