import Knet.KnetArray

"""
    set_vars!(model, new_w)

    set_vars!(chain, new_w)

Set the variables of model or chain to new_w.
Build a vector of KnetArrays from v similar to Knet.params(model.chain).
Then it sets these variables to the nested array.
"""
set_vars!(vars, new_w :: Vector) = map(i -> vars[i].value .= new_w[i], 1:length(vars))
set_vars!(vars, new_w :: Vector{KnetArray}) = map(i -> vars[i].value .= new_w[i], 1:length(vars))
set_vars!(chain::T, new_w :: Vector) where T <: Chain = set_vars!(params(chain), build_nested_array_from_vec!(chain, new_w) ) 
function set_vars!(model::KnetNLPModel{T,S,C}, new_w :: Vector) where {T,S,C}
  param = build_nested_array_from_vec!(model, new_w)	
  set_vars!(param, model.nested_knet_array)
  model.w .= new_w
end 

"""
    build_nested_array_from_vec(model, v)

Build a vector of KnetArrays from v similar to Knet.params(model.chain).
Call build_array iteratively to build each intermediary KnetArrays.
This methods is not optimised, it consumes memory.
"""
build_nested_array_from_vec(model::KnetNLPModel{T,S,C}, v::Vector{T}) where {T,S,C} = begin build_nested_array_from_vec!(model, v); model.nested_knet_array end 
function build_nested_array_from_vec!(model::KnetNLPModel{T,S,C}, v::Vector{T}) where {T,S,C}
  param = params(model.chain)
  size_param = mapreduce((var_layer -> reduce(*,size(var_layer))), +, param)
  size_param == length(v) || error("Dimension of Vector v mismatch, function rebuild_nested_array $(size_param) != $(length(v))")
  index = 0
  
  for (i,variable_layer) in enumerate(param)
    vl = variable_layer.value #passe de Param{KnetArray} à KnetArray
    consumed_index = build_array!(v, vl, index, model.nested_knet_array[i])		
    index += consumed_index
  end
  
  return param
end

"""
    f = obj(nlp, x)

Evaluate ``f(x)``, the objective function of `nlp` at `x`.
"""
function NLPModels.obj(nlp :: KnetNLPModel{T,S,C}, w :: AbstractVector{T}) where {T,S,C}
  w != nlp.w && set_vars!(nlp, w)
  increment!(nlp, :neval_obj)  
  f_w = nlp.chain(nlp.minibatch_train)
  return f_w
end

"""
    g = grad!(nlp, x, g)

Evaluate ``∇f(x)``, the gradient of the objective function at `x` in place.
"""
function NLPModels.grad!(nlp :: KnetNLPModel{T,S,C}, w :: AbstractVector{T}, g :: AbstractVector{T}) where {T,S,C}
  length(w) != length(g) && error("different size vector: w, g: NLPModels.grad!")
  increment!(nlp, :neval_grad)
  w != nlp.w && set_vars!(nlp, w)
  L = Knet.@diff nlp.chain(nlp.minibatch_train)	
  vars = Knet.params(nlp.chain)	
  for (index,wᵢ) in enumerate(vars)
    nlp.layers_g[index] = Param(Knet.grad(L,wᵢ))
  end
  g .= vcat_arrays_vector(nlp.layers_g)
end

""" 
    reset_minibatch_train!(nlp)

Take a new minibatch for the KnetNLPModel. Usually use before a new evaluation.
"""
reset_minibatch_train!(nlp :: KnetNLPModel{T,S,C}) where {T,S,C} = nlp.minibatch_train = create_minibatch(nlp.data_train[1], nlp.data_train[2], nlp.size_minibatch)

"""
    reset_minibatch_test!(nlp)	

Take a new minibatch for the KnetNLPModel. Usually use before a new accuracy test.
"""
reset_minibatch_test!(nlp :: KnetNLPModel{T,S,C}) where {T,S,C} = nlp.minibatch_test = create_minibatch(nlp.data_test[1], nlp.data_test[2], nlp.size_minibatch)

""" 
    accuracy(nlp)

Computes the accuracy of the network nlp.chain given the data in nlp.minibatch_test.
"""
accuracy(nlp :: KnetNLPModel{T,S,C}) where {T,S,C} = Knet.accuracy(nlp.chain; data=nlp.minibatch_test)