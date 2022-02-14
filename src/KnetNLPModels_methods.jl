import Knet.KnetArray

"""
    f = obj(nlp, x)

Evaluate `f(x)`, the objective function of `nlp` at `x`.
"""
function NLPModels.obj(nlp :: KnetNLPModel{T, S, C}, w :: AbstractVector{T}) where {T, S, C}
  w != nlp.w && set_vars!(nlp, w)
  increment!(nlp, :neval_obj)  
  f_w = nlp.chain(nlp.minibatch_train)
  return f_w
end

"""
    g = grad!(nlp, x, g)

Evaluate `∇f(x)`, the gradient of the objective function at `x` in place.
"""
function NLPModels.grad!(nlp :: KnetNLPModel{T, S, C}, w :: AbstractVector{T}, g :: AbstractVector{T}) where {T, S, C}
  length(w) != length(g) && error("different size vector: w, g: NLPModels.grad!")
  increment!(nlp, :neval_grad)
  w != nlp.w && set_vars!(nlp, w)
  L = Knet.@diff nlp.chain(nlp.minibatch_train)	
  vars = Knet.params(nlp.chain)	
  for (index, wᵢ) in enumerate(vars)
    nlp.layers_g[index] = Param(Knet.grad(L, wᵢ))
  end
  g .= vcat_arrays_vector(nlp.layers_g)
end

""" 
    reset_minibatch_train!(nlp)

Take a new minibatch for the KnetNLPModel. Usually use before a new evaluation.
"""
reset_minibatch_train!(nlp :: KnetNLPModel{T, S, C}) where {T, S, C} = nlp.minibatch_train = create_minibatch(nlp.data_train[1], nlp.data_train[2], nlp.size_minibatch)

"""
    reset_minibatch_test!(nlp)	

Take a new minibatch for the KnetNLPModel. Usually use before a new accuracy test.
"""
reset_minibatch_test!(nlp :: KnetNLPModel{T, S, C}) where {T, S, C} = nlp.minibatch_test = create_minibatch(nlp.data_test[1], nlp.data_test[2], nlp.size_minibatch)

""" 
    accuracy(nlp)

Computes the accuracy of the network nlp.chain given the data in nlp.minibatch_test.
"""
accuracy(nlp :: KnetNLPModel{T, S, C}) where {T, S, C} = Knet.accuracy(nlp.chain; data=nlp.minibatch_test)