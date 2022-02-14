"""
    f = obj(nlp, x)

Evaluate `f(x)`, the objective function of `nlp` at `x`.
"""
function NLPModels.obj(nlp :: KnetNLPModel{T, S, C}, w :: AbstractVector{T}) where {T, S, C}
	increment!(nlp, :neval_obj)
	set_vars!(nlp, w)
  f_w = nlp.chain(nlp.minibatch_train)
  return f_w
end

"""
    g = grad!(nlp, x, g)

Evaluate `∇f(x)`, the gradient of the objective function at `x` in place.
"""
function NLPModels.grad!(nlp :: KnetNLPModel{T, S, C}, w :: AbstractVector{T}, g :: AbstractVector{T}) where {T, S, C}
	@lencheck w g
	increment!(nlp, :neval_grad)
	set_vars!(nlp, w)  
  L = Knet.@diff nlp.chain(nlp.minibatch_train)	
  vars = Knet.params(nlp.chain)	
  for (index, wᵢ) in enumerate(vars)
    nlp.layers_g[index] = Param(Knet.grad(L, wᵢ))
  end
  g .= vcat_arrays_vector(nlp.layers_g)
	return g
end