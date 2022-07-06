"""
    f = obj(nlp, x)

Evaluate `f(x)`, the objective function of `nlp` at `x`.
"""
function NLPModels.obj(nlp::AbstractKnetNLPModel{T, S}, w::AbstractVector{T}) where {T, S}
  increment!(nlp, :neval_obj)
  set_vars!(nlp, w)
  f_w = nlp.chain(nlp.current_training_minibatch)
  return f_w
end

"""
    g = grad!(nlp, x, g)

Evaluate `∇f(x)`, the gradient of the objective function at `x` in place.
"""
function NLPModels.grad!(
  nlp::AbstractKnetNLPModel{T, S},
  w::AbstractVector{T},
  g::AbstractVector{T},
) where {T, S}
  @lencheck nlp.meta.nvar w g
  increment!(nlp, :neval_grad)
  set_vars!(nlp, w)
  L = Knet.@diff nlp.chain(nlp.current_training_minibatch)
  vars = Knet.params(nlp.chain)
  for (index, wᵢ) in enumerate(vars)
    nlp.layers_g[index] = Param(Knet.grad(L, wᵢ))
  end
  g .= Vector(vcat_arrays_vector(nlp.layers_g))
  return g
end
