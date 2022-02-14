import Knet.KnetArray

"""
    create_minibatch(X, Y, minibatch_size)

Create a minibatch of the data `X`, `Y` of size `minibatch_size`.
"""
create_minibatch(x_data, y_data, minibatch_size) = minibatch(x_data, y_data, minibatch_size; xsize=(size(x_data, 1), size(x_data, 2), 1, :))

"""
    vector_params(model)

Retrieves the variables as a vector from the vector of variable's arrays that composes `model`.
"""
vector_params(model) = Array(vcat_arrays_vector(params(model)))

"""
    vcat_arrays_vector(arrays_vector)

Flatten a vector of arrays to a vector. It keeps the order induce by Knet.cat1d to flatten an array.
"""
vcat_arrays_vector(arrays_vector) = vcat(Knet.cat1d.(arrays_vector)...)

"""
    build_from_array(vec, var_layer, index)

Inverse of the function Knet.cat1d, it generates a KnetArray similar to `var_layer` from `vec`.
The values are those of the vector in the range of index to index+consumed_index.
This method is not optimised, it consumes memory.
"""
function build_from_array(v::Vector{T}, var_layer::CuArray{T, N, CUDA.Mem.DeviceBuffer} where N, index::Int) where {T <: Number}
  dims = ndims(var_layer)
  size_var_layer = size(var_layer)
  tmp_array = Array{T, dims}(undef, size_var_layer)	
  cuArray = CuArray(tmp_array)
  product_dims = build_from_array!(cuArray, v, index)
  return (cuArray, product_dims)
end

function build_from_array!(cuArray::CuArray{T, N, CUDA.Mem.DeviceBuffer} where N, v::Vector{T}, index::Int) where {T <: Number}
	sizecuArray = reduce(*, size(cuArray))
	copyto!(cuArray, v[index+1:index+sizecuArray])
  return sizecuArray
end 

"""
    build_nested_array_from_vec(chain_ANN, v)

		build_nested_array_from_vec(model, v)
    
Build a vector of KnetArrays from `v` similar to Knet.params(model.chain) or Knet.params(`chain_ANN`).
It calls iteratively build_from_array to build each intermediary KnetArrays.
This method is not optimised, it consumes memory.
"""
build_nested_array_from_vec(model::KnetNLPModel{T, S, C}, v::Vector{T}) where {T, S, C} = build_nested_array_from_vec(model.chain, v)
function build_nested_array_from_vec(chain_ANN :: C, v::Vector{T}) where {C <: Chain, T <: Number}
  param_chain = params(chain_ANN) # :: Param
  size_param = mapreduce((var_layer -> reduce(*, size(var_layer))), +, param_chain)
  size_param == length(v) || error("Dimension of Vector v mismatch, function rebuild_nested_array $(size_param) != $(length(v))")

  param_value = (x -> x.value).(param_chain) # :: Vector{KnetArrays}
  vec_CuArray = build_nested_array_from_vec(param_value, v) 
  return vec_CuArray
end

function build_nested_array_from_vec(nested_array::Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}, v::Vector{T}) where {T <: Number}  
	vec_CuArray = map(i-> similar(nested_array[i]), 1:length(nested_array))
  build_nested_array_from_vec!(vec_CuArray, v)
  return vec_CuArray
end

build_nested_array_from_vec!(model::KnetNLPModel{T, S, C}, new_w :: Vector) where {T, S, C} = build_nested_array_from_vec!(model.nested_cuArray, new_w)
function build_nested_array_from_vec!(vec_CuArray::Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}, v::Vector{T}) where {T <: Number}
  index = 0
  for variable_layer in vec_CuArray
    consumed_indices = build_from_array!(variable_layer, v, index)
    index += consumed_indices
  end	
end

"""
    set_vars!(model, new_w)

    set_vars!(chain_ANN, new_w)

Set the variables of `model` or `chain` to new_w.
Build a vector of KnetArrays from v similar to Knet.params(model.chain).
Then it sets these variables to the nested array.
"""
set_vars!(vars::Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}, new_w :: Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}) where {T<: Number} = map(i -> vars[i].value .= new_w[i], 1:length(vars))
set_vars!(chain_ANN::C, nested_w :: Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}) where {C <: Chain, T <: Number} = set_vars!(params(chain_ANN), nested_w) 
function set_vars!(model::KnetNLPModel{T, S, C}, new_w :: Vector) where {T, S, C}
	build_nested_array_from_vec!(model, new_w)	
  set_vars!(model.chain, model.nested_cuArray)
  model.w .= new_w
end

# """
    # build_nested_array_from_vec(model, v)

# Build a vector of KnetArrays from `v` similar to Knet.params(model.chain).
# Call build_from_array iteratively to build each intermediary KnetArrays.
# This methods is not optimised, it consumes memory.
# """
# build_nested_array_from_vec(model::KnetNLPModel{T, S, C}, v::Vector{T}) where {T, S, C} = build_nested_array_from_vec(model.chain, v)
# function build_nested_array_from_vec!(model::KnetNLPModel{T, S, C}, v::Vector{T}) where {T, S, C} = 
  # param_chain = params(model.chain)
  # size_param = mapreduce((var_layer -> reduce(*, size(var_layer))), +, param_chain)
  # size_param == length(v) || error("Dimension of Vector v mismatch, function rebuild_nested_array $(size_param) != $(length(v))")
  # index = 0
  
  # for (i, variable_layer) in enumerate(param_chain)
    # vl = variable_layer.value
    # consumed_index = build_from_array!(v, vl, index, model.nested_knet_array[i])		
    # index += consumed_index
  # end
  
  # return param_chain
# end

# function build_nested_array_from_vec(nested_array::Array{CuArray{T, N, CUDA.Mem.DeviceBuffer},1}, v::Vector{T}) where {T <: Number, N}
	# vec_CuArray = map(i-> similar(nested_array[i]), 1:length(nested_array))
  # build_nested_array_from_vec!(nested_array, v, vec_CuArray)
  # return vec_CuArray
# end