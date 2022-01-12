import Knet.KnetArray


"""
		create_minibatch(X,Y,minibatch_size)
create a minibatch of the data (X,Y) of minibatch_size's size
"""
create_minibatch(x_data, y_data, minibatch_size) = minibatch(x_data, y_data, minibatch_size; xsize=(size(x_data,1),size(x_data,2),1,:))

# Fonctions utiles faisant le lien entre la représentation des variables en vecteur ou en vecteur de tableau (de dimension ≤ 4)
	# Passe du vecteur de tableau à un vecteur
vector_params(model) = Array(mapreduce_cat1d_vcat(params(model))) # fonctions récupérant un vecteur de variables directement du model

"""
		mapreduce_cat1d_vcat(arrays_vector)
Flatten a vector of arrays to a vector. It keeps the order induce by Knet.cat1d to flatten an array.
"""
mapreduce_cat1d_vcat(arrays_vector) = mapreduce(identity, vcat, Knet.cat1d.(arrays_vector)) # Applati un vecteur de tableaux à dimensions variables

"""
		build_array(vec, var_layer, index)
Inverse of the function Knet.cat1d, it generates a KnetArray similar to var_layer from the vector vec.
The values are those of the vector in the range of index to index+consumed_index.
This methods is not optimised, it consumes memory.
"""
function build_array(v::Vector{T}, var_layer::KnetArray{T,N}, index::Int) where T <: Number where N
	dims = ndims(var_layer)
	size_var_layer = size(var_layer)
	tmp_array = Array{T,dims}(undef, size_var_layer)	
	knetArray = KnetArray(tmp_array)
	product_dims = build_array!(v,var_layer,index,knetArray)
	return (knetArray, product_dims)
end

function build_array!(v::Vector{T}, var_layer::KnetArray{T,N}, index::Int, knetarray::KnetArray{T,N}) where T <: Number where N
	map(i -> knetarray[i] = v[index+i] ,[i for i in eachindex(var_layer)])
	product_dims = length([i for i in eachindex(var_layer)])
	return product_dims
end 
# same performance alternative:
# map(i -> knetarray[i] = v[index+i] ,[1:length(var_layer);])
# product_dims = length(var_layer)
# var_layer .= reshape(v[index:index+product_dims-1], size(var_layer))	
# for i in eachindex(var_layer)
	# knetarray[i] = v[index+i]
# end	
# product_dims = mapreduce(identity,*,size(var_layer)) # compute the consumed index by the layer

"""
		build_nested_array_from_vec(chain, v)
Builds a vector of KnetArrays of the same shape as chain of value v.
It calls iteratively build_array to build each intermediary KnetArrays.
This methods is not optimised, it consumes memory.
"""
function build_nested_array_from_vec(chain, v::Vector{T}) where T <: Number
	param = params(chain)
	flatten_params = mapreduce_cat1d_vcat(param)
	size(flatten_params) == size(v) || error("Le vecteur n'est pas à la bonne dimension, function rebuild_nested_array $(size(flatten_params)) != $(size(v))")

	param_value = (x -> x.value).(param) #param de type Param, (x -> x.value).(param) donne Vector{KnetArrays}
	vec_KnetArray = build_nested_array_from_vec(param_value,v) 
	return vec_KnetArray
end

function build_nested_array_from_vec(nested_array::Vector{KnetArray{T}}, v::Vector{T}) where T <: Number
	vec_KnetArray = Vector{KnetArray{T}}(map(i-> similar(nested_array[i]), [1:length(nested_array);]))
	build_nested_array_from_vec!(nested_array, v, vec_KnetArray)
	return vec_KnetArray
end

function build_nested_array_from_vec!(nested_array::Vector{KnetArray{T}}, v::Vector{T}, vec_KnetArray::Vector{KnetArray{T}} ) where T <: Number
	index = 0
	for (i,variable_layer) in enumerate(nested_array)		
		consumed_indices = build_array!(v, variable_layer, index, vec_KnetArray[i])		
		index += consumed_indices
	end	
end
