#  Chain_NLL, utilisée pour faire le lien entre les différents layers.
# Son evaluation contient également la fonction de perte negative log likehood
# Elle est également utilisé afin de précompilé la structure PS d'un réseau
struct Chain_NLL
	layers
	Chain_NLL(layers...) = new(layers)
end
(c::Chain_NLL)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain_NLL)(x,y) = nll(c(x),y) # nécessaire
(c::Chain_NLL)(data :: Tuple{T1,T2}) where {T1,T2} = nll(c(data[1]), data[2], average=true)
(c::Chain_NLL)(d::Knet.Data) = nll(c; data=d, average=true) 
# no_dropout(c::Chain_NLL)=map(l -> ones(Bool,input(l)), c.layers) 
# à utiliser une fois que vec_dropout a été correctement initialisé
# no_dropout!(c::Chain_NLL,vec_dropout::Vector{Vector{Bool}}) =	map!(l-> l .= ones(Bool, length(l)), vec_dropout, c.layers)


mutable struct Chain_PSLAP
	layers
	Chain_PSLAP(layers...) = new(layers)
end
(c::Chain_PSLAP)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain_PSLAP)(x,y) = PSLAP(c(x),y)
(c::Chain_PSLAP)(d::Knet.Data) = PSLAP(c; data=d, average=true)
function PSLAP(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	for (x,y) in data
		(z,n) = PSLAP(model(x; o...), y; dims=dims, average=false) 
		sum += z; cnt += n
	end
	average ? sum / cnt : (sum, cnt)
end
function PSLAP(scores,labels::AbstractArray{<:Integer}; dims=1, average=true)
	indices = findindices(scores,labels,dims=dims)
	scores = exp.(scores .- maximum(scores, dims=1)) # enlever cette ligne fonctionne moins bien fonction	
	acc = sum(.- log.(scores[indices]))
	ninstances = length(labels); y1 = size(scores,1) 
	tmp = [1:(ninstances*y1);]; splice!(tmp, indices)
	acc += sum(scores[tmp])
	average ? (acc / length(labels)) : (acc, length(labels))
end

#PSLEP
mutable struct Chain_PSLEP
	layers
	Chain_PSLEP(layers...) = new(layers)
end
(c::Chain_PSLEP)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain_PSLEP)(x,y) = PSLEP(c(x),y)
(c::Chain_PSLEP)(d::Knet.Data) = PSLEP(c; data=d, average=true)
function PSLEP(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	for (x,y) in data
		(z,n) = PSLEP(model(x; o...), y; dims=dims, average=false) 
		sum += z; cnt += n
	end
	average ? sum / cnt : (sum, cnt)
end
function PSLEP(scores,labels::AbstractArray{<:Integer}; dims=1, average=true)
	indices = findindices(scores,labels,dims=dims)
	scores = exp.(scores .- maximum(scores, dims=1)) # enlever cette ligne fonctionne moins bien fonction	
	acc = sum(.- log.(scores[indices]))
	average ? (acc / length(labels)) : (acc, length(labels))
end

#PSLDP
mutable struct Chain_PSLDP
	layers
	Chain_PSLDP(layers...) = new(layers)
end
(c::Chain_PSLDP)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain_PSLDP)(x,y) = PSLDP(c(x),y)
(c::Chain_PSLDP)(data :: Tuple{T1,T2}) where {T1,T2} = _PSLDP(c; data=data, average=true)
(c::Chain_PSLDP)(d::Knet.Data) = PSLDP(c; data=d, average=true)
function PSLDP(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	for (x,y) in data
		(z,n) = PSLDP(model(x; o...), y; dims=dims, average=false) 
		sum += z; cnt += n
	end
	average ? sum / cnt : (sum, cnt)
end
function _PSLDP(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	(x,y) = data
	(z,n) = PSLDP(model(x; o...), y; dims=dims, average=false) 
	sum += z; cnt += n
	average ? sum / cnt : (sum, cnt)
end
function PSLDP(scores,labels::AbstractArray{<:Integer}; dims=1, average=true)
	indices = findindices(scores,labels,dims=dims)
	# scores = exp.(scores .- reshape(scores[indices], 1, length(indices))) # diminue par les scores par celui que l'on cherche à obtenir  
  # scores = (x -> x^2).(exp.(scores .- reshape(scores[indices],1, length(indices)))) .- 1. # test
  scores = (x -> x^2).(exp.(scores .- reshape(scores[indices],1, length(indices))))
	# absence de garantie < 1
	acc = sum(scores)
	average ? (acc / length(labels)) : (acc, length(labels))
end

#PSLDP2
mutable struct Chain_PSLDP2
	layers
	Chain_PSLDP2(layers...) = new(layers)
end
(c::Chain_PSLDP2)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain_PSLDP2)(x,y) = PSLDP2(c(x),y)
(c::Chain_PSLDP2)(data :: Tuple{T1,T2}) where {T1,T2} = _PSLDP2(c; data=data, average=true)
(c::Chain_PSLDP2)(d::Knet.Data) = PSLDP2(c; data=d, average=true)
function PSLDP2(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	for (x,y) in data
		(z,n) = PSLDP2(model(x; o...), y; dims=dims, average=false) 
		sum += z; cnt += n
	end
	average ? sum / cnt : (sum, cnt)
end
function _PSLDP2(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	(x,y) = data
	(z,n) = PSLDP2(model(x; o...), y; dims=dims, average=false) 
	sum += z; cnt += n
	average ? sum / cnt : (sum, cnt)
end
function PSLDP2(scores,labels::AbstractArray{<:Integer}; dims=1, average=true)
	indices = findindices(scores,labels,dims=dims)	
  # losses = (x -> x^2).(exp.(scores .- reshape(scores[indices], 1, length(indices))))
    
  # overwrite array, unsupported on GPU
  # labeled_score = reshape(scores[indices],1, length(indices))  
  # tmp = ((_score, _labeled_score) -> exp(_score - _labeled_score)^2).(scores, labeled_score)  
  # tmp[indices] .= (l -> - exp(l)^2).(scores[indices])
  # losses = sum(tmp)

  # incompatible with GPU rerversediff
  # labeled_scores = reshape(scores[indices],1, length(indices))
  # losses = sum( ((_score, _labeled_score) -> _score == _labeled_score ? - exp(_labeled_score)^2 : exp(_score - _labeled_score)^2).(scores, labeled_scores))

  size_NN_output = size(scores, 1) # 10 for MNIST-CIFAR10, 100 for CIFAR100
  indice_max = reduce(*, size(scores))
  losses = sum(index -> (exp(scores[index] - scalar_factor(index, size_NN_output, indices) * scores[indices[index_indices(index, size_NN_output)]]))^2, 1:indice_max)    

	# absence de garantie < 1
	average ? (losses / length(labels)) : (losses, length(labels))
end

#PSLDP3
mutable struct Chain_PSLDP3
	layers
	Chain_PSLDP3(layers...) = new(layers)
end
(c::Chain_PSLDP3)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain_PSLDP3)(x,y) = PSLDP3(c(x),y)
(c::Chain_PSLDP3)(data :: Tuple{T1,T2}) where {T1,T2} = _PSLDP3(c; data=data, average=true)
(c::Chain_PSLDP3)(d::Knet.Data) = PSLDP3(c; data=d, average=true)
function PSLDP3(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	for (x,y) in data
		(z,n) = PSLDP3(model(x; o...), y; dims=dims, average=false) 
		sum += z; cnt += n
	end
	average ? sum / cnt : (sum, cnt)
end
function _PSLDP3(model; data, dims=1, average=true, o...)	
	sum = cnt = 0
	(x,y) = data
	(z,n) = PSLDP3(model(x; o...), y; dims=dims, average=false) 
	sum += z; cnt += n
	average ? sum / cnt : (sum, cnt)
end
function PSLDP3(scores,labels::AbstractArray{<:Integer}; dims=1, average=true)
	indices = findindices(scores,labels,dims=dims)
	scores = ((x -> .- log(x.^2) .+ exp(x).^2).(scores .- reshape(scores[indices],1, length(indices))))
	# absence de garantie < 1
	acc = sum(scores)
	average ? (acc / length(labels)) : (acc, length(labels))
end

index_indices(index, size_NN_output) = Int(((index - 1 - (index-1) % size_NN_output)) / size_NN_output +1 )
scalar_factor(index, size_NN_output, indices) = indices[index_indices(index, size_NN_output)] == index ? 2 : 1


function findindices(scores, labels::AbstractArray{<:Integer}; dims=1)
	ninstances = length(labels)
	nindices = 0
	indices = Vector{Int}(undef,ninstances)
	if dims == 1                   # instances in first dimension
			y1 = size(scores,1)
			y2 = div(length(scores),y1)
			if ninstances != y2; throw(DimensionMismatch()); end
			@inbounds for j=1:ninstances
					if labels[j] == 0; continue; end
					indices[nindices+=1] = (j-1)*y1 + labels[j]
			end
	elseif dims == 2               # instances in last dimension
			y2 = size(scores,ndims(scores))
			y1 = div(length(scores),y2)
			if ninstances != y1; throw(DimensionMismatch()); end
			@inbounds for j=1:ninstances
					if labels[j] == 0; continue; end
					indices[nindices+=1] = (labels[j]-1)*y1 + j
			end
	else
			error("findindices only supports dims = 1 or 2")
	end
	return (nindices == ninstances ? indices : view(indices,1:nindices))
end


#= Enf of layers and chains definition
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
=#
