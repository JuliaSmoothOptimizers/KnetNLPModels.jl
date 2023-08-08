#=
  Temporary layer structures matching the neural network architecture.
  They are used to compute the partitioned structure of a partially-separable training (PSLDP)
=#

  # Structure infering the partitioned structure of a convolutional layer
  struct PsConv; w; b; dp; lengthlayer::Int end 
  PsConv(w1, w2, cx, cy; index=0) = PsConv([index + (j-1)*w2 + (p-1)*w1*w2 + (s-1)*w1*w2*cx + i for i=1:w1,j=1:w2,p=1:cx,s=1:cy], [index+w1*w2*cx*cy+ i for i=1:cy], ones(Bool, w1, w2, cx, cy), w1*w2*cx*cy+cy)
  size(psc::PsConv) = size(psc.w)
  input(psc::PsConv) = maximum(size(psc)) # non claire dans le cas de Conv, en pratique doit être >0
  ps_struct(c::Conv; index::Int=0) = PsConv(size(c.w)...; index=index)

  # Peut importe les dépendances initiales, ie: qu'elles soient larges ou petites elles seront toutes transmises à la couce suivante.
  # Seules les variabes propres au filtrage sont conservées et distinctes.
  function ps(psc::PsConv, dep::Vector{Vector{Int}}; dp=ones(Bool,pss.in))
    common_dep = reduce(((x,y) -> unique!(vcat(x,y))), dep; init=[])
    (w1, w3, cx, cy) = size(psc)
    new_dep = Vector{Vector{Int}}(undef, cy)
    for i in 1:cy
      tmp_dep = vec(psc.w[:,:,:,i])
      total_tmp_dep = vcat(common_dep, tmp_dep, psc.b[i])
      new_dep[i] = total_tmp_dep
    end 
    return new_dep
  end 

  # Structure infering the partitioned structure of a dense layer
  struct PsDense; var_layer::Array{Int,2}; bias_layer::Vector{Int}; dp::Vector{Bool}; lengthlayer::Int end #total length #var_layer+#bias_layer
  PsDense(in::Int, out::Int; index::Int=0) = PsDense( [(index + (j-1)*out + i) for i=1:out, j=1:in], [(index + in*out + i) for i=1:out], trues(in), (in+1)*out) # in+1 : +1 pour considérer le biais
  size(d::PsDense) = size(d.var_layer)
  input(d::PsDense) = size(d)[2]
  ps_struct(d::Dense; index::Int=0) = begin (o,i) = size(d.w); PsDense(i, o; index) end

  """
      ps(psd::PsDense, Dep::Vector{Vector{int}})
      
  pour chaque noeud on va ajouter les dépendances du layer propre (excepté dropout) 
  on y ajoute ensuite les dépendances (car dense) de la couche précédente modélisé par Di (à modérer avec le dropout)
  Le cas de la couche Dense est particulier, sans dropout il n'a que peu d'intérêt
  """
  function ps(psd::PsDense, dep::Vector{Vector{Int}}; dp=ones(Bool,pss.in))  
    (out,in) = size(psd)
    length(dp)==in && psd.dp .= dp
    if length(dep)==in
      transformed_dep =dep
    elseif in%length(dep)==0 #pour faire le lien avec les couches de convolution
      gcd = in/length(dep)
      transformed_dep = [dep[i] for j=1:gcd,i=1:length(dep)]
    else
      @error("nombre de dépendances et nombre de neurones distincts PsDense")	
    end 
    new_dep = Vector{Vector{Int}}(undef, out)
    for i in 1:out
      pertinent_indices = findall(psd.dp) #obtention des indices du dropout
      interlayer_dep = vcat(psd.var_layer[i,pertinent_indices], psd.bias_layer[i]) # dépendances du layer propre
      # accumulation_dep = dep[pertinent_indices] # les dépendances de la couche précédente propagée		
      accumulation_dep = vcat(transformed_dep[pertinent_indices]...) # les dépendances de la couche précédente propagée
      new_dep[i] = sort!(unique!(vcat(interlayer_dep, accumulation_dep))) # concaténation des dépendances
    end 
    return new_dep	
  end 

  # Structure infering the partitioned structure of a separable layer
  struct PsSep; vec_wi::Vector{Array{Int,2}}; vec_bi::Vector{Vector{Int}}; dp::Vector{Bool}; N; gcdi; gcdo; lengthlayer::Int; in; out end #total length #var_layer+#bias_layer
  size(pss::PsSep) = (pss.out, pss.in, pss.N)
  input(pss::PsSep) = pss.in
  function PsSep(in::Int, out::Int; index::Int=0, N::Int=gcd(in, out))
    (gcdi,gcdo) = checkfloor(N,in), checkfloor(N, out)
    vec_wi = map( (j->zeros(Int32, gcdo, gcdi)) ,[1:N;])
    vec_bi = map( (x->zeros(Int32, gcdo)),[1:N;])
    for l in 1:N 		
      vec_wi[l] = [(index + (l-1)*gcdi*gcdo + (j-1)*gcdo + i) for i=1:gcdo, j=1:gcdi]
      vec_bi[l] = [(index + N*gcdi*gcdo + (l-1)*gcdo + i) for i=1:gcdo]
    end 	
    dp = trues(in)
    lengthlayer = N*gcdi*gcdo + N*gcdo
    return PsSep(vec_wi, vec_bi, dp, N, gcdi, gcdo, lengthlayer, in, out)	
  end 
  ps_struct(psd::Sep_layer; index::Int=0) = PsSep(psd.in,psd.out; N=psd.N, index=index)

  """
      ps(pss::PsSep, Dep::Vector{Vector{int}})
      
  pour chaque noeud on va ajouter les dépendances du layer propre: une sous partie de la couche précédente (+ dropout) 
  on y ajoute ensuite certaines dépendances de la couche précédente modélisé par Di (à modérer avec le dropout)
  Dans le cas PS, la propagation des variables sera moins intense que le cas Dense
  """
  function ps(pss::PsSep, dep::Vector{Vector{Int}}; dp=ones(Bool,pss.in))
    (gcdo, gcdi, N, in) = (pss.gcdo, pss.gcdi, pss.N, pss.in)
    length(dp)==in && pss.dp .= dp
    l_dep = length(dep)
    if length(dep)==in
      transformed_dep = dep
    elseif in%length(dep)==0 # make the link with convolutional layer    
      gcd = in/length(dep)
      transformed_dep = [dep[i] for j=1:gcd, i=1:length(dep)]
    else
      error("nombre de dépendances et nombre de neurones distincts PsSep, (in,dep)=$in,$l_dep")	
    end 
    new_dep = Vector{Vector{Int}}(undef, pss.out)	
    for i in 1:N
      # get separable layer indices spared from the dropout    
      brut_pertinent_indices = findall(pss.dp[((i-1)*gcdi+1):(i*gcdi)]) 
      pertinent_indices = brut_pertinent_indices .+ (i-1)*gcdi
      # propagation of previous layer dependencies    
      accumulation_dep = vcat(transformed_dep[pertinent_indices]...) 
      for j in 1:gcdo      
        # separable layer dependencies taking into account the drop-out
        interlayer_dep = vcat(pss.vec_wi[i][j, brut_pertinent_indices], pss.vec_bi[i][j])
        # concatenate dependencies 
        new_dep[(i-1)*gcdo + j] = sort!(unique!(vcat(interlayer_dep, accumulation_dep))) 
      end
    end 
    return new_dep	
  end 

#=
  Methods using coordinating precedents structures and retrieving the partitioned structure.
=#

no_dropout!(c,vec_dropout::Vector{Vector{Bool}}) =	map!(l-> l .= ones(Bool, length(l)), vec_dropout, c.layers)
no_dropout(c) = map(l -> ones(Bool,input(l)), c.layers) 

"""
    precompile_ps_struct(network<:Chain)

The function is called on a network defined by Dense/Sep_layer/Conv layers or any layer defining a `ps_struct(layer)`method.
It uses `ps_struct` successively onto layers to capture the variables each neuron depends on?.
It returns a precompiled PS structure as nested Vector of Integer, each of which informs about the variables (bias included) by each element function.
The dropout, is not tested for now.
"""
function precompile_ps_struct(c)
	index = 0
	precompiled_ps_struct_layers = []
	for l in c.layers
		indexed_var_layer = ps_struct(l; index=index)
		index += indexed_var_layer.lengthlayer #on ajoute le nombre de variables de la couche pour obtenir une numérotation correcte des variables		
		push!(precompiled_ps_struct_layers, indexed_var_layer)
	end 
	precompiled_ps_struct = typeof(c)(precompiled_ps_struct_layers...)
	return precompiled_ps_struct
end 

"""
    PS_deduction(c,dp)
    
Compute the partially separable structure of a network represented by the Chain c by performing a forward evaluation.
"""
function PS_deduction(c; dp=no_dropout(c))
	inputs = input(c.layers[1])
	Dep = Vector{Vector{Int}}(map(i -> zeros(Int,0), 1:inputs)) # dépendance nulles de la taille des entrées
	length(dp)==length(c.layers) || error("size dropout does not match the network")
	for (index,l) in enumerate(c.layers)
		Dep = ps(l, Dep; dp=dp[index])		
	end
	Dep
end

PS(c::T) = PS_deduction(precompile_ps_struct(c))


build_listes_indices(chain) = build_listes_indices(PS(chain))
function build_listes_indices(ps_scores :: Vector{Vector{Int}}) 
	C = length(ps_scores)
	table_indices = reshape(map(i -> Vector{Int}(undef,0), 1:C^2), C, C)
	for i in 1:C
		for j in 1:C
			table_indices[i,j] = unique!(vcat(ps_scores[i], ps_scores[j]))
		end
	end
	return table_indices
end 