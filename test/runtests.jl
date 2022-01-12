using Test
using ChainedNLPModel

using Knet, MLDatasets, IterTools
using Statistics: mean
using NLPModels

@testset "first round" begin
	@test 2==2
end 

xtrn,ytrn = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10
xtst,ytst = MNIST.testdata(Float32);  ytst[ytst.==0] .= 10

@testset "first test" begin
	# définition des couches 
	struct Conv; w; b; f; end
	(c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
	Conv(w1,w2,cx,cy,f=relu) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f)
	
	struct Dense; w; b; f; p; end
	(d::Dense)(x) = d.f.(d.w * mat(dropout(x,d.p)) .+ d.b)
	Dense(i::Int,o::Int,f=sigm;pdrop=0.) = Dense(param(o,i), param0(o), f, pdrop)
	
	
	# La structure Chainnll, utilisée pour faire le lien entre les différents layers.
	# Son evaluation contient également la fonction de perte negative log likehood
	# Elle est également utilisé afin de précompilé la structure PS d'un réseau
	struct Chainnll <: ChainedNLPModel.Chain	
	# struct Chainnll <: Chain	
		layers
		Chainnll(layers...) = new(layers)
	end
	(c::Chainnll)(x) = (for l in c.layers; x = l(x); end; x)
	(c::Chainnll)(x,y) = Knet.nll(c(x),y)  # nécessaire
	(c::Chainnll)(d::Knet.Data) = Knet.nll(c; data=d, average=true)
	
	# La base de donnée, base de test
	dtrn = minibatch($xtrn, ytrn, 100; xsize=(size(xtrn,1),size(xtrn,2),1,:))
	dtst = minibatch(xtst, ytst, 100; xsize=(size(xtst,1),size(xtst,2),1,:))
	
	LeNet = Chainnll(Conv(5,5,1,20), Conv(5,5,20,50), Dense(800,500), Dense(500,10,identity))
	LeNetNLPModel = ChainNLPModel(LeNet; data_train=(xtrn,ytrn), data_test=(xtst,ytst))
	
	x1 = copy(LeNetNLPModel.w)
	x2 = (x -> x+50).(Array(LeNetNLPModel.w))
	
	obj_x1 = obj(LeNetNLPModel,x1)
	grad_x1 = NLPModels.grad(LeNetNLPModel,x1)
	@test x1 == LeNetNLPModel.w
	@test params(LeNetNLPModel.chain)[1].value[1] == x1[1]
	@test params(LeNetNLPModel.chain)[1].value[2] == x1[2]

	obj_x2 = obj(LeNetNLPModel, x2)
	grad_x2 = NLPModels.grad(LeNetNLPModel, x2)
	@test x2 == LeNetNLPModel.w
	@test params(LeNetNLPModel.chain)[1].value[1] == x2[1]
	@test params(LeNetNLPModel.chain)[1].value[2] == x2[2]


	@test obj_x1 != obj_x2
	@test grad_x1 != grad_x2

end 

using BenchmarkTools
@benchmark obj_x2 = obj(LeNetNLPModel, x2) # Memory estimate: 193.96 MiB, allocs estimate: 423864.
@benchmark build_nested_array_from_vec(LeNetNLPModel, LeNetNLPModel.w) # Memory estimate: 85.51 MiB, allocs estimate: 3448386.
@benchmark build_nested_array_from_vec!(LeNetNLPModel, LeNetNLPModel.w) # Memory estimate: 85.51 MiB, allocs estimate: 3448386.


