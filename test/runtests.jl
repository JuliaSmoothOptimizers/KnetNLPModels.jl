using Test
using CUDA, Knet, MLDatasets, NLPModels
using KnetNLPModels

@testset "KnetNLPModels tests" begin
  struct Conv
    w
    b
    f
  end
  (c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
  Conv(w1, w2, cx, cy, f = relu) = Conv(param(w1, w2, cx, cy), param0(1, 1, cy, 1), f)

  struct Dense
    w
    b
    f
    p
  end
  (d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b)
  Dense(i::Int, o::Int, f = sigm; pdrop = 0.0) = Dense(param(o, i), param0(o), f, pdrop)

  struct Chainnll <: KnetNLPModels.Chain
    layers
    Chainnll(layers...) = new(layers)
  end
  (c::Chainnll)(x) = (for l in c.layers
    x = l(x)
  end;
  x)
  (c::Chainnll)(x, y) = Knet.nll(c(x), y)  # nécessaire
  (c::Chainnll)(data::Tuple{T1, T2}) where {T1, T2} = c(first(data, 2)...)
  (c::Chainnll)(d::Knet.Data) = Knet.nll(c; data = d, average = true)

  ENV["DATADEPS_ALWAYS_ACCEPT"] = true # download datasets without having to manually confirm the download
  CUDA.allowscalar(true)

  xtrn, ytrn = MNIST.traindata(Float32)
  ytrn[ytrn .== 0] .= 10
  xtst, ytst = MNIST.testdata(Float32)
  ytst[ytst .== 0] .= 10
  dtrn = minibatch(xtrn, ytrn, 100; xsize = (size(xtrn, 1), size(xtrn, 2), 1, :))
  dtst = minibatch(xtst, ytst, 100; xsize = (size(xtst, 1), size(xtst, 2), 1, :))

  LeNet = Chainnll(Conv(5, 5, 1, 20), Conv(5, 5, 20, 50), Dense(800, 500), Dense(500, 10, identity))
  LeNetNLPModel = KnetNLPModel(LeNet; data_train = (xtrn, ytrn), data_test = (xtst, ytst))

  x1 = copy(LeNetNLPModel.w)
  x2 = (x -> x + 50).(Array(LeNetNLPModel.w))

  obj_x1 = obj(LeNetNLPModel, x1)
  grad_x1 = NLPModels.grad(LeNetNLPModel, x1)
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

  minibatch_size = rand(200:300)
  set_size_minibatch!(LeNetNLPModel, minibatch_size)
  @test LeNetNLPModel.size_minibatch == minibatch_size
  @test length(LeNetNLPModel.training_minibatch_iterator) == floor(length(ytrn) / minibatch_size)
  @test length(LeNetNLPModel.test_minibatch_iterator) == floor(length(ytst) / minibatch_size)

  rand_minibatch_train!(LeNetNLPModel)
  @test LeNetNLPModel.i_train >= 1
  @test LeNetNLPModel.i_train <= LeNetNLPModel.training_minibatch_iterator.length - minibatch_size

  reset_minibatch_train!(LeNetNLPModel)
  @test LeNetNLPModel.i_train == 1
  minibatch_next_train!(LeNetNLPModel)
  i = 1
  i += LeNetNLPModel.size_minibatch
  (next, indice) = iterate(LeNetNLPModel.training_minibatch_iterator, i)
  @test LeNetNLPModel.i_train == i
  @test isequal(LeNetNLPModel.current_training_minibatch, next)
end
