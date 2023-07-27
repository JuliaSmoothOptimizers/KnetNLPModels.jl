module KnetNLPModels

using Knet, NLPModels
export Chain
export AbstractKnetNLPModel, KnetNLPModel
export vector_params, accuracy
export reset_minibatch_train!, rand_minibatch_train!, minibatch_next_train!, set_size_minibatch!
export reset_minibatch_test!, rand_minibatch_test!, minibatch_next_test!
export build_nested_array_from_vec, build_nested_array_from_vec!
export create_minibatch, set_vars!, vcat_arrays_vector

# abstract type Chain end

abstract type AbstractKnetNLPModel{T, S} <: AbstractNLPModel{T, S} end

"""
    KnetNLPModel{T, S, C} <: AbstractNLPModel{T, S}

Data structure that makes the interfaces between neural networks defined with [Knet.jl](https://github.com/denizyuret/Knet.jl) and [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
A KnetNLPModel has fields

* `meta` and `counters` retain informations about the `KnetNLPModel`;
* `chain` is the chained structure representing the neural network;
* `data_train` is the complete data training set;
* `data_test` is the complete data test;
* `size_minibatch` parametrizes the size of an training and test minibatches, which are of size `1/size_minibatch * length(ytrn)` and `1/size_minibatch * length(ytst)`;
* `training_minibatch_iterator` is an iterator over an training minibatches;
* `test_minibatch_iterator` is an iterator over the test minibatches;
* `current_training_minibatch` is the training minibatch used to evaluate the neural network;
* `current_minibatch_test` is the current test minibatch, it is not used in practice;
* `w` is the vector of weights/variables;
* `layers_g` is a nested array used for internal purposes;
* `nested_array` is a vector of `Array{T,N}`; its shape matches that of `chain`.
"""
mutable struct KnetNLPModel{T, S, C, V, Z} <: AbstractKnetNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  chain::C
  counters::Counters
  data_train
  data_test
  size_minibatch::Int
  training_minibatch_iterator
  test_minibatch_iterator
  current_training_minibatch
  current_test_minibatch
  w::S
  layers_g::Z
  nested_array::V
  i_train::Int
  i_test::Int
end

"""
    KnetNLPModel(chain_ANN; size_minibatch=100, data_train=MLDatasets.MNIST.traindata(Float32), data_test=MLDatasets.MNIST.testdata(Float32))

Build a `KnetNLPModel` from the neural network represented by `chain_ANN`.
`chain_ANN` is built using [Knet.jl](https://github.com/denizyuret/Knet.jl), see the [tutorial](https://JuliaSmoothOptimizers.github.io/KnetNLPModels.jl/dev/tutorial/) for more details.
The other data required are: an iterator over the training dataset `data_train`, an iterator over the test dataset `data_test` and the size of the minibatch `size_minibatch`.
Suppose `(xtrn,ytrn) = knetnlp.data_train`, then the size of each training minibatch will be `1/size_minibatch * length(ytrn)`.
By default, the other data are respectively set to the training dataset and test dataset of `MLDatasets.MNIST`, with each minibatch a hundredth of the dataset.
 """
function KnetNLPModel(
  chain_ANN::T;
  size_minibatch::Int = 100,
  data_train = begin
    (xtrn, ytrn) = MNIST(split=:train)[:]
    ytrn[ytrn .== 0] .= 10
    (xtrn, ytrn)
  end,
  data_test = begin
    (xtst, ytst) = MNIST(split=:test)[:]
    ytst[ytst .== 0] .= 10
    (xtst, ytst)
  end,
) where {T}
  x0 = vector_params(chain_ANN)
  n = length(x0)
  meta = NLPModelMeta(n, x0 = x0)

  xtrn = data_train[1]
  ytrn = data_train[2]
  xtst = data_test[1]
  ytst = data_test[2]
  training_minibatch_iterator = create_minibatch(xtrn, ytrn, size_minibatch)
  test_minibatch_iterator = create_minibatch(xtst, ytst, size_minibatch)
  current_training_minibatch = rand(training_minibatch_iterator)
  current_test_minibatch = rand(test_minibatch_iterator)

  nested_array = build_nested_array_from_vec(chain_ANN, x0)
  layers_g = similar.(params(chain_ANN)) # create a Vector of layer variables

  return KnetNLPModel(
    meta,
    chain_ANN,
    Counters(),
    data_train,
    data_test,
    size_minibatch,
    training_minibatch_iterator,
    test_minibatch_iterator,
    current_training_minibatch,
    current_test_minibatch,
    x0,
    layers_g,
    nested_array,
    1, #initialize the batch current i to 1
    1,
  )
end

"""
    set_size_minibatch!(knetnlp::AbstractKnetNLPModel, size_minibatch::Int)

Change the size of both training and test minibatches of the `knetnlp`.
Suppose `(xtrn,ytrn) = knetnlp.data_train`, then the size of each training minibatch will be `1/size_minibatch * length(ytrn)`; the test minibatch follows the same logic.
After a call of `set_size_minibatch!`, you must call `reset_minibatch_train!(knetnlp)` to use a minibatch of the expected size.
"""
function set_size_minibatch!(knetnlp::AbstractKnetNLPModel, size_minibatch::Int)
  knetnlp.size_minibatch = size_minibatch
  knetnlp.training_minibatch_iterator =
    create_minibatch(knetnlp.data_train[1], knetnlp.data_train[2], knetnlp.size_minibatch)
  knetnlp.test_minibatch_iterator =
    create_minibatch(knetnlp.data_test[1], knetnlp.data_test[2], knetnlp.size_minibatch)
  return knetnlp
end

include("utils.jl")
include("KnetNLPModels_methods.jl")
end
