---
Title: 'FluxNLPModels and KnetNLPModels.jl: Wrappers and Connectors for Deep Learning Models in JSOSolvers.jl'

tags:
  - Julia
  - Machine learning
  - Smooth-optimization
authors:
  - name: Farhad Rahbarnia
    corresponding: true
    affiliation: 1
  - name: Paul Raynaud
    # orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: GERAD, Montréal, Canada
   index: 1
 - name: GSCOP, Grenoble, France
   index: 2
date: 19 May 2023
bibliography: paper.bib
---

# Summary

We present two modules KnetNLPModels.jl and   FluxNLPModels.jl, which make a bridge (interface) between modules modelling deep neural networks, Knet.jl @Knet2020 and Flux.jl @Flux.jl-2018, and the JuliaSmoothOptimizers ecosystem.
Concretely, a deep neural network is modelled as a standard smooth optimization problem, whose the minimization of the loss function by solvers from JSOSolvers @orban-siqueira-jsosolvers-2021 will train the deep neural network.

<!-- Knet.jl and Flux.jl offers tools to model a deep neural network while letting a big flexibility on how the architecture is made. -->
Among others things, they provide definition of neural layers: dense layers, convolutional layers or more complex layers, which can be difficult and error-prone to implement itself.
In addition, they propose several loss functions, like the negative log likelihood, or allow the user to define its own loss function.  
Knet.jl and Flux.jl support both CPU and GPU, providing weights respectfully as Vector or CUVector (only CUDA supported?) on which various floating-point systems are available.
Moreover, they both support the training and test datasets from MLDatasets, they facilitate the definition of their minibatches as an iterator of a dataset and they provide a way to evaluate the neural network accuracy on the test dataset.
Finally, they offer state-of-the-art optimizers: stochastic gradient descent @lecun-bouttou-bengio-haffner1998, Nesterov acceleration @Nesterov1983, Adagrad @duchi-hazan-singer2011, and Adam @kingma-ba2017 which all rely solely on first derivatives of the sampled loss.

However, Knet.jl and Flux.jl, given their standalone nature and the traditional nature of stochastic algorithms, lack interfaces with general optimization frameworks such as JuliaSmoothOptimizers.
Once the neural network is settled, the minimization of the loss function can be seen as a smooth optimization problem.
Thus, it can be minimized by the tools furnished by the JuliaSmoothOptimizers ecosystem to train the neural network.
KnetNLPModels.jl and FluxNLPModels.jl model a neural network training problem as an optimization model fulfilling the NLPModels.jl @orban-siqueira-nlpmodels-2020 API which can be then minimized by solvers from JSOSolvers.jl.
KnetNLPModels.jl and FluxNLPModels.jl are designed to be user-friendly.
Setting the affiliated NLPModel does not require much work once the neural network architecture, the loss function and its dataset are instantiated (see the Example section).
Being fully integrated in the JuliaSmoothOptimizers ecosystem, the resulting model can be minimized directly, or enhanced to integrate a hessian approximation through a limited-memory quasi-Newton linear operator (LBFGS or LSR1).
With these modules, we hope that users and researchers will be able to employ a wide variety of optimization solvers and tools not traditionally applied to deep neural network training (not available either in Knet.jl or Flux.jl).

# Training a neural network with R2
This section details how to model a deep neural network from Knet.jl, define a NLPModel from it, and how it can be trained using the `R2` solver from JSOSOlvers.jl.
First, we model a simplified LeNet model @lecun-bouttou-bengio-haffner1998 paired with a negative log likelihood loss function using Knet.jl :
```julia
using Knet
# Define a convolutional layer
struct Conv
  w
  b
  f
end
(c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
Conv(w1, w2, cx, cy, f = relu) = Conv(param(w1, w2, cx, cy), param0(1, 1, cy, 1), f)

# Define a dense layer
struct Dense
  w
  b
  f
  p
end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b)
Dense(i::Int, o::Int, f = sigm; pdrop = 0.0) = Dense(param(o, i), param0(o), f, pdrop)

# A chain of layers ended by a negative log likelihood loss function
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

lenet = Chainnll(Conv(4,4,1,6), Conv(4,4,6,16), Dense(124, 84), Dense(84,10))
nlp_lenet = KnetNLPModel(lenet; data_train = (xtrn, ytrn), data_test = (xtst, ytst))
```
This LeNet architecture is designed to distinguish 10 classes.
Therefore, we chose to use the MNIST dataset @lecun-bouttou-bengio-haffner1998 from MLDataset @MLDataset2016:
```julia
using MLDatasets 

# download datasets without user intervention
ENV["DATADEPS_ALWAYS_ACCEPT"] = true  

xtrn, ytrn = MNIST.traindata(Float32) # MNIST training dataset
ytrn[ytrn.==0] .= 10 # re-arrange indices
xtst, ytst = MNIST.testdata(Float32) # MNIST test dataset
ytst[ytst.==0] .= 10 # re-arrange indices

# define the minibtaches for both training and test dataset.
data_train = minibatch(xtrn, ytrn, 100; xsize=(size(xtrn, 1), size(xtrn, 2), 1, :)) # training minibatch
data_test = minibatch(xtst, ytst, 100; xsize=(size(xtst, 1), size(xtst, 2), 1, :)) # test minibatch
```

From these elements, we transfer the Knet model to a KnetNLPModel:
```julia 
using KnetNLPModels

minibatchSize = 100
LeNetNLPModel = KnetNLPModel(
        LeNet;
        data_train,
        data_test,
        size_minibatch = minibatchSize,
    )
```
which will instantiate automatically the mandatory data-structures as `Vector` or `CuVector` if the code is launched either on a CPU or on a GPU.

Once these steps are completed, a solver from JSOSolver can be used, for example R2 (je n'ai pas trouvé de ref) :
```julia
using JSOSolvers

solver_stats = R2(
        LeNetNLPModel;
        callback = # PaRay to do, 
        # change the minibtach at iteration, and retrieve the accuracy if needed
    )
```
Being implemented as a determinist optimization method, the mini-batch data or sophisticate stopping criteria are managed through a callback method passed to the R2 solver.

The accuracy of a neural network can be checked with:
```julia
acc = NLPModels.accuracy(LeNetNLPModel)
```

An enhanced LBFGS model may define as
```julia
lbfgs_LeNet = NLPModelsModifiers.LBFGSModel(LeNetNLPModel)
```

Add numerical graph.

# Statement of need


# Acknowledgements

This work has been supported by the NSERC Alliance grant 544900-19 in collaboration with Huawei-Canada.


# References