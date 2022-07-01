# KnetNLPModels.jl

## How to install
This module can be installed with the following command:
```julia
julia> ] add KnetNLPModels
pkg> test KnetNLPModels
```

## Synopsis

A `KnetNLPModel` gives the user access to:
- the values of the neural network variables/weights `w`;
- the value of the objective/loss function `L(X, Y; w)` at `w` for a given minibatch `(X,Y)`;
- the gradient `∇L(X, Y; w)` of the objective/loss function at `w` for a given mini-batch `(X,Y)`.

In addition, it provides tools to:
- switch the minibatch used to evaluate the neural network;
- change the minibatch size;
- measure the neural network's accuracy at the current `w`.
