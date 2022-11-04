# KnetNLPModels : An NLPModels Interface to Knet

| **Documentation** | **Linux/macOS/Windows** | **Coverage** | **DOI** |
|:-----------------:|:-------------------------------:|:------------:|:-------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] | [![codecov][codecov-img]][codecov-url] | [![doi][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaSmoothOptimizers.github.io/KnetNLPModels.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://JuliaSmoothOptimizers.github.io/KnetNLPModels.jl/dev
[build-gh-img]: https://github.com/JuliaSmoothOptimizers/KnetNLPModels.jl/workflows/CI/badge.svg?branch=main
[build-gh-url]: https://github.com/JuliaSmoothOptimizers/KnetNLPModels.jl/actions
[codecov-img]: https://codecov.io/gh/JuliaSmoothOptimizers/KnetNLPModels.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/gh/JuliaSmoothOptimizers/KnetNLPModels.jl
[doi-img]: https://zenodo.org/badge/447176402.svg
[doi-url]: https://zenodo.org/badge/latestdoi/447176402

## How to Cite

If you use KnetNLPModels.jl in your work, please cite using the format given in [`CITATION.bib`](CITATION.bib).

## Compatibility
Julia ≥ 1.6.

## How to install
This module can be installed with the following command:
```julia
pkg> add KnetNLPModels
pkg> test KnetNLPModels
```

## Synopsis
KnetNLPModels is an interface between [Knet.jl](https://github.com/denizyuret/Knet.jl.git)'s classification neural networks and [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl.git).

A `KnetNLPModel` gives the user access to:
- the values of the neural network variables/weights `w`;
- the value of the objective/loss function `L(X, Y; w)` at `w` for a given minibatch `(X,Y)`;
- the gradient `∇L(X, Y; w)` of the objective/loss function at `w` for a given minibatch `(X,Y)`.

In addition, it provides tools to:
- switch the minibatch used to evaluate the neural network;
- change the minibatch size;
- measure the neural network's accuracy at the current `w`.

## How to use
Check the [tutorial](https://juliasmoothoptimizers.github.io/KnetNLPModels.jl/stable/).

## How to Cite

If you use KnetNLPModels.jl in your work, please cite using the format given in [`CITATION.bib`](https://github.com/JuliaSmoothOptimizers/KnetNLPModels.jl/blob/main/CITATION.bib).

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/KnetNLPModels.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
