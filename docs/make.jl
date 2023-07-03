using Documenter, KnetNLPModels

makedocs(
  modules = [KnetNLPModels],
  doctest = true,
  # linkcheck = true,
  strict = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "KnetNLPModels.jl",
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md", "LeNet training" => "LeNet_Training.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/KnetNLPModels.jl.git",
  push_preview = true,
  devbranch = "main",
)
