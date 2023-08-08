
function one_adam_run!(chain, dtrn, dtst; iter_max=5, name="undefined", index_seed=0, accuracy::AbstractVector{Float64}=Vector{Float64}(undef, iter_max))
  acc = Knet.accuracy(chain; data=dtst)    
  println(name, "(seed $(index_seed)) initial accuracy: ", acc)
  for i in 1:iter_max # 1 Adam epoch + accuracy check
    progress!(adam(chain, ncycle(dtrn,1)))
    acc = Knet.accuracy(chain; data=dtst)    
    accuracy[i] = acc
    println(name, " accuracy: ", acc, " at the ", i, "-th iterate")
  end  
  return accuracy
end

function seeded_adam_trains(model_function, dtrn, dtst;
   iter_max=5,
   size_minibatch=100,
   name_architecture="undefined",
   name_dataset="undefined",
   max_seed=10)

  println("recap, iter_max: ", iter_max, " size_minibatch :", size_minibatch, " name_architecture: ", name_architecture, " name_dataset :", name_dataset)
  accuracy = Matrix{Float64}(undef, iter_max, max_seed)
  for seed in 1:max_seed
    view_acc = view(accuracy, 1:iter_max, seed)
    one_adam_run!(model_function(), dtrn, dtst; iter_max, name=name_architecture, index_seed=seed, accuracy=view_acc)
  end  
  println("accuracies % of $(name_architecture) on $(name_dataset) : \n", accuracy)

  (_mean, _std) = mean_and_std(accuracy, 2)
  mean = Vector(_mean[:,1])
  std = Vector(_std[:,1])

  println("mean % of $(name_architecture) on $(name_dataset) : \n", mean)
  println("std % of $(name_architecture) on $(name_dataset) : \n", std)

  io = open("AAAI24/results/minbatch$(size_minibatch)/$(name_dataset)/$(name_architecture).jl", "w")
  print(io, "accuracies_$(name_architecture)_$(name_dataset) = ", accuracy, "\n\n")
  print(io, "mean_$(name_architecture)_$(name_dataset) = ", mean, "\n\n")
  print(io, "std_$(name_architecture)_$(name_dataset) = ", std, "\n\n")

  close(io)
  return mean
end

function train_architecture_dataset_for_a_minibatchsize(xtrn, ytrn, xtst, ytst, iter_max, size_minibatch, max_seed, name_dataset = "MNIST")
  dtrn = create_minibatch(xtrn, ytrn, size_minibatch)	 	 
  dtst = create_minibatch(xtst, ytst, size_minibatch)

  acc_Adam_LeNet_NLL = seeded_adam_trains(LeNet_NLL, dtrn, dtst; iter_max, size_minibatch, name_architecture="LeNet_NLL", name_dataset, max_seed)
  acc_Adam_LeNet_PSLDP = seeded_adam_trains(LeNet_PSLDP, dtrn, dtst; iter_max, size_minibatch, name_architecture="LeNet_PSLDP", name_dataset, max_seed)
  acc_Adam_PSNet_NLL = seeded_adam_trains(PSNet_NLL, dtrn, dtst; iter_max, size_minibatch, name_architecture="PSNet_NLL", name_dataset, max_seed)
  acc_Adam_PSNet_PSLDP = seeded_adam_trains(PSNet_PSLDP, dtrn, dtst; iter_max, size_minibatch, name_architecture="PSNet_PSLDP", name_dataset, max_seed)

  mean_accuracies = reshape(vcat(acc_Adam_LeNet_NLL, acc_Adam_LeNet_PSLDP, acc_Adam_PSNet_NLL, acc_Adam_PSNet_PSLDP), iter_max, 4)

  io = open("AAAI24/results/minbatch$(size_minibatch)/$(name_dataset)/mean_recap.jl", "w")
  print(io, "mean_accuracies_$(name_dataset) = ", mean_accuracies)
  close(io)
end
