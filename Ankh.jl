module Ankh

using Pkg
using Shuffle
using Distributions
using Statistics
using LinearAlgebra

function initialise_parameters(layer_dims::Vector{Int64}, name::String)
    
    @assert name ∈ ("he_uniform", "xavier_normal", "glorot_uniform", "random")
        
    params = Dict()

    if name == "he_uniform"
        for i=2:length(layer_dims)
            iw = sqrt(6/(layer_dims[i-1]))
            params[string("W", (i-1))] = rand(Uniform(-iw,iw),layer_dims[i],layer_dims[i-1])
            params[string("b", (i-1))] = zeros(layer_dims[i], 1)
        end
    elseif name == "glorot_uniform"
        for i=2:length(layer_dims)
            iw = sqrt(6/(layer_dims[i-1] + layer_dims[i]))
            params[string("W", (i-1))] = rand(Uniform(-iw,iw),layer_dims[i],layer_dims[i-1])
            params[string("b", (i-1))] = zeros(layer_dims[i], 1)
        end
    elseif name == "xavier_normal"
        for i=2:length(layer_dims)
            iw = sqrt(2/(layer_dims[i-1] + layer_dims[i]))
            params[string("W", (i-1))] = rand(Normal(0,iw),(layer_dims[i],layer_dims[i-1]))
            params[string("b", (i-1))] = zeros(layer_dims[i], 1)
        end
    elseif name == "random"
        for i=2:length(layer_dims)
        params[string("W", (i-1))] = randn(layer_dims[i], layer_dims[i-1])
        params[string("b", (i-1))] = zeros(layer_dims[i], 1)
        end
    end
    return params
end

function activation_function(Z::Matrix{Float64}, name::String)
    
    @assert name ∈ ("Sigmoid", "ReLu","tanh","LReLu","Linear", "Swish")
    
    if name == "Sigmoid"
        A = 1 ./ (1 .+ exp.(.-Z))
        return (A = A, Z = Z)  
    elseif name == "ReLu"
        A = max.(0, Z)
        return (A = A, Z = Z)
    elseif name == "LReLu"
        A = max.(0.1*Z, Z)
        return (A = A, Z = Z)
    elseif name == "tanh"
        A = tanh.(Z)
        return (A = A, Z = Z)
    elseif name == "Linear"
        A = Z
        return (A = A, Z = Z)                
    elseif name == "Swish"
        A = Z * (1 ./ (1 .+ exp.(.-Z)))
        return (A = A, Z = Z)
    end
end

function forward_pass(A::Any, W::Matrix{Float64}, b::Matrix{Float64})
    Z = (A * W') .+ b'
    cache = (A, W, b)
    @assert size(Z) == (size(A, 1), size(W, 1))
    return (Z = Z, cache = cache)
end

function forward_propagation(input::Any, parameters::Any,activation_type::Vector{String}; n_reg::Int64 = 1)
    layer_size = Int(length(parameters) / 2)
    @assert layer_size ∈ (length(activation_type))
    master_cache = []
    A = input
    sum_weights = 0;
    for i = 1 : (layer_size)
        A_prev = A
        A_trans,cache = forward_pass(A_prev, parameters[string("W", (i))], parameters[string("b", (i))])
        A,A_prev = activation_function(A_trans, activation_type[i])
        push!(master_cache , (linear_step_cache = cache, activation_step_cache = A_prev))
        sum_weights = sum_weights + sum((parameters[string("W", (i))].^2).^(1/(n_reg*2%3)))
    end
    Ŷ = A
    return Ŷ, master_cache, sum_weights
end

function calculate_cost(Ŷ::Any, Y::Any,cost_fcn::String, λ, ∑w)
    
    if cost_fcn == "Mean_Squared_Error" || cost_fcn == "mse"
        mse = mean((Y - Ŷ).^2) + 0.5*λ*∑w;
        return mse
    end
    
    if cost_fcn == "binary_cross_entropy"
        epsilon = 1e-8
        bce = -(sum(Y .* log.(Ŷ .+ epsilon) .+ (1 .- Y) .* log.(1 .- Ŷ .+ epsilon))) / size(Y)[2] + 0.5*λ*∑w;
        return bce
    end
end

function derivative_activation(Z::Matrix{Float64}, name::String)
    @assert name ∈ ("Sigmoid", "ReLu","tanh","LReLu","Linear","Swish")
    if name == "Sigmoid"
        temp = 1 ./ (1 .+ exp.(.-Z))
        A = temp .* (1 .- temp)
        return A  
    elseif name == "ReLu"
        Z[Z .<= 0] .= 0
        Z[Z .> 0] .= 1
        A = Z
        return A
    elseif name == "LReLu"
        Z[Z .<= 0.1] .= 0.1
        Z[Z .> 0] .= 1
        A = Z
        return A
    elseif name == "tanh"
        A = 1 .- (tanh.(Z)).^2
        return A
    elseif name == "Linear"
        A = ones(size(Z))
        return A              
    elseif name == "Swish"
        temp = 1 ./ (1 .+ exp.(.-Z))
        A = temp + Z * (temp .* (1 .- temp))
        return A
    end    
end

function output_loss_derivative(Ŷ::Any, Y::Any, cost_fcn::String)
    if cost_fcn == "Mean_Squared_Error" || cost_fcn == "mse"
        m = size(Y, 1)
        return (-2 .* (Y .- Ŷ) ./ m)
    end
    
    if cost_fcn == "binary_cross_entropy"
      epsilon = 1e-8
      return (Ŷ .- Y) ./ (Ŷ .*(1 .- Ŷ) .+ epsilon)
    end
end                                  

function backward_pass(∂E, A, W, λ::Float64, n_reg::Int64)
    m = size(A, 1)
    
    ∂W = (A'* ∂E) ./ m + (0.5)^(n_reg%2)*λ*(W'./((W'.^(n_reg*2%4)).^(1/2)))
    ∂b = (sum(∂E,dims=1)) ./ m
    ∂A = ∂E * W
    
    return (∂W,∂b,∂A)
end

function back_propagation(Ŷ::Any, Y::Any, master_cache::Any, activation_type::Vector{String}, 
        loss_fcn::String, λ::Float64, n_reg::Int64)
    
    ∇ = Dict()
    linear_cache, activation_cache = master_cache[end]
    ∂E = derivative_activation(activation_cache, activation_type[end]) .* output_loss_derivative(Ŷ, Y, loss_fcn)
    for i = reverse(1:length(activation_type))
        linear_cache,_ = master_cache[i]

        ∇[string("∂W",(i))], ∇[string("∂b",(i))], ∇[string("∂A",(i))] = 
                                backward_pass(∂E, linear_cache[1],linear_cache[2], λ, n_reg)

        if i > 1
            _, activation_cache = master_cache[i-1]
            ∂E = ∇[string("∂A",(i))] .* derivative_activation(activation_cache, activation_type[i-1])
            end 
    end
    return ∇
end        

function update_weights(params,∇::Any, learning_rate, network_size ,optimiser)
    
    if optimiser == "gradient_descent" || optimiser == "stochastic_gradient_descent"
        for i = 1:network_size-1
            params[string("W", (i))] -= learning_rate .* ∇[string("∂W",(i))]'
            params[string("b", (i))] -= learning_rate .* ∇[string("∂b",(i))]'
        end
        return params
        
    elseif optimiser == "gradient_descent_with_momentum"
        for i = 1:network_size-1
            params[string("W", (i))] -= learning_rate .* ∇[string("v_dw",(i))]
            params[string("b", (i))] -= learning_rate .* ∇[string("v_db",(i))]
        end
        return params
        
    elseif optimiser == "rmsprop"
        κ = 1e-7
        param_rms, param_bp = ∇
        for i = 1:network_size-1
            params[string("W", (i))] -= learning_rate .* param_bp[string("∂W",(i))]' ./ sqrt.(param_rms[string("s_dw",i)] .+ κ)
            params[string("b", (i))] -= learning_rate .* param_bp[string("∂b",(i))]' ./ sqrt.(param_rms[string("s_db",i)] .+ κ)
        end
        return params
        
    elseif optimiser == "Adam"
        κ = 1e-7
        for i = 1:network_size-1
            params[string("W", (i))] -= learning_rate .* ∇[string("v_dw_corrected",i)] ./ (sqrt.(∇[string("s_dw_corrected",i)]) .+ κ)
            params[string("b", (i))] -= learning_rate .* ∇[string("v_db_corrected",i)] ./ (sqrt.(∇[string("s_db_corrected",i)]) .+ κ)
        end
        return params
    end
    
end

function train_model(params, x_train::Any, y_train::Any, layer_dims::Vector{Int64}, activation_type::Vector{String}, 
        loss_fcn::String; lr::Float64=0.001, epochs::Int64= 1, verbose::Bool=true, optimiser::String = "gradient_descent",
        shuffle_data::Bool = true, lambda::Float64 = 0, n_reg::Int64 = 1, steps_per_epoch = nothing, batch_size = nothing, 
        log::Bool = false, log_file::String = "Loss_Log")
    
    @assert optimiser ∈ ("gradient_descent", "stochastic_gradient_descent", "gradient_descent_with_momentum", "rmsprop", "Adam")
    
    if batch_size == nothing && steps_per_epoch == nothing
      batch_size = Int(ceil(size(x_train)[1] / 2))
      steps_per_epoch = Int(ceil(size(x_train)[1]/batch_size))
      
    elseif batch_size == nothing && steps_per_epoch != nothing
      batch_size = Int(ceil(size(x_train)[1]/steps_per_epoch))

    elseif batch_size != nothing && steps_per_epoch == nothing
      steps_per_epoch = Int(ceil(size(x_train)[1]/batch_size))
    end

    if optimiser == "gradient_descent"
        
        shuffled_index = [i for i = 1:size(x_train)[1]]
            
        for i = 1:epochs
            loss = 0
            if shuffle_data
                shuffle!(shuffled_index)
            end              
            Ŷ, cache, weights_sum = forward_propagation(x_train[shuffled_index[:],:], params, activation_type; n_reg)
            loss = calculate_cost(Ŷ, y_train[shuffled_index[:],:], loss_fcn, lambda, weights_sum)
            ∇ = back_propagation(Ŷ, y_train[shuffled_index[:],:], cache, activation_type, loss_fcn, lambda, n_reg)
            params = update_weights(params,∇,lr,length(layer_dims),optimiser)
            if verbose
                println("Iteration -> $i, Cost -> $loss")
            end
            
            if log
                open(string(log_file,".txt"), "a") do file
                    write(file, string(loss))
                    write(file, "\n")
                end
            end 
        end
        if log
            open(string(log_file,".txt"), "a") do file
                write(file, "*******")
                write(file, "\n")
            end
        end
    end
    
    if optimiser == "stochastic_gradient_descent"  
        
        shuffled_index = [i for i = 1:size(x_train)[1]]
        
        for i = 1:epochs
            loss = 0
            shuffle!(shuffled_index)
            for j = 1:length(shuffled_index)
                Ŷ, cache, weights_sum = forward_propagation(x_train[shuffled_index[j],:], params, activation_type; n_reg)
                loss = calculate_cost(Ŷ, y_train[shuffled_index[j],:], loss_fcn, lambda, weights_sum)
                ∇ = back_propagation(Ŷ, y_train[shuffled_index[j],:], cache, activation_type, loss_fcn, lambda, n_reg)
                params = update_weights(params,∇,lr,length(layer_dims),optimiser)
            end
            if verbose
                println("Iteration -> $i, Cost -> $loss")
            end
            
            if log
                open(string(log_file,".txt"), "a") do file
                    write(file, string(loss))
                    write(file, "\n")
                end
            end 
        end
        if log
            open(string(log_file,".txt"), "a") do file
                write(file, "*******")
                write(file, "\n")
            end
        end
    end
    
    if optimiser == "gradient_descent_with_momentum"  
        β = 0.9
        momentum = Dict()
        shuffled_index = [i for i = 1:size(x_train)[1]]
        
        for i = 1:length(layer_dims)-1
            momentum[string("v_dw",i)] = zeros(size(params[string("W",i)]))
            momentum[string("v_db",i)] = zeros(size(params[string("b",i)]))
        end
        
        for i = 1:epochs
            marker = 1
            loss = 0
            batch = deepcopy(batch_size)
            if shuffle_data
                shuffle!(shuffled_index)
            end 
            
            for k = 1:steps_per_epoch

                Ŷ, cache, weights_sum = forward_propagation(x_train[shuffled_index[marker:batch],:], params, 
                    activation_type; n_reg)
                loss = calculate_cost(Ŷ, y_train[shuffled_index[marker:batch],:], loss_fcn, lambda, weights_sum)            
                ∇ = back_propagation(Ŷ, y_train[shuffled_index[marker:batch],:], cache, activation_type, 
                    loss_fcn, lambda, n_reg)

                for j = 1:length(layer_dims)-1
                    momentum[string("v_dw",j)] = β .* momentum[string("v_dw",j)] .+ ((1 .- β) .* ∇[string("∂W",j)])'
                    momentum[string("v_db",j)] = β .* momentum[string("v_db",j)] .+ ((1 .- β) .* ∇[string("∂b",j)])' 
                end

                params = update_weights(params,momentum,lr,length(layer_dims),optimiser)
                if size(x_train)[1] - batch < batch_size
                  marker += batch_size
                  batch += size(x_train)[1] - batch
                  if marker >= size(x_train)[1]
                    break
                  end
                else
                    marker += batch_size
                    batch += batch_size
                end
                
            end
        
            if verbose
                println("Iteration -> $i, Cost -> ", loss/steps_per_epoch)
            end
            
            if log
                open(string(log_file,".txt"), "a") do file
                    write(file, string(loss/steps_per_epoch))
                    write(file, "\n")
                end
            end 
        end
        if log
            open(string(log_file,".txt"), "a") do file
                write(file, "*******")
                write(file, "\n")
            end
        end
    end
                
    if optimiser == "rmsprop"  
        β = 0.9999
        rms = Dict()
        shuffled_index = [i for i = 1:size(x_train)[1]]
        
        for i = 1:length(layer_dims)-1
            rms[string("s_dw",i)] = zeros(size(params[string("W",i)]))
            rms[string("s_db",i)] = zeros(size(params[string("b",i)]))
        end
        
        for i = 1:epochs
            marker = 1
            loss = 0
            batch = deepcopy(batch_size)
            if shuffle_data
                shuffle!(shuffled_index)
            end 
            
            for k = 1:steps_per_epoch
                
                Ŷ, cache, weights_sum = forward_propagation(x_train[shuffled_index[marker:batch],:], params, 
                    activation_type; n_reg)
                loss = calculate_cost(Ŷ, y_train[shuffled_index[marker:batch],:], loss_fcn, lambda, weights_sum)            
                ∇ = back_propagation(Ŷ, y_train[shuffled_index[marker:batch],:], cache, activation_type, 
                    loss_fcn, lambda, n_reg)

                for j = 1:length(layer_dims)-1
                    rms[string("s_dw",j)] = β .* rms[string("s_dw",j)] .+ ((1 .- β) .* (∇[string("∂W",j)]) .^ 2)'
                    rms[string("s_db",j)] = β .* rms[string("s_db",j)] .+ ((1 .- β) .* (∇[string("∂b",j)]) .^ 2)'
                end

                params = update_weights(params,(rms,∇),lr,length(layer_dims),optimiser)
                
                if size(x_train)[1] - batch < batch_size
                  marker += batch_size
                  batch += size(x_train)[1] - batch
                  if marker >= size(x_train)[1]
                    break
                  end
                else
                    marker += batch_size
                    batch += batch_size
                end
            end
            
            if verbose
                println("Iteration -> $i, Cost -> ", loss/steps_per_epoch)
            end
            
            if log
                open(string(log_file,".txt"), "a") do file
                    write(file, string(loss/steps_per_epoch))
                    write(file, "\n")
                end
            end 
        end
        if log
            open(string(log_file,".txt"), "a") do file
                write(file, "*******")
                write(file, "\n")
            end
        end
    end
    
    if optimiser == "Adam"  
        β1 = 0.9
        β2 = 0.999
        adam_opt = Dict()
        shuffled_index = [i for i = 1:size(x_train)[1]]
        
        for i = 1:length(layer_dims)-1
            adam_opt[string("v_dw",i)] = zeros(size(params[string("W",i)]))
            adam_opt[string("v_db",i)] = zeros(size(params[string("b",i)]))
            adam_opt[string("s_dw",i)] = zeros(size(params[string("W",i)]))
            adam_opt[string("s_db",i)] = zeros(size(params[string("b",i)]))
        end
        
        for i = 1:epochs
            marker = 1
            loss = 0
            batch = deepcopy(batch_size)
            if shuffle_data
                shuffle!(shuffled_index)
            end 
            
            for k = 1:steps_per_epoch
                
                Ŷ, cache, weights_sum = forward_propagation(x_train[shuffled_index[marker:batch],:], params, 
                    activation_type; n_reg)
                loss = calculate_cost(Ŷ, y_train[shuffled_index[marker:batch],:], loss_fcn, lambda, weights_sum)            
                ∇ = back_propagation(Ŷ, y_train[shuffled_index[marker:batch],:], cache, activation_type, 
                    loss_fcn, lambda, n_reg)
                
                for j = 1:length(layer_dims)-1
                    adam_opt[string("v_dw",j)] = β1 .* adam_opt[string("v_dw",j)] .+ (1 - β1) .* ∇[string("∂W",j)]'
                    adam_opt[string("v_db",j)] = β1 .* adam_opt[string("v_db",j)] .+ (1 - β1) .* ∇[string("∂b",j)]'
                    adam_opt[string("s_dw",j)] = β2 .* adam_opt[string("s_dw",j)] .+ (1 - β2) .* ((∇[string("∂W",j)]') .^ 2 )
                    adam_opt[string("s_db",j)] = β2 .* adam_opt[string("s_db",j)] .+ (1 - β2) .* ((∇[string("∂b",j)]') .^ 2 )

                    adam_opt[string("v_dw_corrected",j)] = adam_opt[string("v_dw",j)] ./ (1 - β1^j)
                    adam_opt[string("v_db_corrected",j)] = adam_opt[string("v_db",j)] ./ (1 - β1^j)
                    adam_opt[string("s_dw_corrected",j)] = adam_opt[string("s_dw",j)] ./ (1 - β2^j)
                    adam_opt[string("s_db_corrected",j)] = adam_opt[string("s_db",j)] ./ (1 - β2^j)

                end
                
                params = update_weights(params,adam_opt,lr,length(layer_dims),optimiser)
                
                if size(x_train)[1] - batch < batch_size
                  marker += batch_size
                  batch += size(x_train)[1] - batch
                  if marker >= size(x_train)[1]
                    break
                  end
                else
                    marker += batch_size
                    batch += batch_size
                end
                
            end

            if verbose
                println("Iteration -> $i, Cost -> ", loss/steps_per_epoch)
            end
            
            if log
                open(string(log_file,".txt"), "a") do file
                    write(file, string(loss/steps_per_epoch))
                    write(file, "\n")
                end
            end                
        end
        if log
            open(string(log_file,".txt"), "a") do file
                write(file, "*******")
                write(file, "\n")
            end
        end
    end
    
    return params
end




