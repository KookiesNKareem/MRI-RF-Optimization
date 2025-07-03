m0s         = [M_eq for z in zs]
targets     = [(abs(z) <= slice_thickness) ? M_ex : M_eq for z in zs]

function objective(x)
    total_loss = 0.0
    for i in 1:num_spins
        params[i].x = x
        final_state_i = solve(m0s[i], params[i])
        total_loss += sum(abs2, final_state_i .- targets[i])
    end
    return total_loss
end

function gradient_descent(x, step_size, num_iters, objective)
    println("Starting optimization...")
    loss = zeros(num_iters)
    for i in 1:num_iters 
        loss[i], grad = value_and_gradient(objective, backend, x)
        x = x - step_size * grad
        # Optional: Print loss every 50 iterations
        if (i % 50 == 0)
            println("Iteration $i: Loss = $(loss[i])")
        end
    end
    return x, loss
end