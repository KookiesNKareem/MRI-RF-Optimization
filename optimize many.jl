using StaticArrays
using LinearAlgebra
using DifferentiationInterface
using Enzyme
using Plots
using Random

## 
Random.seed!(42)
targets = rand(3, 10)
X = rand(3, 10)
## 


params = (
    γ = 42.58e6,
    B = 1e-6,      
    T1 = 1.0,     
    T2 = 0.5,     
    M0 = 1.0,   
)
dt = 0.001 
struct ForwardEuler
end

function bloch(m, params)
    γ, B, T1, T2, M0 = params
    cross_term = γ * cross(m, [0, 0, B])
    relaxation_term = [m[1] / T2, m[2] / T2, (m[3] - M0) / T1]
    dM = cross_term - relaxation_term
    return dM
end

function step(dt, m, params, ::ForwardEuler)
    dM = bloch(m, params)
    m_next = m + dM .* dt
    return m_next
end

function solve(m0, dt, tmax, params, method)
    Nsteps = Int(round(tmax / dt))
    mt = zeros(3, Nsteps + 1)
    mt[:, 1] = m0
    for i in 1:Nsteps
        mt[:, i + 1] = step(dt, mt[:, i], params, method)
    end
    return mt
end

function solve_objective(X)
    final_states = zeros(3, size(X, 2))
    for i in 1:size(X, 2)
        m0 = X[:, i]
        mt = solve(m0, dt, 1.0, params, ForwardEuler())
        final_states[:, i] = mt[:, end]
    end
    return sum(abs2, final_states .- targets)
end

function gradient_descent_step(x, t, objective)
    value, grad = value_and_gradient(objective, AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Forward)), x)
    x_new = x .- t .* grad
    return x_new, value
end

function optimize(x, step_size, iters, objective)
    println("Starting optimization...")
    println("Initial state: ", X)
    println("Target state: ", targets)
    
    loss = zeros(iters)
    
    for i in 1:iters 
        x, loss[i] = gradient_descent_step(x, step_size, objective)
        
        if i % 100 == 0
            println("Iteration $i: Loss = $(loss[i])")
        end
    end
    
    return x, loss
end

x_solution, loss = optimize(X, 1e-2, 1500, solve_objective)

##
p1 = plot(loss, title="Optimization Loss", xlabel="Iteration", ylabel="Loss", 
          label="Loss", legend=:topright, lw=2)
p = map(1:6) do i
    time_steps = 0:dt:1.0
    mt_optimal = solve(x_solution[:, i], dt, 1.0, params, ForwardEuler())
    p = plot(time_steps, mt_optimal[1, :], label=nothing, lw=2, xlabel="Time (s)", ylabel="Magnetization", legend=:topright)
    plot!(time_steps, mt_optimal[2, :], label=nothing, lw=2)
    plot!(time_steps, mt_optimal[3, :], label=nothing, lw=2)
    hline!([targets[1, i]], ls=:dash, color=1, alpha=0.7, label=nothing)
    hline!([targets[2, i]], ls=:dash, color=2, alpha=0.7, label=nothing)
    hline!([targets[3, i]], ls=:dash, color=3, alpha=0.7, label=nothing)
    title!("Optimal Magnetization Trajectory")
end


final_plot = plot(p..., layout=(3,2), size=(800, 600))
display(final_plot)