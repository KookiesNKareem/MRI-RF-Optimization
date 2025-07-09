using StaticArrays
using LinearAlgebra
using DifferentiationInterface
import Enzyme, ForwardDiff
using Plots
using Printf

## Point setup
num_time_segments = 35  

# Define excited slice parameters
slice_thickness = 4e-3
δ = 0.5e-3 # regions of slice edge +- δ are relaxed
total_time = 5.0e-3

# Slice Parameters
γ = 2π * 42.58e6
Gz = 10e-3
dz = 2π / (γ * Gz * total_time)

function target_positions(N, dz)
    left = -floor(Int, (N - 1) / 2)
    right =  ceil(Int, (N - 1) / 2)
    return dz * collect(left:right)
end

# This avoids aliasing (Nyquist criterion)
z_positions = target_positions(2 * num_time_segments + 1, dz / 2)
num_points = length(z_positions)

params = map(z_positions) do z
    (
        γ = γ,
        T1 = 1000.0,     
        T2 = 1000.0, 
        M0 = 1.0, 
        Bz = Gz * z,
    )
end

# ROI definition. Weights from:
# Yang J, Nielsen J-F, Fessler JA, Jiang Y. Multidimensional RF pulse design using auto-differentiable spin-domain optimization and its application to reduced field-of-view imaging. Magn Reson Med. 2025; 1-19. doi: 10.1002/mrm.30607
inner_edge = slice_thickness / 2 - δ
outer_edge = slice_thickness / 2 + δ
weights_factor = sum(abs.(z_positions) .>= outer_edge) / sum(abs.(z_positions) .<= inner_edge)
weights = ones(length(z_positions))
# weights = map(z_positions) do z
#     if abs(z) <= inner_edge
#         weights_factor
#     elseif abs(z) >= outer_edge
#         1.0
#     else
#         0.0
#     end
# end

println("FOV: ", round.((z_positions[end] - z_positions[1]) * 1e3; digits=2), " [mm], dz: ", round(dz * 1e3; digits=2), " [mm]")
# z_positions = collect(range(-0.7, 0.7, length=num_points)) .* 1e-2
m0s = fill(SA[0.0, 0.0, 1.0], num_points)

targets = [(abs(z) <= slice_thickness / 2) ? SA[0.0, 0.5, 0.0] : SA[0.0, 0.0, 1.0] for z in z_positions]

dt = 1e-7
Nsteps = Int(round(total_time / dt))
x0 = zeros(2num_time_segments)

# Pre-compute constants
segment_duration = total_time / num_time_segments
inv_segment_duration = (num_time_segments - 1) / total_time

# Pre-compute Bz values
m0s_static = SVector{num_points}(m0s)
targets_static = SVector{num_points}(targets)

struct ForwardEuler
end

## Optimized Functions
function linear_interp(t, x)
    idx1 = min(Int(floor(t * inv_segment_duration)) + 1, num_time_segments)
    idx2 = min(Int(floor(t * inv_segment_duration)) + 2, num_time_segments)
    α = (t * inv_segment_duration) - (idx1 - 1)
    return x[idx1] * (1 - α) + x[idx2] * α
end

function bloch(m, Bx, By, params)
    γ, T1, T2, M0, Bz = params
    
    # Pre-computed magnetic field components
    B_field = SA[Bx, By, Bz]
    
    # Cross product and relaxation terms
    cross_term = γ * cross(m, B_field)
    relaxation_term = SA[-m[1] / T2, -m[2] / T2, -(m[3] - M0) / T1]
    
    return cross_term + relaxation_term
end

function step(dt, m, t, Bx_nodes, By_nodes, params)
    Bx = linear_interp(t, Bx_nodes)
    By = linear_interp(t, By_nodes)
    dM = bloch(m, Bx, By, params)
    return m + dM * dt
end

function solve(m0, Bx_nodes, By_nodes, params)
    m = m0
    for i in 1:Nsteps
        t = (i - 1) * dt
        m = step(dt, m, t, Bx_nodes, By_nodes, params)
    end
    return m
end

# Vectorized objective function
function objective(x, params)
    # Unpack field
    Bx_nodes = x[1:end÷2]
    By_nodes = x[end÷2+1:end]
    # Compute losses for all points at once - only comparing Mx and My
    total_loss = 0.0
    for i in 1:num_points
        final_state = solve(m0s_static[i], Bx_nodes, By_nodes, params[i])
        target = targets_static[i]
        total_loss += weights[i] * sum(abs2, final_state[1:2] .- target[1:2]) / num_points
    end
    return total_loss
end

backend = AutoForwardDiff() # AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
∇f_prep = prepare_gradient(objective, backend, zero(x0), Constant(params))

function optimize(x, step_size, Niters, f, ∇f_prep)
    println("Starting optimization...")

    ∇f = similar(x)
    ∇f_prev = similar(x)
    x_prev = copy(x)

    loss = zeros(Niters)   
    sol = zeros(length(x), Niters) 

    # First iter
    λ = step_size
    x_prev .= x
    gradient!(f, ∇f, ∇f_prep, backend, x, Constant(params))
    @. x -= λ * ∇f

    sol[:, 1] = x
    loss[1] = objective(x, params)
    @printf "Iteration %d: Loss = %.6f | λ = %.2e\n" 1 loss[1] λ

    θ = Inf
    ∇f_prev .= ∇f
    λ_prev = λ

    # From Malitsky, Yura, and Konstantin Mishchenko. "Adaptive gradient descent without descent." (2019).
    for i in 2:Niters
        # Calculate gradient 
        gradient!(f, ∇f, ∇f_prep, backend, x, Constant(params))
        # Malitsky-Mishchenko step size
        dx = x .- x_prev
        dg = ∇f .- ∇f_prev
        λ = min(√(1 + θ) * λ_prev, norm(dx) / (2 * norm(dg)))
        # Update
        x_prev .= x
        θ = λ / λ_prev
        λ_prev = λ
        ∇f_prev .= ∇f
        # Step
        @. x -= λ * ∇f
        # Save
        sol[:, i] = x
        loss[i] = objective(x, params)
        @printf "Iteration %d: Loss = %.6f | λ = %.2e\n" i loss[i] λ
        # Early stopping condition
        if abs(loss[i] - loss[i-1]) < 1e-6
            println("Convergence reached at iteration $i")
            loss = loss[1:i]
            sol = sol[:, 1:i]
            break
        end
    end
    return sol, loss
end

## Run Optimization
step_size = 1e-10
Niters = 30

sol_history, loss_history = @time optimize(x0, step_size, Niters, objective, ∇f_prep)
x_opt = sol_history[:, end]

## Plotting Results

# Plot 1: Loss convergence
p1 = plot(1:length(loss_history), loss_history, 
         xlabel="Iteration", ylabel="Loss", 
         title="Optimization Convergence",
         lw=2, color=:blue, label=nothing)

# Calculate final magnetization states for all points using optimized functions
optimized_Bx = x_opt[1:end÷2]
optimized_By = x_opt[end÷2+1:end]
final_states = zeros(3, num_points)
for idx in 1:num_points
    final_state = solve(m0s_static[idx], optimized_Bx, optimized_By, params[idx])
    final_states[:, idx] = final_state
end

# Plot 2: Final magnetization vs targets
target_transverse = [sqrt(target[1]^2 + target[2]^2) for target in targets_static]
final_transverse = sqrt.(final_states[1, :].^2 .+ final_states[2, :].^2)

p2 = plot(z_positions * 1e3, final_transverse, marker=:circle, markersize=6, 
         label="Final |Mxy|", color=:red, lw=2,
         xlabel="Z Position (mm)", ylabel="Transverse Magnetization |Mxy|",
         title="Final vs Target Transverse Magnetization")
plot!(p2, z_positions * 1e3, target_transverse, marker=:square, markersize=6,
      label="Target |Mxy|", color=:green, lw=2)
vline!(p2, 1e3 * [- slice_thickness / 2 - δ, - slice_thickness / 2 + δ, slice_thickness / 2 - δ, slice_thickness / 2 + δ], label=nothing, color=:black, linestyle=:dash)

# Plot 3: Optimized Bx field over time
time_interval = range(0, total_time, 100)
Bx_interp = [linear_interp(t, optimized_Bx) for t in time_interval]
By_interp = [linear_interp(t, optimized_By) for t in time_interval]

p3 = plot(time_interval * 1e3, Bx_interp * 1e6, 
         xlabel="Time (ms)", ylabel="B1 (uT)",
         title="Optimized Bx Field Profile",
         label="Bx",
         color=:purple, lw=2)
plot!(p3, time_interval * 1e3, By_interp * 1e6, label="By", color=:orange, lw=2)

# Plot 4: Magnetization components for all points
p4 = plot(xlabel="Z Position (mm)", ylabel="Magnetization", 
          title="All Magnetization Components")
plot!(p4, z_positions * 1e3, final_states[1, :], marker=:circle, label="Final Mx", color=:red)
plot!(p4, z_positions * 1e3, final_states[2, :], marker=:circle, label="Final My", color=:blue)
plot!(p4, z_positions * 1e3, final_states[3, :], marker=:circle, label="Final Mz", color=:green, lw=2)


# Create combined plot
combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
display(combined_plot)

# Print summary statistics
println("\n=== Optimization Summary ===")
println("Converged in $(length(loss_history)) iterations")
println("Final loss: $(loss_history[end])")
# println("Optimized Bx values: $(x_opt)")

# Calculate and print overall accuracy
# my_error = sqrt(mean((target_my .- final_my).^2))
# println("\nRMS error in My: $(round(my_error, digits=4))")
# println("Max error in My: $(round(maximum(abs.(target_my .- final_my)), digits=4))")

# Enzyme: 406.229149 seconds (1.61 G allocations: 63.397 GiB, 12.47% gc time, 1.03% compilation time: 68% of which was recompilation)
# Mooncake: 1066.555720 seconds (11.55 G allocations: 278.927 GiB, 6.37% gc time, 7.49% compilation time)