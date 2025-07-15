using StaticArrays
using LinearAlgebra
using DifferentiationInterface
using Enzyme
using Plots
using Random
using Statistics
import ForwardDiff

## Point setup

# Define excited slice parameters
excited_lower_bound = -0.2e-2
excited_upper_bound = 0.2e-2
δ = 0.2e-2  # Slice thickness 
total_time = 3.0e-3
dt = 1e-7
num_time_segments = 30 
Nsteps = Int(round(total_time / dt))

# Pre-compute constants
segment_duration = total_time / num_time_segments
inv_segment_duration = num_time_segments / total_time

# Parameters
params = (
    γ = 2*π * 42.58e6,
    T1 = 1.0,     
    T2 = 1.0, 
    M0 = 1.0, 
    By = 0.0,
    Gz = 4e-3,
    max_slope = 70.0
)

function intelligent_Bx_initialization(num_segments, total_time, slice_thickness, target_flip_angle=π/2)    
    typical_amplitude = 1e-6
    
    Bx_init = (rand(num_segments) .- 0.5) .* 2.0 .* typical_amplitude
    
    max_val = maximum(abs.(Bx_init))
    if max_val > 8e-6
        Bx_init *= (8e-6 / max_val) * 0.9
    end
    
    return Bx_init
end


initial_Bx = intelligent_Bx_initialization(num_time_segments, total_time, δ)
println("Initial Bx values (intelligent): ", initial_Bx)
println("Initial Bx range: [$(minimum(initial_Bx)), $(maximum(initial_Bx))]")


p_initial = plot(1:num_time_segments, initial_Bx * 1e6,
    xlabel="Time Segment", ylabel="Bx Field (μT)",
    title="Initial Bx Values (Random Initialization)",
    marker=:circle, markersize=4,
    color=:blue, lw=2,
    label="Initial Bx")
display(p_initial)


dz = 2π / (params.γ * params.Gz * total_time)

function target_positions(N, dz)
    left = -floor(Int, (N - 1) / 2)
    right =  ceil(Int, (N - 1) / 2)
    return dz * collect(left:right)
end

z_positions = target_positions(3 * num_time_segments + 1, dz / 2)
num_points = length(z_positions)
m0s = fill(SA[0.0, 0.0, 1.0], num_points)

targets = [(excited_lower_bound <= z <= excited_upper_bound) ? SA[0.0, 0.5, 0.0] : SA[0.0, 0.0, 1.0] for z in z_positions]

println("Z positions (cm): ", z_positions .* 1e2)
println("Excited slice positions: ", [z for z in z_positions if excited_lower_bound <= z <= excited_upper_bound] .* 1e2)

z_positions_static = SVector{num_points}(z_positions)
m0s_static = SVector{num_points}(m0s)
targets_static = SVector{num_points}(targets)

struct ForwardEuler
end

## Optimized Functions
function get_current_Bx_fast(t, Bx_values)
    segment_idx1 = min(Int(floor(t * inv_segment_duration)) + 1, num_time_segments)
    segment_idx2 = min(Int(floor(t * inv_segment_duration)) + 2, num_time_segments)
    α = (t * inv_segment_duration) - (segment_idx1 - 1)
    return Bx_values[segment_idx1] * (1 - α) + Bx_values[segment_idx2] * α
end

function get_current_Gz(t, fixed_params)
    Gz_max = fixed_params.Gz
    max_slope = fixed_params.max_slope
    ramp_time = Gz_max / max_slope

    if t < ramp_time
        return (t / ramp_time) * Gz_max
    elseif t > total_time - ramp_time
        return max(0.0, ((total_time - t) / ramp_time) * Gz_max)
    else
        return Gz_max
    end
end

function bloch_fast(m::SVector{3,T}, Bx::T, Bz::Float64, fixed_params) where {T<:Real}
    γ, T1, T2, M0, By, _, _ = fixed_params
    
    # Pre-computed magnetic field components
    B_field = SA[Bx, By, Bz]
    
    # Cross product and relaxation terms
    cross_term = γ * cross(m, B_field)
    relaxation_term = SA[-m[1] / T2, -m[2] / T2, -(m[3] - M0) / T1]
    
    return cross_term + relaxation_term
end

function step_fast(dt, m::SVector{3,T}, t, Bx_values, z::Float64, fixed_params) where {T<:Real}
    current_Bx = get_current_Bx_fast(t, Bx_values)
    current_Gz = get_current_Gz(t, fixed_params)
    Bz = current_Gz * z
    dM = bloch_fast(m, current_Bx, Bz, fixed_params)
    return m + dM * dt
end

function solve_fast(m0::SVector{3,T}, Bx_values, z::Float64, fixed_params) where {T<:Real}
    m = m0
    for i in 1:Nsteps
        t = (i - 1) * dt
        m = step_fast(dt, m, t, Bx_values, z, fixed_params)
    end
    return m
end

# Vectorized objective function
function objective_vectorized(Bx_values; perturb_positions=true, perturb_scale=1e-4)
    if perturb_positions
        perturbations = (rand(num_points) .- 0.5) .* 2.0 .* perturb_scale
        perturbed_z_positions = z_positions_static .+ perturbations
    else
        perturbed_z_positions = z_positions_static
    end
    
    final_states = map(1:num_points) do idx
        T = eltype(Bx_values)
        m0_promoted = SVector{3,T}(m0s_static[idx])
        solve_fast(m0_promoted, Bx_values, perturbed_z_positions[idx], params)
    end

    target_transverse = [sqrt(target[1]^2 + target[2]^2) for target in targets_static]
    max_target_transverse = maximum(target_transverse)
    
    total_loss = 0.0
    
    for idx in 1:num_points
        final_state = final_states[idx]
        target = targets_static[idx]
        
        data_consistency_loss = sum(abs2, final_state[1:2] .- target[1:2])
        
        target_mag = target_transverse[idx]
        if target_mag == 0.0
            data_consistency_loss *= 2.0
        elseif target_mag == max_target_transverse
            ratio = length(filter(t -> t == 0.0, target_transverse)) / length(filter(t -> t == max_target_transverse, target_transverse))
            data_consistency_loss *= ratio
        end
        
        total_loss += data_consistency_loss
    end
    
    # Add variance loss to penalize large changes between neighbors
    variance_loss = 0.0
    λ_variance = 5
    for idx in 2:num_points
        # Penalize difference in all components of magnetization
        diff = final_states[idx] - final_states[idx-1]
        variance_loss += sum(abs2, diff)
    end
    total_loss += λ_variance * variance_loss
    
    λ = 5.65e11
    end_loss = λ * (Bx_values[end])^2
    total_loss += end_loss
    return (total_loss / num_points)
end

function gradient_descent_step(x, step_size, objective, backend)
    value, grad = value_and_gradient(objective, backend, x)
    x_new = x .- step_size .* grad
    x_new = clamp.(x_new, -8e-6, 8e-6)
    return x_new, value
end

function gradient_descent_step_with_momentum(x, v, step_size, objective, backend; momentum=0.9)
    value, grad = value_and_gradient(objective, backend, x)
    
    # Update velocity with momentum
    v_new = momentum .* v .- step_size .* grad
    
    # Update position
    x_new = x .+ v_new
    x_new = clamp.(x_new, -8e-6, 8e-6)
    
    return x_new, v_new, value
end

# function optimize(x, step_size, iters, objective)
#     # Set up forwward differentiation    
#     backend = AutoForwardDiff()
#     loss = zeros(iters)
#     final_iter = iters

#     for i in 1:iters
#         # Prevent overfitting by perturbing positions
#         perturbed_objective(Bx) = objective(Bx; perturb_positions=true)
#         # Perform a gradient descent step
#         x, loss[i] = gradient_descent_step(x, step_size, perturbed_objective, backend)
#         if i % 5 == 0 
#             println("Iteration $i: Loss = $(loss[i])")
#         end
        
#         final_iter = i
#     end
    
#     return x, loss[1:final_iter]
# end


# ## Run Optimization
# optimized_Bx, loss_history = @time optimize(initial_Bx, 1e-10, 15, objective_vectorized)

# println("Final optimized Bx values: ", optimized_Bx)

# ## Plotting Results

# ## Plot 1: Loss convergence
# p1 = plot(1:length(loss_history), loss_history, 
#          xlabel="Iteration", ylabel="Loss", 
#          title="Optimization Convergence",
#          lw=2, color=:blue)

# # Calculate final magnetization states for all points using optimized functions
# final_states = zeros(3, num_points)
# for idx in 1:num_points
#     # Use unperturbed positions for final evaluation
#     final_state = solve_fast(m0s_static[idx], optimized_Bx, z_positions_static[idx], params)
#     final_states[:, idx] = final_state
# end

# ## Plot 2: Final magnetization vs targets
# target_transverse = [sqrt(target[1]^2 + target[2]^2) for target in targets_static]
# final_transverse = sqrt.(final_states[1, :].^2 .+ final_states[2, :].^2)

# p2 = plot(z_positions * 1e2, final_transverse, marker=:circle, markersize=6, 
#          label="Final |Mxy|", color=:red, lw=2,
#          xlabel="Z Position (cm)", ylabel="Transverse Magnetization |Mxy|",
#          title="Final vs Target Transverse Magnetization")
# plot!(p2, z_positions * 1e2, target_transverse, marker=:square, markersize=6,
#       label="Target |Mxy|", color=:green, lw=2)

# ## Plot 3: Optimized Bx field over time
# time_segments = 1:num_time_segments
# t = collect(range(0, total_time, 100))
# Bx_interp = [get_current_Bx_fast(time_val, optimized_Bx) for time_val in t]

# p3 = plot(t * 1e3, Bx_interp * 1e6, 
#          xlabel="Time (ms)", ylabel="Bx Field (uT)",
#          title="Optimized Bx Field Profile",
#          color=:purple, lw=2)

# ## Plot 4: Magnetization components for all points
# p4 = plot(xlabel="Z Position", ylabel="Magnetization", 
#           title="All Magnetization Components")
# plot!(p4, z_positions, final_states[1, :], marker=:circle, label="Final Mx", color=:red)
# plot!(p4, z_positions, final_states[2, :], marker=:triangle, label="Final My", color=:blue)
# plot!(p4, z_positions, final_states[3, :], marker=:square, label="Final Mz", color=:green, lw=2)

# ## Create combined plot
# combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
# display(combined_plot)

# # Print summary statistics
# println("\n=== Optimization Summary ===")
# println("Converged in $(length(loss_history)) iterations")
# println("Final loss: $(loss_history[end])")
# println("Optimized Bx values: $(optimized_Bx)")

# # Calculate and print overall accuracy
# # my_error = sqrt(mean((target_my .- final_my).^2))
# # println("\nRMS error in My: $(round(my_error, digits=4))")
# # println("Max error in My: $(round(maximum(abs.(target_my .- final_my)), digits=4))")