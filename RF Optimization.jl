using StaticArrays
using LinearAlgebra
using DifferentiationInterface
using Enzyme
using Plots

## Point setup
num_time_segments = 25  

# Define excited slice parameters
excited_lower_bound = -0.2e-2
excited_upper_bound = 0.2e-2

total_time = 4.0e-3

# Parameters
params = (
    γ = 2π * 42.58e6,
    T1 = 1000.0,     
    T2 = 1000.0, 
    M0 = 1.0, 
    By = 0.0,
    Gz = 10e-3
)

dz = 2π / (params.γ * params.Gz * total_time)

function target_positions(N, dz)
    left = -floor(Int, (N - 1) / 2)
    right =  ceil(Int, (N - 1) / 2)
    return dz * collect(left:right)
end

# This avoids aliasing (Nyquist criterion)
z_positions = target_positions(2 * num_time_segments, dz / 2)
num_points = length(z_positions)

println("FOV: ", round.((z_positions[end] - z_positions[1]) * 1e3; digits=2), " [mm], dz: ", round(dz * 1e3; digits=2), " [mm]")
# z_positions = collect(range(-0.7, 0.7, length=num_points)) .* 1e-2
m0s = fill(SA[0.0, 0.0, 1.0], num_points)

targets = [(excited_lower_bound <= z <= excited_upper_bound) ? SA[0.0, 0.5, 0.0] : SA[0.0, 0.0, 1.0] for z in z_positions]

dt = 1e-7
Nsteps = Int(round(total_time / dt))
initial_Bx = zeros(num_time_segments)

# Pre-compute constants
segment_duration = total_time / num_time_segments
inv_segment_duration = (num_time_segments - 1) / total_time

# Pre-compute Bz values
Bz_values = SVector{num_points}(params.Gz .* z_positions)

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

function bloch_fast(m, Bx, Bz, fixed_params)
    γ, T1, T2, M0, By, _ = fixed_params
    
    # Pre-computed magnetic field components
    B_field = SA[Bx, By, Bz]
    
    # Cross product and relaxation terms
    cross_term = γ * cross(m, B_field)
    relaxation_term = SA[-m[1] / T2, -m[2] / T2, -(m[3] - M0) / T1]
    
    return cross_term + relaxation_term
end

function step_fast(dt, m, t, Bx_values, Bz, fixed_params)
    current_Bx = get_current_Bx_fast(t, Bx_values)
    dM = bloch_fast(m, current_Bx, Bz, fixed_params)
    return m + dM * dt
end

function solve_fast(m0, Bx_values, Bz, fixed_params)
    m = m0
    for i in 1:Nsteps
        t = (i - 1) * dt
        m = step_fast(dt, m, t, Bx_values, Bz, fixed_params)
    end
    return m
end

# Vectorized objective function
function objective_vectorized(Bx_values)
    # Solve for all points using vectorized operations
    final_states = map(1:num_points) do idx
        solve_fast(m0s_static[idx], Bx_values, Bz_values[idx], params)
    end

    # Compute losses for all points at once - only comparing Mx and My
    total_loss = 0.0

    # λ = 0.1
    
    for idx in 1:num_points
        final_state = final_states[idx]
        target = targets_static[idx]
        
        # Standard data consistency loss - only Mx and My components
        total_loss += sum(abs2, final_state[1:2] .- target[1:2]) / num_points

        # # Total variation
        # if idx < num_points
        #     total_loss += λ * sum(abs2, final_states[idx+1][1:2] .- final_states[idx][1:2]) / (num_points - 1)
        # end
    end
    
    return total_loss
end

backend = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
function gradient_descent_step(x, step_size, objective)
    loss, grad = value_and_gradient(objective, backend, x)
    x_new = x .- step_size .* grad
    # Apply constraints via projection (projected gradient method)
    # x_new = clamp.(x_new, -8e-6, 8e-6)
    return x_new, loss
end

function optimize(x, step_size, iters, objective)
    println("Starting optimization... step_size: ", step_size)
    loss = zeros(iters)   
    sol = zeros(num_time_segments, iters) 
    for i in 1:iters 
        x, loss[i] = gradient_descent_step(x, step_size, objective)
        sol[:, i] = x
        # if i % 10 == 0
            println("Iteration $i: Loss = $(loss[i])")
        # end
    end
    return sol, loss
end


## Run Optimization
sol_history, loss_history = @time optimize(initial_Bx, 2e-10, 1, objective_vectorized)
optimized_Bx = sol_history[:, end]

## Plotting Results

## Plot 1: Loss convergence
p1 = plot(1:length(loss_history), loss_history, 
         xlabel="Iteration", ylabel="Loss", 
         title="Optimization Convergence",
         lw=2, color=:blue)

# Calculate final magnetization states for all points using optimized functions
final_states = zeros(3, num_points)
for idx in 1:num_points
    final_state = solve_fast(m0s_static[idx], optimized_Bx, Bz_values[idx], params)
    final_states[:, idx] = final_state
end

## Plot 2: Final magnetization vs targets
target_transverse = [sqrt(target[1]^2 + target[2]^2) for target in targets_static]
final_transverse = sqrt.(final_states[1, :].^2 .+ final_states[2, :].^2)

p2 = plot(z_positions * 1e2, final_transverse, marker=:circle, markersize=6, 
         label="Final |Mxy|", color=:red, lw=2,
         xlabel="Z Position (cm)", ylabel="Transverse Magnetization |Mxy|",
         title="Final vs Target Transverse Magnetization")
plot!(p2, z_positions * 1e2, target_transverse, marker=:square, markersize=6,
      label="Target |Mxy|", color=:green, lw=2)

## Plot 3: Optimized Bx field over time
time_segments = 1:num_time_segments
t = collect(range(0, total_time, 100))
Bx_interp = [get_current_Bx_fast(time_val, optimized_Bx) for time_val in t]

p3 = plot(t * 1e3, Bx_interp * 1e6, 
         xlabel="Time (ms)", ylabel="Bx Field (uT)",
         title="Optimized Bx Field Profile",
         color=:purple, lw=2)

## Plot 4: Magnetization components for all points
p4 = plot(xlabel="Z Position", ylabel="Magnetization", 
          title="All Magnetization Components")
plot!(p4, z_positions, final_states[1, :], marker=:circle, label="Final Mx", color=:red)
plot!(p4, z_positions, final_states[2, :], marker=:circle, label="Final My", color=:blue)
plot!(p4, z_positions, final_states[3, :], marker=:circle, label="Final Mz", color=:green, lw=2)


# Create combined plot
combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
display(combined_plot)

# Print summary statistics
println("\n=== Optimization Summary ===")
println("Converged in $(length(loss_history)) iterations")
println("Final loss: $(loss_history[end])")
# println("Optimized Bx values: $(optimized_Bx)")

# Calculate and print overall accuracy
# my_error = sqrt(mean((target_my .- final_my).^2))
# println("\nRMS error in My: $(round(my_error, digits=4))")
# println("Max error in My: $(round(maximum(abs.(target_my .- final_my)), digits=4))")