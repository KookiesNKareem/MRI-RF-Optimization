using StaticArrays
using LinearAlgebra
using DifferentiationInterface
import Enzyme, ForwardDiff
using Plots
using Printf

## Point setup
num_time_segments = 31 # Input
num_points = 41 # Output

# Define excited slice parameters
slice_thickness = 4e-3
total_time = 10.0e-3

# Slice Parameters
γ = 2π * 42.58e6
Gz = 10e-3

# This avoids aliasing (Nyquist criterion)
z_positions = range(-3slice_thickness/2, 3slice_thickness/2, num_points)

params = map(z_positions) do z
    (γ = γ,
     T1 = 1000.0,
     T2 = 1000.0,
     M0 = 1.0,
     Bz = Gz * z)
end

m0s = fill(SA[0.0, 0.0, 1.0], num_points)
# Slice profile following 10th order Butterworth filter
targets = map(z_positions) do z
    slice_profile = 1 / sqrt.(1 + (2z / slice_thickness) ^ 20)
    SA[0.0, slice_profile] # Mx, My
end

dt = 1e-7
Nsteps = Int(round(total_time / dt))
x0 = 1e-6 * rand(2num_time_segments)

# Pre-compute constants
segment_duration = total_time / num_time_segments
inv_segment_duration = (num_time_segments - 1) / total_time

# Pre-compute Bz values
m0s_static = SVector{num_points}(m0s)
targets_static = SVector{num_points}(targets)

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
        total_loss += sum(abs2, final_state[1:2] .- targets_static[i]) / num_points
    end
    # total_loss += 1e12 * (Bx_nodes' * Bx_nodes + By_nodes' * By_nodes)
    return total_loss
end

backend = AutoForwardDiff() # AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
∇f_prep = prepare_gradient(objective, backend, zero(x0), Constant(params))

function optimize(x, step_size, Niters, f, ∇f_prep;
                  β1=0.9, β2=0.999, ε=1e-8)
    println("Starting Adam optimization...")

    ∇f = similar(x)
    m = zeros(length(x))       # 1st moment vector
    v = zeros(length(x))       # 2nd moment vector

    loss = zeros(Niters)
    sol = zeros(length(x), Niters)

    # First iteration
    gradient!(f, ∇f, ∇f_prep, backend, x, Constant(params))
    @. m = (1 - β1) * ∇f
    @. v = (1 - β2) * ∇f^2
    m̂ = m ./ (1 - β1)
    v̂ = v ./ (1 - β2)
    @. x -= step_size * m̂ / (sqrt(v̂) + ε)

    sol[:, 1] = x
    loss[1] = objective(x, params)
    @printf "Iteration %d: Loss = %.6f\n" 1 loss[1]

    for i in 2:Niters
        gradient!(f, ∇f, ∇f_prep, backend, x, Constant(params))

        # Update biased moments
        @. m = β1 * m + (1 - β1) * ∇f
        @. v = β2 * v + (1 - β2) * ∇f^2

        # Bias-corrected
        m̂ = m ./ (1 - β1^i)
        v̂ = v ./ (1 - β2^i)

        # Step
        @. x -= step_size * m̂ / (sqrt(v̂) + ε)

        # Save
        sol[:, i] = x
        loss[i] = objective(x, params)
        @printf "Iteration %d: Loss = %.6f\n" i loss[i]

        # Early stopping
        if abs(loss[i] - loss[i-1]) < 1e-5
            println("Convergence reached at iteration $i")
            loss = loss[1:i]
            sol = sol[:, 1:i]
            break
        end
    end

    return sol, loss
end

## Run Optimization
step_size = 1e-6
Niters = 100

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
for i in 1:num_points
    final_state = solve(m0s_static[i], optimized_Bx, optimized_By, params[i])
    final_states[:, i] = final_state
end
# Plot 2: Final magnetization vs targets
target_transverse = [norm(target) for target in targets_static]
final_transverse = sqrt.(final_states[1, :].^2 .+ final_states[2, :].^2)
p2 = plot(z_positions * 1e3, final_transverse, marker=:circle, markersize=6, 
         label="Final |Mxy|", color=:red, lw=2,
         xlabel="Z Position (mm)", ylabel="Transverse Magnetization |Mxy|",
         title="Final vs Target Transverse Magnetization")
plot!(p2, z_positions * 1e3, target_transverse, marker=:square, markersize=6,
      label="Target |Mxy|", color=:green, lw=2)
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
