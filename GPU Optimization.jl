using StaticArrays
using LinearAlgebra
using DifferentiationInterface
using Enzyme
using Plots
using Random
using Statistics
using Metal
using KernelAbstractions
using Pkg
Pkg.activate(".")

## Point setup
num_points = 20

# Define excited slice parameters
excited_lower_bound = -0.2f-2
excited_upper_bound = 0.2f-2

z_positions = collect(range(-0.7f0, 0.7f0, length=num_points)) .* 1f-2
m0s = fill(SA[0.0f0, 0.0f0, 1.0f0], num_points)

targets = [(excited_lower_bound <= z <= excited_upper_bound) ? SA[0.0f0, 1.0f0, 0.0f0] : SA[0.0f0, 0.0f0, 1.0f0] for z in z_positions]

println("Z positions (cm): ", z_positions)
println("Excited slice positions: ", [z for z in z_positions if -0.15f-2 <= z <= 0.15f-2])

total_time = 2.0f-3
dt = 1f-7
num_time_segments = 10  
Nsteps = Int(round(total_time / dt))
initial_Bx = zeros(Float32, num_time_segments)
println("Initial Bx values: ", initial_Bx)

# Pre-compute constants
segment_duration = total_time / num_time_segments
inv_segment_duration = num_time_segments / total_time

# Parameters
params = (
    γ = 2f0*π * 42.58f6,
    T1 = 1.0f0,     
    T2 = 1.0f0, 
    M0 = 1.0f0, 
    By = 0.0f0,
    Gz = 10f-3
)

# Pre-compute Bz values
Bz_values = SVector{num_points}(params.Gz .* z_positions)

z_positions_static = SVector{num_points}(z_positions)
m0s_static = SVector{num_points}(m0s)
targets_static = SVector{num_points}(targets)

# Transfer static data to GPU
const d_z_positions = MtlArray(z_positions)
const d_m0s_static = MtlArray(m0s_static)
const d_Bz_values = MtlArray(Bz_values)
const d_targets_static = MtlArray(targets_static)

# Define a backend for Metal
struct MetalBackend end
KernelAbstractions.backend(::MtlArray) = MetalBackend()
KernelAbstractions.synchronize(::MetalBackend) = Metal.synchronize()

# GPU kernel for a single point
@kernel function solve_point_kernel_ka(d_out, d_m0s, d_Bx, d_Bz, params, dt, Nsteps)
    idx = @index(Global)
    m = d_m0s[idx]
    for step in 1:Nsteps
        t = (step - 1) * dt
        m = step_fast(dt, m, t, d_Bx, d_Bz[idx], params)
    end
    d_out[idx] = m
end

# GPU kernel to compute final magnetization states per point
function gpu_solve_all!(d_out, d_m0s, d_Bx, d_Bz, params, dt, Nsteps)
    kernel = solve_point_kernel_ka(MetalBackend(), 256)
    kernel(d_out, d_m0s, d_Bx, d_Bz, params, dt, Nsteps, ndrange=length(d_m0s))
    Metal.synchronize()
end

# GPU objective function
function objective_gpu(Bx_values)
    # Clamp Bx values - this is fine on CPU as Bx_values is small
    for i in eachindex(Bx_values)
        Bx_values[i] = clamp(Bx_values[i], -8e-6, 8e-6)
    end
    d_Bx_values = MtlArray(BxValues)

    # Allocate output array on GPU
    d_final_states = MtlArray{SVector{3,Float32}}(undef, num_points)

    # Run simulation on GPU
    gpu_solve_all!(d_final_states, d_m0s_static, d_Bx_values, d_Bz_values, params, dt, Nsteps)

    # Compute loss on GPU
    total_loss = sum(abs2, d_final_states .- d_targets_static)
    
    return total_loss
end


struct ForwardEuler
end

## Optimized Functions
@inline function get_current_Bx_fast(t, Bx_values)
    segment_idx = min(Int(floor(t * inv_segment_duration)) + 1, num_time_segments)
    return Bx_values[segment_idx]
end

@inline function bloch_fast(m::SVector{3,Float32}, Bx::Float32, Bz::Float32, fixed_params)
    γ, T1, T2, M0, By, _ = fixed_params
    
    # Pre-computed magnetic field components
    B_field = SA[Bx, By, Bz]
    
    # Cross product and relaxation terms
    cross_term = γ * cross(m, B_field)
    relaxation_term = SA[-m[1] / T2, -m[2] / T2, -(m[3] - M0) / T1]
    
    return cross_term + relaxation_term
end

@inline function step_fast(dt, m, t, Bx_values, Bz::Float32, fixed_params)
    current_Bx = get_current_Bx_fast(t, Bx_values)
    dM = bloch_fast(m, current_Bx, Bz, fixed_params)
    return m + dM * dt
end

function solve_fast(m0::SVector{3,Float32}, Bx_values, Bz::Float32, fixed_params)
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

    for i in eachindex(Bx_values)
        Bx_values[i] = clamp(BxValues[i], -8e-6, 8e-6)
    end

    # Compute losses for all points at once
    total_loss = 0.0f0
    zero_target_penalty = 0.0f0
    
    for idx in 1:num_points
        final_state = final_states[idx]
        target = targets_static[idx]
        
        # Standard data consistency loss
        data_consistency_loss = sum(abs2, final_state .- target)
        total_loss += data_consistency_loss
        
        # for component in 1:3
        #     if abs(target[component]) < 1e-8
        #         final_val = final_state[component]
        #         zero_target_penalty += final_val^2 
        #     end
        # end
    end
    # B1 power loss
    # B1_power_loss = sum(abs2, Bx_values)
    # total_loss += zero_target_penalty + B1_power_loss * 5e3
    # total_loss += zero_target_penalty
    
    return total_loss
end

function gradient_descent_step(x, step_size, objective)
    value, grad = value_and_gradient(objective, AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)), x)
    x_new = x .- step_size .* grad
    return x_new, value
end

function optimize(x, step_size, iters, objective)
    println("Starting optimization...")
    
    loss = zeros(Float32, iters)
    final_iter = iters
    
    tolerance_percent = 0.1f0
    patience = 10 
    min_iterations = 10 
    no_improvement_count = 0
    best_loss = Inf32
    
    for i in 1:iters 
        x, loss[i] = gradient_descent_step(x, step_size, objective)
        
        # Early stopping logic
        if i >= min_iterations
            if best_loss > 0
                improvement_percent = (best_loss - loss[i]) / best_loss * 100
            else
                improvement_percent = abs(loss[i] - best_loss)
            end
            
            if improvement_percent > tolerance_percent
                best_loss = loss[i]
                no_improvement_count = 0
            else
                no_improvement_count += 1
            end
            
            if no_improvement_count >= patience
                println("Early stopping at iteration $i - no improvement > $(tolerance_percent)% for $patience iterations")
                final_iter = i
                break
            end
        end
        
        if i % 50 == 0 
            println("Iteration $i: Loss = $(loss[i]) | Best: $best_loss | No improvement: $no_improvement_count")
        end
        
        final_iter = i
    end
    
    return x, loss[1:final_iter]
end

## Run Optimization
optimized_Bx, loss_history = optimize(initial_Bx, 2f-11, 500, objective_gpu)

println("Final optimized Bx values: ", optimized_Bx)
println("Final loss: ", loss_history[end])

## Plotting Results

## Plot 1: Loss convergence
p1 = plot(1:length(loss_history), loss_history, 
         xlabel="Iteration", ylabel="Loss", 
         title="Optimization Convergence",
         lw=2, color=:blue)

# Calculate final magnetization states for all points using optimized functions
d_Bx_optimized = MtlArray(optimized_Bx)
d_final_states = MtlArray{SVector{3,Float32}}(undef, num_points)
gpu_solve_all!(d_final_states, d_m0s_static, d_Bx_optimized, d_Bz_values, params, dt, Nsteps)
final_states_cpu = Array(d_final_states)

final_states = zeros(Float32, 3, num_points)
for idx in 1:num_points
    final_states[:, idx] = final_states_cpu[idx]
end


## Plot 2: Final magnetization vs targets
target_my = [target[2] for target in targets_static]
final_my = final_states[2, :]

p2 = plot(z_positions * 1e2, final_my, marker=:circle, markersize=6, 
         label="Final My", color=:red, lw=2,
         xlabel="Z Position (cm)", ylabel="Magnetization (My)",
         title="Final vs Target Magnetization")
plot!(p2, z_positions * 1e2, target_my, marker=:square, markersize=6,
      label="Target My", color=:green, lw=2)

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
plot!(p4, z_positions, final_states[2, :], marker=:triangle, label="Final My", color=:blue)
plot!(p4, z_positions, final_states[3, :], marker=:square, label="Final Mz", color=:green, lw=2)

## Create combined plot
combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
display(combined_plot)

# Print summary statistics
println("\n=== Optimization Summary ===")
println("Converged in $(length(loss_history)) iterations")
println("Final loss: $(loss_history[end])")
println("Optimized Bx values: $(optimized_Bx)")

# Calculate and print overall accuracy
my_error = sqrt(mean((target_my .- final_my).^2))
println("\nRMS error in My: $(round(my_error, digits=4))")
println("Max error in My: $(round(maximum(abs.(target_my .- final_my)), digits=4))")

# Replace CPU arrays and loops with MetalArrays for GPU acceleration
# using MetalArrays - Already imported

# Transfer data to GPU
# const d_z_positions   = MetalArray(z_positions)
# const d_m0s           = MetalArray(m0s_static)
# const d_Bz_values     = MetalArray(Bz_values)
# const d_Bx_optimized  = MetalArray(optimized_Bx)

# GPU compute for interpolated Bx over time
t_cpu       = collect(range(0, total_time, 100))
d_t         = MtlArray(t_cpu)
d_Bx_interp = MtlArray(get_current_Bx_fast.(d_t, Ref(d_Bx_optimized)))
# Bring back to CPU for plotting
Bx_interp   = Array(d_Bx_interp)

# GPU kernel to compute final magnetization states per point
# function gpu_solve_all!(d_out, d_m0s, d_Bx, d_Bz, params, dt, Nsteps)
#     @metal for idx in 1:length(d_m0s)
#         m = d_m0s[idx]
#         for step in 1:Nsteps
#             t  = (step-1)*dt
#             m  = step_fast(dt, m, t, d_Bx, d_Bz[idx], params)
#         end
#         d_out[idx] = m
#     end
# end

# Execute GPU computation
# d_final_states = MetalArray{SVector{3,Float64}}(undef, num_points)
# gpu_solve_all!(d_final_states, d_m0s, d_Bx_optimized, d_Bz_values, params, dt, Nsteps)

# Bring final results back to CPU
# final_states = Array(d_final_states)