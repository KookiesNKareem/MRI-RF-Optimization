
using Optim
using DifferentiationInterface
using StaticArrays
using LinearAlgebra
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
const total_time = 4.0e-3
const dt = 1e-7
const num_time_segments = 40 
const Nsteps = Int(round(total_time / dt))

# Pre-compute constants
const segment_duration = total_time / num_time_segments
const inv_segment_duration = num_time_segments / total_time

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

z_positions_static = SVector{num_points}(z_positions)
m0s_static = SVector{num_points}(m0s)
targets_static = SVector{num_points}(targets)

struct ForwardEuler
end

## Optimized Functions
function get_current_Bx_fast(t, Bx_values, num_time_segments, inv_segment_duration)
    segment_idx1 = min(Int(floor(t * inv_segment_duration)) + 1, num_time_segments)
    segment_idx2 = min(Int(floor(t * inv_segment_duration)) + 2, num_time_segments)
    α = (t * inv_segment_duration) - (segment_idx1 - 1)
    return Bx_values[segment_idx1] * (1 - α) + Bx_values[segment_idx2] * α
end

function get_current_Gz(t, fixed_params, total_time)
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
    _Bx, _By, _Bz = promote(Bx, By, Bz)
    B_field = SA[_Bx, _By, _Bz]
    
    # Cross product and relaxation terms
    cross_term = γ * cross(m, B_field)
    relaxation_term = SA[-m[1] / T2, -m[2] / T2, -(m[3] - M0) / T1]
    
    return cross_term + relaxation_term
end

function step_fast(dt, m::SVector{3,T}, t, Bx_values, z::Float64, fixed_params, num_time_segments, inv_segment_duration, total_time) where {T<:Real}
    current_Bx = get_current_Bx_fast(t, Bx_values, num_time_segments, inv_segment_duration)
    current_Gz = get_current_Gz(t, fixed_params, total_time)
    Bz = current_Gz * z
    dM = bloch_fast(m, current_Bx, Bz, fixed_params)
    return m + dM * dt
end

function solve_fast(m0::SVector{3,T}, Bx_values, z::Float64, fixed_params, num_time_segments, inv_segment_duration, total_time, Nsteps) where {T<:Real}
    m = m0
    for i in 1:Nsteps
        t = (i - 1) * dt
        m = step_fast(dt, m, t, Bx_values, z, fixed_params, num_time_segments, inv_segment_duration, total_time)
    end
    return m
end

# Vectorized objective function
function objective_vectorized(Bx_values, num_points, z_positions_static, m0s_static, targets_static, num_time_segments, inv_segment_duration, total_time, Nsteps; perturb_positions=true, perturb_scale=1e-4)
    if perturb_positions
        perturbations_vec = (rand(num_points) .- 0.5) .* 2.0 .* perturb_scale
        perturbations_svec = SVector{num_points}(perturbations_vec)
        perturbed_z_positions = z_positions_static .+ perturbations_svec
    else
        perturbed_z_positions = z_positions_static
    end
    
    T = eltype(Bx_values) # Get the element type

    final_states = map(1:num_points) do idx
        m0_promoted = SVector{3,T}(m0s_static[idx])
        solve_fast(m0_promoted, Bx_values, perturbed_z_positions[idx], params, num_time_segments, inv_segment_duration, total_time, Nsteps)
    end

    target_transverse = [sqrt(target[1]^2 + target[2]^2) for target in targets_static]
    max_target_transverse = maximum(target_transverse)
    
    total_loss = T(0.0) # Initialize with the correct type
    
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
    # variance_loss = T(0.0) # Initialize with the correct type
    # λ_variance = 1.5
    # for idx in 2:num_points
    #     # Penalize difference in all components of magnetization
    #     diff = final_states[idx] - final_states[idx-1]
    #     variance_loss += sum(abs2, diff)
    # end
    # total_loss += λ_variance * variance_loss
    
    λ = 5.65e11
    end_loss = λ * (Bx_values[end])^2
    total_loss += end_loss
    return (total_loss / num_points)
end

result = Optim.optimize(x -> objective_vectorized(x, num_points, z_positions_static, m0s_static, targets_static, num_time_segments, inv_segment_duration, total_time, Nsteps; perturb_positions=false), initial_Bx, LBFGS(), Optim.Options(iterations=15, show_every=1, show_trace=true), ; autodiff=AutoEnzyme())
plot(result.minimizer, title="Optimized Bx", xlabel="Segment Index", ylabel="Bx Value", label="Bx", legend=:topright)
println("Optimized Bx values: ", result.minimizer)