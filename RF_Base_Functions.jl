using StaticArrays, LinearAlgebra, Random, Statistics
using DifferentiationInterface
using Plots
import ForwardDiff

const total_time        = 4.0e-3
const dt                = 8.0e-5
const Nsteps            = Int(cld(total_time, dt))
const num_time_segments = 50
const inv_seg_dur       = num_time_segments / total_time

const max_Gz   = 10e-3
const max_slew = 70.0
const rf_bound = 12e-6

const params = (γ = 2π*42.58e6, T1 = 1.0, T2 = 1.0, M0 = 1.0)

Random.seed!(1234)

const δ       = 4.0e-3
const zFOV    = 14.0e-3
const Nz      = 121
const z_base  = collect(range(-zFOV/2, zFOV/2; length=Nz))  # Base positions

const transition_frac = 0.00
const δ_core = δ*(1 - transition_frac)

const voxel_resolution = zFOV / (Nz - 1)  # Calculate voxel spacing
const position_offset_range = voxel_resolution / 2 

function generate_roi_masks(z_positions)
    in_roi   = (.-δ/2 .<= z_positions) .& (z_positions .<= δ/2)
    in_core  = (.-δ_core/2 .<= z_positions) .& (z_positions .<= δ_core/2)
    in_trans = in_roi .& .!in_core
    in_stop  = .!in_roi
    
    target_mag = fill(0.0, Nz)
    target_mag[in_core] .= 1.0
    target_mag[in_trans] .= NaN
    
    return in_roi, in_core, in_trans, in_stop, target_mag
end

# Generate base masks (unperturbed)
const in_roi_base, in_core_base, in_trans_base, in_stop_base, target_mag_base = generate_roi_masks(z_base)

@inline function rf_at(t::Real, I::AbstractVector, Q::AbstractVector)
    s  = clamp(t * inv_seg_dur, 0, num_time_segments - 1e-10)
    i1 = clamp(Int(floor(s)) + 1, 1, num_time_segments - 1)
    i2 = i1 + 1
    α  = s - (i1 - 1)
    bx = @inbounds I[i1]*(1-α) + I[i2]*α
    by = @inbounds Q[i1]*(1-α) + Q[i2]*α
    return bx, by
end

@inline function gz_at(t::Real; Gmax=max_Gz, slew=max_slew)
    rampT = min(Gmax/slew, total_time/2)
    if 2rampT == total_time
        Gpeak = slew * rampT
        return t <= rampT ? (t/rampT)*Gpeak :
               t >= total_time - rampT ? ((total_time - t)/rampT)*Gpeak : Gpeak
    elseif 2rampT < total_time
        if t < rampT
            return (t/rampT)*Gmax
        elseif t <= total_time - rampT
            return Gmax
        else
            return ((total_time - t)/rampT)*Gmax
        end
    else
        Gpeak = slew * rampT
        return t <= rampT ? (t/rampT)*Gpeak : ((total_time - t)/rampT)*Gpeak
    end
end

@inline function bloch_rhs(m::SVector{3,Ta}, Bx, By, Bz, p) where {Ta<:Real}
    T  = promote_type(Ta, typeof(Bx), typeof(By), typeof(Bz))
    mT = SVector{3}(T(m[1]), T(m[2]), T(m[3]))
    B  = SVector{3}(T(Bx),   T(By),   T(Bz))
    γT, T1T, T2T, M0T = T(p.γ), T(p.T1), T(p.T2), T(p.M0)
    return γT*cross(mT, B) + SVector{3}(-mT[1]/T2T, -mT[2]/T2T, -(mT[3]-M0T)/T1T)
end

function Rot(B)
    # Vector in R³
    θ = norm(B)
    nx, ny, nz = B ./ norm(B)
    # Skew-symmetric matrix in so(3)
    K = [  0 -nz ny;
          nz  0 -nx;
         -ny nx 0   ]
    # Rotation matrix in SO(3)
    expK = I + sin(θ) .* K + (1 .- cos(θ)) .* K^2
    return expK
end

@inline function bloch_step(dt, t, m::SVector{3,Ta}, I, Q, z, p) where {Ta<:Real}
    T  = promote_type(Ta, typeof(dt), typeof(z))
    mT = SVector{3}(T(m[1]), T(m[2]), T(m[3]))
    
    # Magnus(4) method
    # Evaluate B field at current time t
    bx_n, by_n = rf_at(t, I, Q)
    gz_n = gz_at(t)
    B_n = SVector{3}(T(bx_n), T(by_n), T(gz_n * z))
    
    # Evaluate B field at next time t+dt
    bx_n1, by_n1 = rf_at(t + dt, I, Q)
    gz_n1 = gz_at(t + dt)
    B_n1 = SVector{3}(T(bx_n1), T(by_n1), T(gz_n1 * z))
    
    # Magnus expansion terms
    A_n = -T(p.γ) * B_n
    A_n1 = -T(p.γ) * B_n1
    
    # Ω₁ = dt/2 * (A_n + A_{n+1})
    Ω1 = T(dt/2) * (A_n + A_n1)
    
    # Ω₂ = dt²/12 * (A_{n+1} × A_n)
    Ω2 = T(dt^2/12) * cross(A_n1, A_n)
    
    # Total rotation vector
    Ω = Ω1 + Ω2
    
    m_rot = Rot(Ω) * mT 
    
    # Apply relaxation terms (T1 and T2)
    relax = SVector{3}(-mT[1]/T(p.T2)*T(dt), -mT[2]/T(p.T2)*T(dt), -(mT[3]-T(p.M0))/T(p.T1)*T(dt))
    
    return m_rot + relax
end

# Pre-allocated cache for simulation
struct SimulationCache
    m::Vector{SVector{3,Float64}}
    Mx::Vector{Float64}
    My::Vector{Float64}
    Mz::Vector{Float64}
end

function SimulationCache(n::Int)
    SimulationCache(
        [SVector{3}(0.0, 0.0, params.M0) for _ in 1:n],
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n)
    )
end

# Original single-threaded version
function simulate_profile(I::AbstractVector, Q::AbstractVector; z_positions=z_base)
    T = promote_type(eltype(I), eltype(Q))
    Mx = Vector{T}(undef, Nz); My = similar(Mx); Mz = similar(Mx)
    @inbounds for iz in 1:Nz
        mm = SVector{3}(T(0), T(0), T(params.M0))
        t  = 0.0
        zz = z_positions[iz]  # Use provided positions
        for _ in 1:Nsteps
            mm = bloch_step(dt, t, mm, I, Q, T(zz), params)
            t += dt
        end
        Mx[iz] = mm[1]; My[iz] = mm[2]; Mz[iz] = mm[3]
    end
    return Mx, My, Mz
end

# Parallel version using Threads
function simulate_profile_parallel(I::AbstractVector, Q::AbstractVector; z_positions=z_base)
    T = promote_type(eltype(I), eltype(Q))
    Mx = Vector{T}(undef, Nz)
    My = Vector{T}(undef, Nz)
    Mz = Vector{T}(undef, Nz)
    
    Threads.@threads for iz in 1:Nz
        mm = SVector{3}(T(0), T(0), T(params.M0))
        t  = 0.0
        zz = z_positions[iz]
        for _ in 1:Nsteps
            mm = bloch_step(dt, t, mm, I, Q, T(zz), params)
            t += dt
        end
        Mx[iz] = mm[1]
        My[iz] = mm[2]
        Mz[iz] = mm[3]
    end
    return Mx, My, Mz
end

@inline function TV(I::AbstractVector, Q::AbstractVector)
    T = promote_type(eltype(I), eltype(Q))
    s = zero(real(T))  # Always return real value
    @inbounds for i in 2:length(I)
        dI = I[i] - I[i-1]
        dQ = Q[i] - Q[i-1]
        s += abs2(dI) + abs2(dQ)
    end
    return s
end

@inline function TV_profile(mag::AbstractVector)
    T = eltype(mag)
    s = zero(T)
    @inbounds for i in 2:length(mag)
        s += abs(mag[i] - mag[i-1])
    end
    return s
end

function init_iq(nseg::Int; Ttot=total_time, tbw=3.0, flip_deg=90.0, kind="gaussian")
    tgrid = collect(range(0, Ttot; length=nseg))
    tmid = Ttot/2

    if kind == "gaussian"
        σ = Ttot / (2.5 * tbw)
        s = exp.(-0.5 * ((tgrid .- tmid) ./ σ).^2)
        edge_fraction = 0.15
        n_edge = Int(ceil(nseg * edge_fraction))
    elseif kind == "sech"
        β = 2*tbw / Ttot
        s = sech.(β * (tgrid .- tmid))
        edge_fraction = 0.1
        n_edge = Int(ceil(nseg * edge_fraction))
        for i in 1:n_edge
            w = sin(π * i / (2 * n_edge))^2
            s[i] *= w
            s[end-i+1] *= w
        end
    elseif kind == "sinc"
        BW = tbw / Ttot
        arg = 2π*BW .* (tgrid .- tmid)
        s = [abs(a) < 1e-12 ? 1.0 : sin(a)/a for a in arg]
        α = 0.5
        w = ones(length(s))
        n_taper = Int(floor(α * length(s) / 2))
        for i in 1:n_taper
            w[i] = 0.5 * (1 + cos(π * (2*i/(α*length(s)) - 1)))
            w[end-i+1] = w[i]
        end
        s .= s .* w
    else
        error("Unknown kind: $kind")
    end

    θ = deg2rad(flip_deg)
    dt_seg = Ttot / (nseg - 1)
    area = sum(abs.(s)) * dt_seg
    scale = (θ / params.γ) / area

    I = clamp.(s .* scale, -rf_bound*0.8, rf_bound*0.8)
    Q = zeros(length(I))
    return I, Q
end

function plot_progress(I, Q, iter, loss_val; save_data=false)
    Mx, My, Mz = simulate_profile(I, Q; z_positions=z_base)
    mag = hypot.(Mx, My)

    if save_data
        phase = atan.(My, Mx)
        mag_core = mag[in_core_base]
        mag_stop = mag[in_stop_base]
        ripple = maximum(mag_core) - minimum(mag_core)
        stopmax = maximum(mag_stop)
    end

    t_rf = collect(range(0, total_time, length=length(I))) * 1e3  # ms

    p = plot(layout=(1,2), size=(900, 400))
    plot!(p[1], t_rf, I*1e6, label="Bx (I)", lw=2)
    plot!(p[1], t_rf, Q*1e6, label="By (Q)", lw=2, alpha=0.7)
    hline!(p[1], [rf_bound*1e6, -rf_bound*1e6], ls=:dot, alpha=0.5, label="")
    plot!(p[1], xlabel="Time (ms)", ylabel="RF (μT)",
          title="iter $iter | loss $(round(loss_val, sigdigits=4))", legend=:topright)

    plot!(p[2], z_base*1e3, mag, label="|Mxy|", lw=2)
    plot!(p[2], z_base*1e3, target_mag_base, label="Target", lw=2, ls=:dash)
    vspan!(p[2], [-δ*1e3/2, -δ_core*1e3/2], alpha=0.12, color=:yellow, label="")
    vspan!(p[2], [ δ_core*1e3/2,  δ*1e3/2], alpha=0.12, color=:yellow, label="")
    plot!(p[2], xlabel="z (mm)", ylabel="|Mxy|", title="Slice Profile", ylims=(-0.1, 1.2), legend=:topright)

    if save_data
        return p, (Mx=Mx, My=My, Mz=Mz, mag=mag, phase=phase,
                   ripple=ripple, stopband_max=stopmax)
    else
        return p
    end
end