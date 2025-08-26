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

function objective(;
    λ_profile=10.0,      # profile fidelity
    λ_smooth=1e12,     # TV on RF
    λ_power=100.0,       # power constraint
    λ_profile_tv=10,    # TV on profile
    target_magnitude=1.0,
    use_perturbation=true,
    perturbation_scale=1.0,
    use_parallel=true
)
    function obj(x::AbstractVector)
        T = eltype(x); n = length(x) ÷ 2
        I = @view x[1:n]; Q = @view x[n+1:end]
        
        if use_perturbation
            perturbations = (rand(Nz) .- 0.5) .* (2 * position_offset_range * perturbation_scale)
            z_perturbed = z_base .+ perturbations
            in_roi, in_core, in_trans, in_stop, _ = generate_roi_masks(z_perturbed)
        else
            z_perturbed = z_base
            in_roi, in_core, in_trans, in_stop = in_roi_base, in_core_base, in_trans_base, in_stop_base
        end
        
        sim_func = use_parallel && Threads.nthreads() > 1 ? simulate_profile_parallel : simulate_profile
        Mx, My, _ = sim_func(I, Q; z_positions=z_perturbed)
        mag = Mx.^2 .+ My.^2
        loss = zero(T)

        # profile
        num_pass = zero(T); cnt_pass = 0
        @inbounds for i in 1:Nz
            if in_core[i]
                m = (Mx[i] - target_magnitude)^2 +  My[i]^2
                num_pass += m
                cnt_pass += 1
            end
        end
        if cnt_pass > 0
            loss += T(λ_profile) * num_pass / T(cnt_pass)
        end
        # stopband
        num_stop = zero(T); cnt_stop = 0
        @inbounds for i in 1:Nz
            if in_stop[i] 
                m = Mx[i]^2 +  My[i]^2
                num_stop += m
                cnt_stop += 1
            end
        end
        if cnt_stop > 0
            loss += T(λ_profile) * 2.0 * num_stop / T(cnt_stop)
        end

        tv_val = TV(I, Q)
        tv_contrib = λ_smooth * tv_val
        loss += tv_contrib
        
        profile_tv = TV_profile(mag)
        loss += T(λ_profile_tv) * profile_tv

        power = (sum(abs2, I) + sum(abs2, Q)) / T(2*length(I))
        loss += T(λ_power) * power

        return loss
    end
    return obj
end

function init_iq(nseg::Int; Ttot=total_time, tbw=3.0, flip_deg=90.0, kind="gaussian")
    tgrid = collect(range(0, Ttot; length=nseg))
    tmid = Ttot/2

    if kind == "gaussian"
        σ = Ttot / (2.5 * tbw)
        s = exp.(-0.5 * ((tgrid .- tmid) ./ σ).^2)
        edge_fraction = 0.15
        n_edge = Int(ceil(nseg * edge_fraction))
        # for i in 1:n_edge
        #     w = sin(π * i / (2 * n_edge))^2
        #     s[i] *= w
        #     s[end-i+1] *= w
        # end
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

function optimize_adam(x0::Vector{Float64};
    max_iters=200,
    lr=5e-6,
    lr_decay=0.995,
    β1=0.9, β2=0.999, ϵ=1e-8,
    grad_clip=5e2,
    patience=50,
    min_delta=1e-6,
    rel_tol=1e-4,
    plot_every=1,
    save_plots=true,
    perturbation_schedule = "constant",  # "constant", "linear_decay", or "cosine_decay"
    use_parallel = Threads.nthreads() > 1,  # Auto-detect if parallel available
    obj_params = (λ_profile=200.0, λ_smooth=1e10, λ_power=10000.0, λ_profile_tv=5)  # Configurable params
)
    gr()

    backend = AutoForwardDiff()
    x = copy(x0)
    m = zero(x)
    v = zero(x)

    best_loss = Inf
    best_x = copy(x)
    best_iter = 0
    patience_counter = 0

    loss_hist = Float64[]
    plots_array = Any[]
    metrics_history = Any[]

    println("Voxel resolution: $(round(voxel_resolution*1e3, digits=3)) mm")
    println("Offset range: ±$(round(position_offset_range*1e3, digits=3)) mm (uniform distribution)")
    println("Using $(use_parallel ? "parallel" : "serial") simulation with $(Threads.nthreads()) threads")
    
    for t in 1:max_iters
        if perturbation_schedule == "linear_decay"
            perturb_scale = 1.0 - (t-1) / max_iters
        elseif perturbation_schedule == "cosine_decay"
            perturb_scale = 0.5 * (1 + cos(π * (t-1) / max_iters))
        else
            perturb_scale = 1.0
        end
        
        obj_perturbed = objective(
            λ_profile = obj_params.λ_profile,
            λ_smooth = obj_params.λ_smooth,
            λ_power = obj_params.λ_power,
            λ_profile_tv = obj_params.λ_profile_tv,
            target_magnitude = 1.0,
            use_perturbation = true,
            perturbation_scale = perturb_scale,
            use_parallel = use_parallel
        )
        
        val, g = value_and_gradient(obj_perturbed, backend, x)
        push!(loss_hist, val)

        gnorm = norm(g)
        if gnorm > grad_clip
            g .= g .* (grad_clip / gnorm)
        end

        m .= β1 .* m .+ (1 .- β1) .* g
        v .= β2 .* v .+ (1 .- β2) .* (g .^ 2)
        mhat = m ./ (1 .- β1^t)
        vhat = v ./ (1 .- β2^t)

        lr_t = lr * (lr_decay^t)
        x .-= lr_t .* (mhat ./ (sqrt.(vhat) .+ ϵ))
        x .= clamp.(x, -rf_bound, rf_bound)

        if isinf(best_loss)
            best_loss = val; best_x .= x; best_iter = t; patience_counter = 0
        else
            improvement = best_loss - val
            rel_impr = abs(improvement / max(abs(best_loss), 1e-10))
            if improvement > min_delta || rel_impr > rel_tol
                best_loss = val; best_x .= x; best_iter = t; patience_counter = 0
            else
                patience_counter += 1
            end
        end

        if t % plot_every == 0
            n = length(x) ÷ 2
            I_curr = x[1:n]; Q_curr = x[n+1:end]
            save_full = (t % 10 == 0) || (t == max_iters)
            if save_full
                p, metrics = plot_progress(I_curr, Q_curr, t, val; save_data=true)
                push!(metrics_history, (iter=t, loss=val, metrics=metrics))
            else
                p = plot_progress(I_curr, Q_curr, t, val; save_data=false)
            end
            display(p)
            if save_plots
                push!(plots_array, p)
                if length(plots_array) > 400
                    deleteat!(plots_array, 1:200)
                end
            end
            println("iter $t: loss=$(round(val,sigdigits=5)) |g|=$(round(gnorm,sigdigits=3)) lr=$(round(lr_t,sigdigits=3)) perturb_scale=$(round(perturb_scale,digits=3)) best=$(round(best_loss,sigdigits=5))@$best_iter patience=$patience_counter/$patience")
        end

        if patience_counter >= patience
            println("Early stopping at iter $t (best @$best_iter, loss=$(round(best_loss,sigdigits=6)))")
            break
        end
        if t > 50 && gnorm < 1e-7
            println("Converged at iter $t (loss=$(round(val,sigdigits=6)), |g|=$(round(gnorm,sigdigits=6)))")
            break
        end
    end

    if best_loss < loss_hist[end]
        println("Returning best weights from iter $best_iter")
        return best_x, loss_hist, plots_array, metrics_history
    else
        return x, loss_hist, plots_array, metrics_history
    end
end

function run_optim(; max_iters=150, lr=1e-6, patience=40, plot_every=1, perturbation_schedule="constant")
    I0, Q0 = init_iq(num_time_segments; Ttot=total_time, tbw=3.0, flip_deg=90.0, kind="gaussian")
    x0 = vcat(I0, Q0)

    @time x_opt, L_hist, P_list, M_hist = optimize_adam(
        x0; max_iters=max_iters, lr=lr, lr_decay=0.998, patience=patience,
        min_delta=1e-6, rel_tol=1e-4, grad_clip=1e2, plot_every=plot_every,
        save_plots=true, perturbation_schedule=perturbation_schedule
    )

    return (xopt=x_opt, loss_hist=L_hist, plots=P_list, metrics=M_hist)
end

println("Running optimization...")
best = run_optim(max_iters=300, lr=2e-6, patience=40, plot_every=1, perturbation_schedule="constant")

xopt = best.xopt
loss_hist = best.loss_hist
plots_array = best.plots
metrics_history = best.metrics

##
if !isempty(plots_array)
    println("\nCreating animation from $(length(plots_array)) frames...")
    anim = Animation()
    for p in plots_array
        frame(anim, p)             
    end
    gif(anim, "rf_optimization.gif", fps=5)
    println("Animation saved as 'rf_optimization.gif'")
end

##
n   = length(xopt) ÷ 2
Iopt = xopt[1:n]
Qopt = xopt[n+1:end]

Mx, My, Mz = simulate_profile(Iopt, Qopt; z_positions=z_base)
mag = hypot.(Mx, My)
phase = atan.(My, Mx)
mag_core = mag[in_core_base]
mag_stop = mag[in_stop_base]

println("\nFINAL PERFORMANCE METRICS")
println("Optimization iters: $(length(loss_hist))")
println("Final loss: $(round(loss_hist[end], sigdigits=5))")
println("Passband mean: $(round(mean(mag_core), digits=3))")
println("Passband ripple: $(round(maximum(mag_core) - minimum(mag_core), digits=3))")
println("Max stopband: $(round(maximum(mag_stop), digits=4))")
println("Mean stopband: $(round(mean(mag_stop), digits=4))")
println("Phase RMS (core): $(round(sqrt(mean((phase[in_core_base]).^2)), digits=3)) rad")
println("Max RF amplitude: $(round(maximum(hypot.(Iopt, Qopt))*1e6, digits=2)) μT")

p_final = plot_progress(Iopt, Qopt, length(loss_hist), loss_hist[end])
display(p_final)
savefig(p_final, "final_optimized_profilet.png")