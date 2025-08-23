using StaticArrays, LinearAlgebra, Random, Statistics
using DifferentiationInterface
using Plots
import ForwardDiff

const total_time        = 4.0e-3
const dt                = 2.0e-7
const Nsteps            = Int(cld(total_time, dt))
const num_time_segments = 100
const inv_seg_dur       = num_time_segments / total_time

const max_Gz   = 10e-3
const max_slew = 70.0
const rf_bound = 12e-6

const params = (γ = 2π*42.58e6, T1 = 1.0, T2 = 1.0, M0 = 1.0)

Random.seed!(1234)

const δ       = 4.0e-3
const zFOV    = 14.0e-3
const Nz      = 121
const z       = collect(range(-zFOV/2, zFOV/2; length=Nz))

const transition_frac = 0.20
const δ_core = δ*(1 - transition_frac)

in_roi   = (.-δ/2 .<= z) .& (z .<= δ/2)
in_core  = (.-δ_core/2 .<= z) .& (z .<= δ_core/2)
in_trans = in_roi .& .!in_core
in_stop  = .!in_roi

target_mag = fill(0.0, Nz)
target_mag[in_core] .= 1.0
target_mag[in_trans] .= NaN

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

@inline function bloch_step(dt, m::SVector{3,Ta}, Bx, By, Bz, p) where {Ta<:Real}
    T  = promote_type(Ta, typeof(dt), typeof(Bx), typeof(By), typeof(Bz))
    mT = SVector{3}(T(m[1]), T(m[2]), T(m[3]))
    return mT + T(dt)*bloch_rhs(mT, T(Bx), T(By), T(Bz), p)
end

function simulate_profile(I::AbstractVector, Q::AbstractVector)
    T = promote_type(eltype(I), eltype(Q))
    Mx = Vector{T}(undef, Nz); My = similar(Mx); Mz = similar(Mx)
    @inbounds for iz in 1:Nz
        mm = SVector{3}(T(0), T(0), T(params.M0))
        t  = 0.0
        zz = z[iz]
        for _ in 1:Nsteps
            bx, by = rf_at(t, I, Q)
            gz     = gz_at(t)
            mm     = bloch_step(dt, mm, bx, by, T(gz*zz), params)
            t += dt
        end
        Mx[iz] = mm[1]; My[iz] = mm[2]; Mz[iz] = mm[3]
    end
    return Mx, My, Mz
end

@inline function tv_smooth(v::AbstractVector; ε=1e-8)
    T = eltype(v)
    s = zero(T)
    @inbounds for i in 2:length(v)
        diff = v[i] - v[i-1]
        s += sqrt(diff^2 + T(ε)^2)
    end
    return s
end

@inline function tv_iso(I::AbstractVector, Q::AbstractVector; ε=1e-8)
    T = promote_type(eltype(I), eltype(Q))
    s = zero(T)
    @inbounds for i in 2:length(I)
        dI = I[i] - I[i-1]
        dQ = Q[i] - Q[i-1]
        s += sqrt(dI^2 + dQ^2 + T(ε)^2)
    end
    return s
end

function objective(;
    λ_passband=15.0, λ_stopband=8.0, λ_transition=0.2,
    λ_ripple=20.0,
    λ_ctrl=500.0, λ_tail=10000.0, λ_tv=5000.0, λ_tv_iso=3000.0,
    λ_accel=5000.0, λ_jerk=2000.0, λ_q=1000.0,
    target_magnitude=1.0
)
    function obj(x::AbstractVector)
        T = eltype(x); n = length(x) ÷ 2
        I = @view x[1:n]; Q = @view x[n+1:end]
        Mx, My, _ = simulate_profile(I, Q)
        loss = zero(T)

        if λ_passband != 0
            num = zero(T); cnt = 0
            b = complex(T(target_magnitude), T(0))
            @inbounds for i in 1:Nz
                if in_core[i]
                    a = complex(Mx[i], My[i])
                    num += abs2(a - b)
                    cnt += 1
                end
            end
            if cnt > 0; loss += T(λ_passband) * num / T(cnt); end
        end

        if λ_stopband != 0
            num = zero(T); cnt = 0
            @inbounds for i in 1:Nz
                if in_stop[i]
                    a = complex(Mx[i], My[i])
                    num += abs2(a)
                    cnt += 1
                end
            end
            if cnt > 0; loss += T(λ_stopband) * num / T(cnt); end
        end

        if λ_transition != 0
            num = zero(T); cnt = 0
            b = complex(T(0.5), T(0))
            @inbounds for i in 1:Nz
                if in_trans[i]
                    a = complex(Mx[i], My[i])
                    num += abs2(a - b)
                    cnt += 1
                end
            end
            if cnt > 0; loss += T(λ_transition) * num / T(cnt); end
        end

        if λ_ripple != 0
            s1 = zero(T); s2 = zero(T); cnt = 0
            @inbounds for i in 1:Nz
                if in_core[i]
                    m = hypot(Mx[i], My[i])
                    s1 += m; s2 += m^2; cnt += 1
                end
            end
            if cnt > 0
                μ = s1 / T(cnt)
                var = s2 / T(cnt) - μ^2
                loss += T(λ_ripple) * var
            end
        end

        if λ_ctrl   != 0; loss += T(λ_ctrl)   * (sum(abs2, I) + sum(abs2, Q)) / T(length(I)); end
        if λ_q      != 0; loss += T(λ_q)      * sum(abs2, Q) / T(length(Q)); end
        if λ_tv     != 0; loss += T(λ_tv)     * (tv_smooth(I; ε=1e-10) + tv_smooth(Q; ε=1e-10)) / T(length(I)); end
        if λ_tv_iso != 0; loss += T(λ_tv_iso) *  tv_iso(I, Q; ε=1e-10) / T(length(I)); end

        if λ_accel != 0 && length(I) >= 3
            num = zero(T)
            @inbounds for i in 2:length(I)-1
                d2I = I[i+1] - 2*I[i] + I[i-1]
                d2Q = Q[i+1] - 2*Q[i] + Q[i-1]
                num += d2I^2 + d2Q^2
            end
            loss += T(λ_accel) * num / T(length(I))
        end
        if λ_jerk != 0 && length(I) >= 5
            num = zero(T)
            @inbounds for i in 3:length(I)-2
                d3I = I[i+2] - 3*I[i+1] + 3*I[i] - I[i-1]
                d3Q = Q[i+2] - 3*Q[i+1] + 3*Q[i] - Q[i-1]
                num += d3I^2 + d3Q^2
            end
            loss += T(λ_jerk) * num / T(length(I))
        end

        if λ_tail != 0
            n_tail = max(1, Int(ceil(n * 0.15)))
            num = zero(T)
            for i in 1:n_tail
                w = T((n_tail - i + 1) / n_tail)^2
                num += w * (I[i]^2 + Q[i]^2)
                num += w * (I[n-i+1]^2 + Q[n-i+1]^2)
            end
            loss += T(λ_tail) * num / T(2*n_tail)
        end

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
        for i in 1:n_edge
            w = sin(π * i / (2 * n_edge))^2
            s[i] *= w
            s[end-i+1] *= w
        end
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
    scale = (θ / params.γ) / area * 0.6

    I = clamp.(s .* scale, -rf_bound*0.8, rf_bound*0.8)
    Q = zeros(length(I))
    return I, Q
end

function plot_progress(I, Q, iter, loss_val; save_data=false, tag="")
    Mx, My, Mz = simulate_profile(I, Q)
    mag = hypot.(Mx, My)

    if save_data
        phase = atan.(My, Mx)
        mag_core = mag[in_core]
        mag_stop = mag[in_stop]
        ripple = maximum(mag_core) - minimum(mag_core)
        stopmax = maximum(mag_stop)
    end

    t_rf = collect(range(0, total_time, length=length(I))) * 1e3  # ms

    p = plot(layout=(1,2), size=(900, 400))
    plot!(p[1], t_rf, I*1e6, label="Bx (I)", lw=2)
    plot!(p[1], t_rf, Q*1e6, label="By (Q)", lw=2, alpha=0.7)
    hline!(p[1], [rf_bound*1e6, -rf_bound*1e6], ls=:dot, alpha=0.5, label="")
    plot!(p[1], xlabel="Time (ms)", ylabel="RF (μT)",
          title="$tag iter $iter | loss $(round(loss_val, sigdigits=4))", legend=:topright)

    plot!(p[2], z*1e3, mag, label="|Mxy|", lw=2)
    plot!(p[2], z*1e3, target_mag, label="Target", lw=2, ls=:dash)
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
    tag="",
    obj = objective()
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

    println("Starting $tag optimization")
    for t in 1:max_iters
        val, g = value_and_gradient(backend, obj, x)
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
                p, metrics = plot_progress(I_curr, Q_curr, t, val; save_data=true, tag=tag)
                push!(metrics_history, (iter=t, loss=val, metrics=metrics))
            else
                p = plot_progress(I_curr, Q_curr, t, val; save_data=false, tag=tag)
            end
            display(p)
            if save_plots
                push!(plots_array, p)
                if length(plots_array) > 400
                    deleteat!(plots_array, 1:200)
                end
            end
            println("[$tag] iter $t: loss=$(round(val,sigdigits=5)) |g|=$(round(gnorm,sigdigits=3)) lr=$(round(lr_t,sigdigits=3)) best=$(round(best_loss,sigdigits=5))@$best_iter patience=$patience_counter/$patience")
        end

        if patience_counter >= patience
            println("[$tag] Early stopping at iter $t (best @$best_iter, loss=$(round(best_loss,sigdigits=6)))")
            break
        end
        if t > 50 && gnorm < 1e-7
            println("[$tag] Converged at iter $t (loss=$(round(val,sigdigits=6)), |g|=$(round(gnorm,sigdigits=6)))")
            break
        end
    end

    if best_loss < loss_hist[end]
        println("[$tag] Returning best weights from iter $best_iter")
        return best_x, loss_hist, plots_array, metrics_history
    else
        return x, loss_hist, plots_array, metrics_history
    end
end

function run_optim(; max_iters=150, lr=1e-6, patience=40, plot_every=1)
    I0, Q0 = init_iq(num_time_segments; Ttot=total_time, tbw=3.0, flip_deg=90.0, kind="gaussian")
    x0 = vcat(I0, Q0)

    obj = objective(
        λ_passband = 15.0, λ_stopband = 8.0, λ_transition = 0.2, λ_ripple = 20.0,
        λ_ctrl = 500.0, λ_tail = 10000.0, λ_tv = 5000.0, λ_tv_iso = 3000.0,
        λ_accel = 5000.0, λ_jerk = 2000.0, λ_q = 1000.0,
        target_magnitude = 1.0
    )

    @time x_opt, L_hist, P_list, M_hist = optimize_adam(
        x0; max_iters=max_iters, lr=lr, lr_decay=0.998, patience=patience,
        min_delta=1e-6, rel_tol=1e-4, grad_clip=1e2, plot_every=plot_every,
        save_plots=true, tag="REAL", obj=obj
    )

    return (xopt=x_opt, loss_hist=L_hist, plots=P_list, metrics=M_hist)
end

best = run_optim(max_iters = 150, lr = 1e-6, patience = 40, plot_every = 1)

xopt = best.xopt
loss_hist = best.loss_hist
plots_array = best.plots
metrics_history = best.metrics

if !isempty(plots_array)
    println("\nCreating animation from $(length(plots_array)) frames...")
    anim = Animation()
    for p in plots_array
        display(p)
        frame(anim)
    end
    gif(anim, "rf_optimization_realistic.gif", fps=5)
    println("Animation saved as 'rf_optimization_realistic.gif'")
end

# Final analysis / plot
n   = length(xopt) ÷ 2
Iopt = xopt[1:n]
Qopt = xopt[n+1:end]

Mx, My, Mz = simulate_profile(Iopt, Qopt)
mag = hypot.(Mx, My)
phase = atan.(My, Mx)
mag_core = mag[in_core]
mag_stop = mag[in_stop]

println("FINAL PERFORMANCE METRICS")
println("Optimization iters: $(length(loss_hist))")
println("Final loss: $(round(loss_hist[end], sigdigits=5))")
println("Passband mean: $(round(mean(mag_core), digits=3))")
println("Passband ripple: $(round(maximum(mag_core) - minimum(mag_core), digits=3))")
println("Max stopband: $(round(maximum(mag_stop), digits=4))")
println("Mean stopband: $(round(mean(mag_stop), digits=4))")
println("Phase RMS (core): $(round(sqrt(mean((phase[in_core]).^2)), digits=3)) rad")
println("Max RF amplitude: $(round(maximum(hypot.(Iopt, Qopt))*1e6, digits=2)) μT")

p_final = plot_progress(Iopt, Qopt, length(loss_hist), loss_hist[end]; tag="FINAL REAL")
display(p_final)
savefig(p_final, "final_optimized_profile_realistic.png")