using StaticArrays, LinearAlgebra, Random, Statistics
using Optim, LineSearches
using DifferentiationInterface
using Plots
import ForwardDiff

# Include base functions and constants (without execution code)
include("RF_Base_Functions.jl")

# L-BFGS optimization with advanced features
# tried some really high toelrances here to try and get it to run for longer
function optimize_lbfgs(x0::Vector{Float64};
    max_iters=500,
    g_tol=3f10,          
    f_reltol=3f10,       
    x_abstol=1e13,       
    m=10,              
    linesearch=BackTracking(order=2, iterations=20),
    plot_every=10,
    save_plots=true,
    use_perturbation=true,
    perturbation_schedule="cosine_decay",
    λ_profile=200,      
    λ_smooth=1e12,         
    λ_power=100,         
    λ_profile_tv=10,    
    λ_stopband=0.001,     
    adaptive_lambda=true,
    use_parallel=Threads.nthreads() > 1,
    verbose=true
)
    gr()
    
    # Storage for history
    loss_hist = Float64[]
    plots_array = Any[]
    metrics_history = Any[]
    lambda_history = []
    
    # Current lambda values (will be adapted if adaptive_lambda is true)
    current_lambdas = (
        profile = λ_profile,
        smooth = λ_smooth,
        power = λ_power,
        profile_tv = λ_profile_tv,
        stopband = λ_stopband
    )
    
    # Iteration counter for callbacks
    iter_count = 0
    best_profile_error = Inf
    best_x = copy(x0)
    
    # Create objective function with current parameters
    function create_objective(lambdas, perturb_scale)
        function obj(x::AbstractVector)
            T = eltype(x)
            n = length(x) ÷ 2
            I = @view x[1:n]
            Q = @view x[n+1:end]
            
            # Apply perturbation if requested
            if use_perturbation && perturb_scale > 0
                perturbations = (rand(Nz) .- 0.5) .* (2 * position_offset_range * perturb_scale)
                z_perturbed = z_base .+ perturbations
                in_roi, in_core, in_trans, in_stop, _ = generate_roi_masks(z_perturbed)
            else
                z_perturbed = z_base
                in_roi, in_core, in_trans, in_stop = in_roi_base, in_core_base, in_trans_base, in_stop_base
            end
            
            # Simulate (use parallel if available)
            sim_func = use_parallel ? simulate_profile_parallel : simulate_profile
            Mx, My, _ = sim_func(I, Q; z_positions=z_perturbed)
            
            loss = zero(T)
            
            # 1. Passband fidelity (want |Mxy| ≈ 1 in core)
            pass_error = zero(T)
            cnt_pass = 0
            @inbounds for i in 1:Nz
                if in_core[i]
                    # Target: Mx = 1, My = 0
                    pass_error += (Mx[i] - T(1))^2 + My[i]^2
                    cnt_pass += 1
                end
            end
            if cnt_pass > 0
                loss += T(lambdas.profile) * pass_error / T(cnt_pass)
            end
            
            # 2. Stopband suppression
            stop_error = zero(T)
            cnt_stop = 0
            @inbounds for i in 1:Nz
                if in_stop[i]
                    stop_error += Mx[i]^2 + My[i]^2
                    cnt_stop += 1
                end
            end
            if cnt_stop > 0
                loss += T(lambdas.stopband) * stop_error / T(cnt_stop)
            end
            
            # 3. RF smoothness (TV regularization)
            tv_val = TV(I, Q)
            loss += T(lambdas.smooth) * tv_val
            
            # 4. Profile smoothness
            mag = @. sqrt(Mx^2 + My^2)
            profile_tv = TV_profile(mag)
            loss += T(lambdas.profile_tv) * profile_tv
            
            # 5. Power constraint
            power = (sum(abs2, I) + sum(abs2, Q)) / T(2*length(I))
            loss += T(lambdas.power) * power
            
            return loss
        end
        return obj
    end
    
    # Callback function for monitoring
    function callback(tr)
        if isempty(tr)
            return false
        end
        
        state = tr[end]  # Get latest trace element
        iter_count += 1
        
        # Get current perturbation scale
        if perturbation_schedule == "linear_decay"
            perturb_scale = 1.0 - (iter_count - 1) / max_iters
        elseif perturbation_schedule == "cosine_decay"
            perturb_scale = 0.5 * (1 + cos(π * (iter_count - 1) / max_iters))
        elseif perturbation_schedule == "adaptive"
            # Start with full perturbation, reduce when progress slows
            if length(loss_hist) > 10
                recent_improvement = abs(loss_hist[end] - loss_hist[end-9]) / abs(loss_hist[end-9])
                perturb_scale = recent_improvement > 0.01 ? 1.0 : 0.5
            else
                perturb_scale = 1.0
            end
        else
            perturb_scale = 1.0
        end
        
        # Record loss
        push!(loss_hist, state.value)
        
        # Adaptive lambda adjustment
        if adaptive_lambda && iter_count % 20 == 0 && iter_count > 0 && haskey(state.metadata, "x")
            x_curr = state.metadata["x"]
            n = length(x_curr) ÷ 2
            I_curr = x_curr[1:n]
            Q_curr = x_curr[n+1:end]
            
            # Evaluate individual objectives
            Mx, My, _ = simulate_profile(I_curr, Q_curr; z_positions=z_base)
            mag = hypot.(Mx, My)
            
            # Profile error
            profile_error = mean((mag[in_core_base] .- 1.0).^2)
            
            # Adjust lambdas based on performance
            if profile_error < best_profile_error
                best_profile_error = profile_error
                best_x .= x_curr
                
                # If profile is good, we can afford more regularization
                if profile_error < 0.01
                    current_lambdas = (
                        profile = current_lambdas.profile,
                        smooth = current_lambdas.smooth * 1.2,
                        power = current_lambdas.power * 1.1,
                        profile_tv = current_lambdas.profile_tv,
                        stopband = current_lambdas.stopband
                    )
                end
            elseif profile_error > 0.1
                # Profile is bad, reduce regularization
                current_lambdas = (
                    profile = current_lambdas.profile * 1.5,
                    smooth = current_lambdas.smooth * 0.8,
                    power = current_lambdas.power * 0.9,
                    profile_tv = current_lambdas.profile_tv * 0.9,
                    stopband = current_lambdas.stopband * 1.2
                )
            end
            
            push!(lambda_history, current_lambdas)
        end
        
        # Plotting (use result.minimizer instead of metadata)
        if iter_count % plot_every == 0
            # For plotting, we'll use the current best x
            x_curr = best_x
            n = length(x_curr) ÷ 2
            I_curr = x_curr[1:n]
            Q_curr = x_curr[n+1:end]
            
            save_full = (iter_count % 50 == 0) || (iter_count >= max_iters)
            if save_full
                p, metrics = plot_progress(I_curr, Q_curr, iter_count, state.value; save_data=true)
                push!(metrics_history, (iter=iter_count, loss=state.value, metrics=metrics))
            else
                p = plot_progress(I_curr, Q_curr, iter_count, state.value; save_data=false)
            end
            
            display(p)
            
            if save_plots
                push!(plots_array, p)
                if length(plots_array) > 100
                    deleteat!(plots_array, 1:50)
                end
            end
            
            if verbose
                println("Iter $iter_count: loss=$(round(state.value, sigdigits=5)), |g|=$(round(state.g_norm, sigdigits=3))")
                
                # Check convergence criteria
                println("  Convergence checks:")
                println("    |g| < g_tol ($(g_tol))? $(state.g_norm < g_tol)")
                if iter_count > 1
                    rel_f_change = abs(state.value - loss_hist[end-1]) / max(abs(loss_hist[end-1]), 1e-10)
                    println("    rel_f_change < f_reltol ($(f_reltol))? $(rel_f_change < f_reltol) (rel_change = $(round(rel_f_change, sigdigits=3)))")
                end
                
                if adaptive_lambda && iter_count % 20 == 0
                    println("  λs: prof=$(round(current_lambdas.profile,digits=3)), smooth=$(round(current_lambdas.smooth,sigdigits=2)), power=$(round(current_lambdas.power,sigdigits=2))")
                end
            end
        end
        
        return false  # Don't stop
    end
    
    # Initial perturbation scale
    initial_perturb = perturbation_schedule == "constant" ? 1.0 : 1.0
    
    # Create initial objective
    obj_func = create_objective(current_lambdas, initial_perturb)
    
    # DEBUG: Check initial conditions
    val0 = obj_func(x0)
    backend = AutoForwardDiff()
    grad0 = gradient(obj_func, backend, x0)
    grad_norm0 = norm(grad0)
    
    # Set up optimization problem
    if verbose
        println("Starting L-BFGS optimization")
        println("  Max iterations: $max_iters")
        println("  Gradient tolerance: $g_tol")
        println("  Function tolerance: $f_reltol") 
        println("  Parameter tolerance: $x_abstol")
        println("  Line search: $(typeof(linesearch))")
        println("  L-BFGS history: $m")
        println("  Perturbation: $perturbation_schedule")
        println("  Adaptive λ: $adaptive_lambda")
        println("  Parallel: $use_parallel ($(Threads.nthreads()) threads)")
        println("  Initial λs: prof=$(λ_profile), smooth=$(λ_smooth), power=$(λ_power)")
        println("  DEBUG: Initial objective = $(round(val0, sigdigits=4))")
        println("  DEBUG: Initial |grad| = $(round(grad_norm0, sigdigits=3))")
        println("  DEBUG: |grad| > g_tol? $(grad_norm0 > g_tol)")
        println("-"^60)
    end
    
    # Run L-BFGS optimization
    result = optimize(
        obj_func,
        x0,
        LBFGS(
            m = m,
            alphaguess = LineSearches.InitialStatic(alpha=1.0),
            linesearch = linesearch
        ),
        Optim.Options(
            iterations = max_iters,
            g_tol = g_tol,
            f_reltol = f_reltol,
            x_abstol = x_abstol,
            show_trace = false,
            extended_trace = true,
            store_trace = true,
            callback = callback,
            allow_f_increases = true
        );
        autodiff = :forward
    )
    
    # Get final solution
    x_final = Optim.minimizer(result)
    
    # Check if best_x is better than final
    n = length(x_final) ÷ 2
    I_final = x_final[1:n]
    Q_final = x_final[n+1:end]
    
    Mx_final, My_final, _ = simulate_profile(I_final, Q_final; z_positions=z_base)
    mag_final = hypot.(Mx_final, My_final)
    final_profile_error = mean((mag_final[in_core_base] .- 1.0).^2)
    
    if best_profile_error < final_profile_error && !isempty(best_x)
        if verbose
            println("\nUsing best solution from iteration history")
            println("  Best profile error: $(round(best_profile_error, sigdigits=4))")
            println("  Final profile error: $(round(final_profile_error, sigdigits=4))")
        end
        x_final = best_x
    end
    
    # Print final statistics
    if verbose
        println("\n" * "="^60)
        println("OPTIMIZATION COMPLETE")
        println("  Iterations: $(Optim.iterations(result))")
        println("  Final loss: $(round(Optim.minimum(result), sigdigits=5))")
        println("  Converged: $(Optim.converged(result))")
        println("  |g|: $(round(Optim.g_residual(result), sigdigits=3))")
    end
    
    return x_final, loss_hist, plots_array, metrics_history, result
end


# Main execution
println("RF Pulse Optimization with L-BFGS")
println("="^70)

# Initialize
I0, Q0 = init_iq(num_time_segments; Ttot=total_time, tbw=3.0, flip_deg=90.0, kind="gaussian")
x0 = vcat(I0, Q0)

# Run L-BFGS optimization with working parameters
println("\nRunning L-BFGS optimization with debugged parameters...")
@time x_opt, loss_hist, plots_array, metrics_history, result = optimize_lbfgs(x0)

# Final analysis
n = length(x_opt) ÷ 2
I_opt = x_opt[1:n]
Q_opt = x_opt[n+1:end]

Mx, My, Mz = simulate_profile(I_opt, Q_opt; z_positions=z_base)
mag = hypot.(Mx, My)
phase = atan.(My, Mx)
mag_core = mag[in_core_base]
mag_stop = mag[in_stop_base]

println("\n" * "="^70)
println("FINAL PERFORMANCE METRICS (L-BFGS)")
println("="^70)
println("Optimization iterations: $(length(loss_hist))")
println("Final loss: $(round(loss_hist[end], sigdigits=5))")
println("Passband mean: $(round(mean(mag_core), digits=3))")
println("Passband ripple: $(round(maximum(mag_core) - minimum(mag_core), digits=3))")
println("Max stopband: $(round(maximum(mag_stop), digits=4))")
println("Mean stopband: $(round(mean(mag_stop), digits=4))")
println("Phase RMS (core): $(round(sqrt(mean(phase[in_core_base].^2)), digits=3)) rad")
println("Max RF amplitude: $(round(maximum(hypot.(I_opt, Q_opt))*1e6, digits=2)) μT")

# Create final plot
p_final = plot_progress(I_opt, Q_opt, length(loss_hist), loss_hist[end])
display(p_final)
savefig(p_final, "lbfgs_optimized_profile.png")

# Create animation if plots were saved
if !isempty(plots_array)
    println("\nCreating animation from $(length(plots_array)) frames...")
    anim = Animation()
    for p in plots_array
        frame(anim, p)
    end
    gif(anim, "lbfgs_optimization.gif", fps=5)
    println("Animation saved as 'lbfgs_optimization.gif'")
end

println("\n" * "="^70)
println("OPTIMIZATION COMPLETE")
println("Results saved:")
println("  - lbfgs_optimized_profile.png")
println("  - lbfgs_optimization.gif")