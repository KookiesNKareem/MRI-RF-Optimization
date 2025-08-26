using KomaMRI
using Plots

# RF pulse parameters from optimization
rf_dur = total_time  # 4.0e-3 seconds
rf_amp = xopt[1:end÷2] .+ im .* xopt[end÷2+1:end]  # Complex RF waveform

# Create RF pulse object
rf90 = RF(rf_amp, rf_dur)

# Gradient parameters (matching optimization)
gz_dur = dur(rf90)
gz = Grad(max_Gz, gz_dur)

# Create sequence
seq = Sequence([Grad(0,0); Grad(0,0); gz;;], [rf90;;])
plot_seq(seq)

# Simulate slice profile over the same range as optimization
z = collect(range(-zFOV/2, zFOV/2, 241))  # Match z range from optimization: -7mm to 7mm, 241 points

# Run simulation
M = simulate_slice_profile(seq; z=z)

# Get the unperturbed simulation results from RF optimization
n = length(xopt) ÷ 2
Iopt = xopt[1:n]
Qopt = xopt[n+1:end]

# Simulate on the unperturbed target points from optimization
Mx_sim, My_sim, Mz_sim = simulate_profile(Iopt, Qopt; z_positions=z_base)
mag_sim = hypot.(Mx_sim, My_sim)

# Create the plot with both Koma simulation and target points
p = plot(size=(800, 500), title="Koma Simulation vs RF Optimization Results and Target Profile")

# Plot Koma simulation results
plot!(p, z*1e3, abs.(M.xy), 
     xlabel="Position (mm)", 
     ylabel="|Mxy|", 
     label="Koma Simulation",
     lw=2, color=:blue)

# Plot the RF optimization simulation results (connected with line)
plot!(p, z_base*1e3, mag_sim, 
     label="RF Optimization Simulation", 
     color=:red, lw=2, ls=:dash, marker=:circle, markersize=2, markerstrokewidth=0, alpha=0.8)

# Plot the actual TARGET PROFILE (the box function that was optimized for)
# The target is 1.0 in core, 0.0 in stopband, and NaN (unspecified) in transition
z_target_plot = Float64[]
mag_target_plot = Float64[]

# Add core region points (target = 1.0)
for i in 1:length(z_base)
    if in_core_base[i]
        push!(z_target_plot, z_base[i]*1e3)
        push!(mag_target_plot, 1.0)
    elseif in_stop_base[i]
        push!(z_target_plot, z_base[i]*1e3)
        push!(mag_target_plot, 0.0)
    end
end

# Sort for proper plotting
idx = sortperm(z_target_plot)
z_target_plot = z_target_plot[idx]
mag_target_plot = mag_target_plot[idx]

# Plot the target profile
scatter!(p, z_target_plot, mag_target_plot, 
     label="Target Profile", 
     color=:black, marker=:dot, markersize=1.25, markerstrokewidth=0, alpha=0.9)

# Highlight the slice regions
δ_core = δ*(1 - transition_frac)  # Core region

# Core region (should be 1.0)
vspan!(p, [-δ_core*1e3/2, δ_core*1e3/2], alpha=0.1, color=:green, label="Core")

# Transition regions
vspan!(p, [-δ*1e3/2, -δ_core*1e3/2], alpha=0.1, color=:yellow, label="Transition")
vspan!(p, [δ_core*1e3/2, δ*1e3/2], alpha=0.1, color=:yellow, label="")

# Add reference lines
hline!(p, [0, 1], ls=:dash, color=:gray, alpha=0.3, label="")

# Set y-axis limits for better visibility
ylims!(p, (-0.1, 1.2))

# Add legend
plot!(p, legend=:topright)

display(p)

# Print performance comparison
println("\n=== Performance Comparison ===")
println("RF Optimization Simulation (unperturbed):")
println("  Passband mean: $(round(mean(mag_sim[in_core_base]), digits=3))")
println("  Passband ripple: $(round(maximum(mag_sim[in_core_base]) - minimum(mag_sim[in_core_base]), digits=3))")
println("  Max stopband: $(round(maximum(mag_sim[in_stop_base]), digits=4))")
println("\nTarget Profile:")
println("  Core region: 1.0")
println("  Stopband region: 0.0")
println("  Transition region: unspecified (free to vary)")

# Also plot the individual components for detailed analysis
p2 = plot(size=(800, 500), title="Magnetization Components Comparison")

# Koma components (Mx and My only)
plot!(p2, z*1e3, real.(M.xy), label="Koma Mx", lw=1.5, color=:red, alpha=0.7)
plot!(p2, z*1e3, imag.(M.xy), label="Koma My", lw=1.5, color=:green, alpha=0.7)

# RF optimization simulation components (Mx and My only)
plot!(p2, z_base*1e3, Mx_sim, label="RF Opt Mx", color=:darkred, lw=1.5, ls=:dash, marker=:circle, markersize=1.5, markerstrokewidth=0, alpha=0.8)
plot!(p2, z_base*1e3, My_sim, label="RF Opt My", color=:darkgreen, lw=1.5, ls=:dash, marker=:circle, markersize=1.5, markerstrokewidth=0, alpha=0.8)

# Add reference regions
vspan!(p2, [-δ_core*1e3/2, δ_core*1e3/2], alpha=0.05, color=:green, label="")
vspan!(p2, [-δ*1e3/2, -δ_core*1e3/2], alpha=0.05, color=:yellow, label="")
vspan!(p2, [δ_core*1e3/2, δ*1e3/2], alpha=0.05, color=:yellow, label="")

plot!(p2, xlabel="Position (mm)", ylabel="Magnetization", legend=:topright)
ylims!(p2, (-1.1, 1.1))

display(p2)