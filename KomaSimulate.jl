using KomaMRICore
using Plots

## RF pulse parameters to match Solve B.jl
rf_amp = optimized_Bx .+ 1im * optimized_By  # Complex RF amplitude
rf_dur = total_time

# Create RF pulse object
rf90 = RF(rf_amp, rf_dur)

# Gradient parameters (matching Solve B.jl)
gz_amp = params.Gz  # 10 mT/m gradient strength
gz_dur = total_time
gz = Grad(gz_amp, gz_dur)

# Create sequence
seq = Sequence([Grad(0,0); Grad(0,0); gz;;], [rf90;;])

sim_params = KomaMRICore.default_sim_params()
sim_params["Δt_rf"] = 1e-7
# p = plot_seqd(seq; sampling_params=sim_params) # You need KomaMRIPlots
# display(p)

## Simulate slice profile over the same range as Solve B.jl
z = target_positions(num_time_segments * 10, dz / 10)  # Match z_positions range from RF optimization.jl

# Run simulation
M = simulate_slice_profile(seq; z, sim_params)

# ## Plot results using Plots.jl
plot(z*1e3, abs.(M.xy), 
     xlabel="Position (mm)", 
     ylabel="|Mxy|", 
     title="Optimized RF Slice Profile",
     label="Optimized RF |Mxy|",
     color=:blue)

# Also plot the individual components
plot!(z*1e3, real.(M.xy), label="Mx", lw=1, color=:red)
plot!(z*1e3, imag.(M.xy), label="My", lw=1, color=:green)
plot!(z*1e3, M.z, label="Mz", lw=1, color=:orange)
plot!(z_positions*1e3, zeros(num_points), marker=:circle, label="opt points")