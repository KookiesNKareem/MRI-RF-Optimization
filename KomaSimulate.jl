using KomaMRI
using Plots

# RF pulse parameters from optimization
rf_dur = total_time  # 4.0e-3 seconds
rf_amp = xopt

# Create RF pulse object
rf90 = RF(rf_amp, rf_dur)

# Gradient parameters (matching optimization)
gz_amp = 10e-3  # 10 mT/m gradient strength
gz_dur = dur(rf90)
gz = Grad(gz_amp, gz_dur)

# Create sequence
seq = Sequence([Grad(0,0); Grad(0,0); gz;;], [rf90;;])
plot_seq(seq)

# Simulate slice profile over the same range as optimization
z = collect(range(-7e-3, 7e-3, 241))  # Match z range from optimization: -7mm to 7mm, 241 points

# Run simulation
M = simulate_slice_profile(seq; z=z)

# Plot results using Plots.jl
plot(z*1e3, abs.(M.xy), 
     xlabel="Position (mm)", 
     ylabel="|Mxy|", 
     title="Optimized RF Slice Profile",
     label="Optimized RF |Mxy|",
     lw=2, color=:blue)

# Also plot the individual components
plot!(z*1e3, real.(M.xy), label="Mx", lw=1, color=:red, alpha=0.7)
plot!(z*1e3, imag.(M.xy), label="My", lw=1, color=:green, alpha=0.7)
plot!(z*1e3, M.z, label="Mz", lw=1, color=:orange, alpha=0.7)

# Add target box for reference
δ = 4.0e-3  # Slice thickness
hline!([0, 1], ls=:dash, color=:gray, alpha=0.5, label="Target")
vline!([-δ/2*1e3, δ/2*1e3], ls=:dash, color=:gray, alpha=0.5, label=nothing)