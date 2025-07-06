using KomaMRI
using Plots

# RF pulse parameters to match Solve B.jl
rf_dur = total_time
rf_amp = optimized_Bx  # Use optimized values directly

# Create RF pulse object
rf90 = RF(rf_amp, rf_dur)

# Gradient parameters (matching Solve B.jl)
gz_amp = 10e-3  # 10 mT/m gradient strength
gz_dur = dur(rf90)
gz = Grad(gz_amp, gz_dur)

# Create sequence
seq = Sequence([Grad(0,0); Grad(0,0); gz;;], [rf90;;])
plot_seq(seq)
# Simulate slice profile over the same range as Solve B.jl
z = range(-0.7, 0.7, 300) * 1e-2 |> collect  # Match z_positions range from Solve B.jl

# Run simulation
M = simulate_slice_profile(seq; z)

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
