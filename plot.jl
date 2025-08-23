using Plots
default(framestyle=:box, grid=:y, legendfontsize=8, guidefontsize=11, tickfontsize=9)

# --- assumes you already defined these in your main code:
# rf_dur, rf_amp::Vector{<:Complex}, seq, z::Vector, M (from simulate_slice_profile)

# ===== 1) RF waveform (|B₁|, I, Q) =====
t_ms = range(0, rf_dur; length=length(rf_amp)) .* 1e3
BxI  = real.(rf_amp) .* 1e6                 # μT
ByQ  = imag.(rf_amp) .* 1e6                 # μT
Bmag = abs.(rf_amp) .* 1e6                  # μT

p_rf = plot(t_ms, Bmag; size=(600,360), lw=3, label="|B₁|",
    xlabel="Time (ms)", ylabel="RF amplitude (μT)", title="Optimized RF Waveform")
plot!(p_rf, t_ms, BxI; lw=1.6, ls=:solid, label="I")
plot!(p_rf, t_ms, ByQ; lw=1.6, ls=:dash,  label="Q")

# ===== 2) Slice profile (linear; simple & clear) =====
z_mm   = z .* 1e3
mag_xy = abs.(M.xy)

p_prof = plot(z_mm, mag_xy;
    size=(600,360), lw=3, label="|Mₓᵧ|",
    xlabel="Position (mm)", ylabel="|Mₓᵧ|", title="Slice Profile")
# Optional components
plot!(p_prof, z_mm, real.(M.xy); lw=1, ls=:dashdot, alpha=0.6, label="Mₓ")
plot!(p_prof, z_mm, imag.(M.xy); lw=1, ls=:dashdot, alpha=0.6, label="Mᵧ")
plot!(p_prof, z_mm, M.z;        lw=1, ls=:dot,     alpha=0.6, label="M_z")
ylims!(p_prof, 0, 1.1)

# ===== 3) Optional dB slice profile =====
show_db = false
p_prof_db = nothing
if show_db
    epsv = 1e-12
    mag_db = 20 .* log10.(mag_xy .+ epsv)
    p_prof_db = plot(z_mm, mag_db; size=(600,360), lw=2.5, label="|Mₓᵧ| (dB)",
        xlabel="Position (mm)", ylabel="Magnitude (dB)", title="Slice Profile (dB)")
    ylims!(p_prof_db, -80, 5)
end

# ===== 4) Try sequence sketch (some KomaMRI versions return `nothing`) =====
seq_plot = try
    plot_seq(seq)
catch
    nothing
end

# ===== 5) Compose safely with if statement =====
if seq_plot isa Plots.Plot
    if show_db
        display(plot(seq_plot, p_rf, p_prof, p_prof_db;
            layout=(1,4), size=(2100,420),
            left_margin=20, right_margin=12, bottom_margin=16, top_margin=10))
    else
        display(plot(seq_plot, p_rf, p_prof;
            layout=(1,3), size=(1700,420),
            left_margin=20, right_margin=12, bottom_margin=16, top_margin=10))
    end
else
    if show_db
        display(plot(p_rf, p_prof, p_prof_db;
            layout=(1,3), size=(1700,420),
            left_margin=20, right_margin=12, bottom_margin=16, top_margin=10))
    else
        display(plot(p_rf, p_prof;
            layout=(1,2), size=(1200,420),
            left_margin=20, right_margin=12, bottom_margin=16, top_margin=10))
    end
end

# savefig("koma_rf_profile_plots.png")