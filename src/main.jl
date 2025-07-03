using LinearAlgebra: cross
using DifferentiationInterface
using Enzyme

# General parameters
num_spins = 20
slice_thickness = 4e-3
M_ex = [0.0, 1.0, 0.0] # Excitation magnetization
M_eq = [0.0, 0.0, 1.0] # Equilibrium magnetization

# Simulation parameters
Gz = 10e-3
zs = range(-0.7e-2, 0.7e-2, num_spins)
params = map(zs) do z
    (
        γ = 2π * 42.58e6,
        T1 = 1.0,     
        T2 = 1.0, 
        M0 = 1.0, 
        x = zeros(num_time_segments), # This is mutable and will be optimized
        Bz = Gz * z,
        total_time = 3.0e-3,
        dt = 1e-7,
        Nsteps = Int(round(total_time / dt))
    )
end

# Relevant functions
include("src/simulation.jl")
include("src/optimization.jl")

# Optimize RF pulse
num_iters = 50
step_size = 0.001
backend = AutoEnzyme()
x = zeros(num_time_segments)
x, loss = gradient_descent(x, step_size, num_iters, objective)