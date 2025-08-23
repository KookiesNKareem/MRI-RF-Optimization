using CUDA
using KernelAbstractions
using Adapt
using DifferentiationInterface
import Enzyme
import ChainRulesCore: rrule, NoTangent, ProjectTo
using LinearAlgebra: norm

backend = CPU()

println("Starting script...")
# backend = CUDABackend()
# CUDA.allowscalar(false)

include("miniKomaCore.jl") 

function excitation_caller!(M_xy, M_z, obj, seq, backend)
    nthreads = cld(length(M_xy), GROUP_SIZE) * GROUP_SIZE
    excitation_kernel!(backend, GROUP_SIZE)(
        M_xy, M_z,
        obj.p_x, obj.p_y, obj.p_z, obj.p_ΔBz, obj.p_T1, obj.p_T2, obj.p_ρ, UInt32(length(obj.p_x)),
        seq.s_Gx, seq.s_Gy, seq.s_Gz, seq.s_Δt, seq.s_f, seq.s_B1, UInt32(length(seq.s_Δt));
        ndrange = nthreads
    )
end

function f(M_xy, M_z, obj, seq, target, backend)

    # N = length(X) ÷ 2
    # Z = complex.(X[1:N], X[N+1:end])
    # mag_new = (; mag..., M_xy=X)
    println("Calling excitation_caller!")
    excitation_caller!(M_xy, M_z, obj, seq, backend)
    println("Finished excitation_caller!")
    loss = sum(abs.(M_xy .- target).^2)
    return loss
end

# Magnetization
Nspins = 100
# mag = (;
#     M_xy = zeros(Float32, 2*Nspins),
#     M_z  = zeros(Float32, Nspins)
# )
M_xy = zeros(Float32, 2*Nspins)
M_z  = zeros(Float32, Nspins)
# Phantom
T1 = 1.0f0; T2 = 0.5f0; M0 = 1.0f0
obj = (;
    p_x = zeros(Float32, Nspins),
    p_y = zeros(Float32, Nspins),
    p_z = zeros(Float32, Nspins),
    p_ΔBz = zeros(Float32, Nspins),
    p_T1 = fill(T1, Nspins),
    p_T2 = fill(T2, Nspins),
    p_ρ  = fill(M0, Nspins)
)
# Sequence
Nt = 100
dt = 1f-5
seq = (;
    s_Gx = zeros(Float32, Nt),
    s_Gy = zeros(Float32, Nt),
    s_Gz = zeros(Float32, Nt),
    s_Δt = fill(dt, Nt),
    s_f  = zeros(Float32, Nt),
    s_B1 = 1f-6 * ones(Float32, 2 * Nt)
)
# Target
target = rand(Float32, Nspins * 2)

# Move everything to GPU backend
# mag    = mag    |> adapt(backend)
M_xy = M_xy |> adapt(backend)
M_z  = M_z  |> adapt(backend)
obj    = obj    |> adapt(backend)
seq    = seq    |> adapt(backend)
target = target |> adapt(backend)

# X = cu(zeros(Float32, Nspins * 2))

# Gradient via Enzyme
enzyme_mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
ad_backend  = AutoEnzyme(; mode = enzyme_mode)

# println("Running kernel")
f(M_xy, M_z, obj, seq, target, backend)
@show M_xy
println("Running Enzyme AD...")
loss, ∇X = value_and_gradient(f, ad_backend, M_xy, Constant(M_z), Constant(obj), Constant(seq),
                              Constant(target), Constant(backend))

@info "loss" loss
@info "‖grad‖" norm(Array(∇X))