using KernelAbstractions
using CUDA
using DifferentiationInterface
using Enzyme

# x = CuArray([1.0f0, 2.0f0, 3.0f0])
# y = CuArray([2.0f0, 3.0f0, 4.0f0])
# x = CUDA.fill(1.0f0, 10000)
# y = CUDA.fill(2.0f0, 10000)

# @kernel function multiply!(x, y)
#     i = @index(Global)
#     y[i] = x[i] * y[i]
# end

params = (
    γ = Float32(42.58e6),
    B = Float32(1e-6),      
    T1 = 1.0f0,     
    T2 = 0.5f0,     
    M0 = 1.0f0,   
)

@inline function cross2(a, b)
    return (a[2] * b[3] - a[3] * b[2],
            a[3] * b[1] - a[1] * b[3],
            a[1] * b[2] - a[2] * b[1])
end

@kernel function bloch!(dm, m, @Const(params))
    γ, B, T1, T2, M0 = params
    i = @index(Global)
    m_i = @views(m[:, i])
    dm_i = @views(dm[:, i])
    B = (0.0f0, 0.0f0, B) 
    cross_term = γ .* cross2(m_i, B)
    relax_term = (m_i[1] / T2, m_i[2] / T2, (m_i[3] - M0) / T1)
    dM = cross_term .- relax_term
    dm_i .= dM
end

# multiply!(backend, 64)(x, y, ndrange=size(x))
# KernelAbstractions.synchronize(backend)
# all(y .== x .* 2.0f0)

function test!(output, input1, input2)
    output .= 2*input1 .+ 3*input2
    return nothing
end

output = [0.0f0]
∂output = [1.0f0]
input1 = [1.0f0]
∂input1 = [0.0f0]
input2 = [2.0f0]
∂input2 = [0.0f0]


# autodiff(Reverse, test!, Duplicated(output, ∂output), Duplicated(input1, ∂input1), Duplicated(input2, ∂input2))[1]
# @show output, ∂output, input1, ∂input1, input2, ∂input2
# bloch!(backend, 64)(dm, m, params, ndrange=size(m, 2))
# KernelAbstractions.synchronize(backend)
# println(dm)
m = cu(ones(3, 10))  # Initial magnetization vector
dm = cu(ones(3, 10))
∂dm = cu(ones(3, 10))
∂m = cu(zeros(3, 10))
backend = get_backend(m)
bloch_kernel = bloch!(backend, 10, size(m, 2))
autodiff(Reverse, bloch_kernel, Duplicated(dm, ∂dm), Duplicated(m, ∂m), Const(params))[1]
@show dm, ∂dm, m, ∂m