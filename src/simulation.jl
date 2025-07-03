# RF pulse
t_nodes = range(0.0, total_time, num_time_segments)

function solve(m0, params)
    m = m0
    for i in 1:Nsteps
        t = (i - 1) * dt
        m = step(dt, m, t, params)
    end
    return m
end

function step(dt, m, t, params)
    dm = bloch(m, t, params)
    return m + dm * dt
end

function bloch(m, t, params)
    γ, T1, T2, M0, Bz, x = params

    # Magnetic field components
    Bx = ConstantInterpolation(x, t_nodes)(t) # function from DataInterpolations.jl
    B_field = [Bx, 0, Bz]

    # Cross product and relaxation terms
    cross_term = γ * cross(m, B_field)
    relaxation_term = [-m[1] / T2, -m[2] / T2, -(m[3] - M0) / T1]
    
    return cross_term + relaxation_term
end