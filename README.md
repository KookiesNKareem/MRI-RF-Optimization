# MRI RF Pulse Optimization

A Julia implementation for optimizing RF pulses in MRI using automatic differentiation and Bloch equation simulation.

## Overview

This project optimizes RF (radiofrequency) pulse shapes for selective excitation in MRI. It uses:
- Bloch equation simulation for magnetization dynamics
- Automatic differentiation with Enzyme.jl for gradient computation
- Vectorized operations with StaticArrays for performance
- Gradient descent optimization with early stopping

## Features

- **Vectorized Bloch Equation Solver**: Fast simulation of magnetization evolution
- **Automatic Differentiation**: Efficient gradient computation using Enzyme.jl
- **Selective Excitation**: Optimizes for slice-selective RF pulses
- **Performance Optimized**: Uses StaticArrays and pre-computed values
- **Penalty Functions**: Includes penalties for unwanted magnetization components

## Usage

Run the optimization:
```julia
julia "RF Optimization.jl"
```

## Parameters

- `num_points`: Number of spatial points (default: 20)
- `total_time`: RF pulse duration (default: 1.0e-3 s)
- `dt`: Time step for simulation (default: 1e-7 s)
- `num_time_segments`: Number of RF segments to optimize (default: 30)

## Output

The optimization produces:
1. Optimized Bx field values over time
2. Convergence plots
3. Final vs target magnetization comparison
4. RF pulse profile visualization

## Dependencies

- StaticArrays.jl
- LinearAlgebra.jl
- DifferentiationInterface.jl
- Enzyme.jl
- Plots.jl
- Random.jl

## License

MIT License
