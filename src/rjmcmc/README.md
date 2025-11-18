# Reversible Jump MCMC for Pigeons.jl

This directory contains the implementation of Reversible Jump Markov Chain Monte Carlo (RJMCMC) for Pigeons.jl, enabling transdimensional sampling across different model spaces.

## Status

**Phase 1 (Basic Infrastructure)**: ✅ Complete
- Core types and interfaces implemented
- Compatible with existing Pigeons architecture
- Basic test suite created

**Next Steps**:
- Variational reference integration
- Advanced jump kernels
- Comprehensive examples

## Files

- `RJState.jl`: Variable-dimensional state type with state interface implementation
- `RJLogPotential.jl`: Log potential over model space, target interface
- `JumpKernel.jl`: Jump kernel interface and basic implementations (BirthKernel, DeathKernel)
- `RJMCMCExplorer.jl`: Explorer for between-model moves
- `rjmcmc.jl`: Module file that includes all components

## Quick Start

```julia
using Pigeons
using Distributions

# Define models
model1_lp(params) = logpdf(Normal(0, 1), params[:x][1])
model2_lp(params) = logpdf(Normal(0, 1), params[:x][1]) + logpdf(Normal(0, 1), params[:theta][1])

# Specify model structure
model_specs = Dict(
    1 => (continuous = [:x], discrete = Symbol[]),
    2 => (continuous = [:x, :theta], discrete = Symbol[])
)

# Create RJMCMC target
target = RJLogPotential(
    models = Dict(1 => model1_lp, 2 => model2_lp),
    model_prior = Categorical([0.5, 0.5]),
    model_specs = model_specs
)

# Define jump kernels
birth, death = create_birth_death_pair(
    model_low = 1,
    model_high = 2,
    new_params = Dict(:theta => Normal(0, 1))
)

# Compose explorer
explorer = Compose(
    RJMCMCExplorer(jump_kernels = Dict(birth, death)),
    SliceSampler()
)

# Run
pt = pigeons(target = target, explorer = explorer)
```

## Design Documentation

See `docs/research/rjmcmc-design.md` for comprehensive design documentation including:
- Architecture overview
- Compatibility analysis with MPI/threading/variational references
- Implementation roadmap
- Theoretical considerations

## Testing

Run tests with:
```julia
using Pkg
Pkg.test("Pigeons", test_args=["rjmcmc"])
```

Or include the test file directly:
```julia
include("test/test_rjmcmc.jl")
```

## Key Features

1. **Natural MPI/Threading Compatibility**: Swap-by-chain-index design means variable dimensions work automatically in distributed settings

2. **SliceSampler Compatibility**: RJState implements array interface for coordinate-wise sampling

3. **Explorer Composition**: Mix and compose RJMCMC with any existing explorer (SliceSampler, MALA, AAPS, etc.)

4. **Clean Interfaces**: Follows Pigeons' informal interface pattern for extensibility

## Implementation Details

### State Management

`RJState` provides:
- Dynamic variable lists based on current model
- Efficient copy semantics
- Array-like indexing for SliceSampler
- Full state interface implementation

### Jump Kernels

Current implementations:
- `BirthKernel`: Add parameters (increase dimension)
- `DeathKernel`: Remove parameters (decrease dimension)
- `create_birth_death_pair`: Utility for reversible pairs

Future kernels:
- `SplitKernel`: Split one parameter into two
- `MergeKernel`: Merge two parameters into one
- `VariationalBirthKernel`: Use learned proposals

### Explorer

`RJMCMCExplorer`:
- Proposes model jumps
- Computes Metropolis-Hastings acceptance
- Records jump statistics
- Composes with within-model explorers

## Future Work

### Phase 2: Variational Integration
- `RJVariationalReference`: Per-model Gaussian references
- Automatic proposal learning
- Stabilized two-leg RJMCMC-PT

### Phase 3: Advanced Features
- Post-processing utilities (`posterior_model_probabilities`, `bayes_factors`)
- Specialized recorders for model visits
- Performance optimizations

### Phase 4: Applications
- Change-point detection example
- Variable selection example
- Mixture model components example

## References

- Green (1995): "Reversible jump Markov chain Monte Carlo computation and Bayesian model determination"
- Syed et al. (2021): "Non-reversible parallel tempering: a scalable highly parallel MCMC scheme"
- Design document: `docs/research/rjmcmc-design.md`
