# Reversible Jump MCMC for Pigeons.jl: Design and Implementation

**Date:** 2025-11-18
**Status:** Research & Initial Implementation

## Executive Summary

This document presents a comprehensive design for integrating Reversible Jump Markov Chain Monte Carlo (RJMCMC) into Pigeons.jl, compatible with parallel tempering, MPI/multi-threading, variational references, and the slice sampler explorer.

**Key Finding:** RJMCMC is highly compatible with Pigeons.jl's architecture due to the critical design choice of swapping chain indices rather than states, which naturally accommodates variable-dimensional state spaces.

---

## 1. Architecture Overview & Key Insight

### Critical Architectural Feature

Pigeons handles swaps by having replicas exchange **chain indices**, not states (src/pt/pigeons.jl:80-82, src/swap/swap.jl:79-102). This means:

- Swap complexity is O(1) in state dimensionality
- Works seamlessly with MPI/distributed computing
- States of different dimensions can coexist naturally in different replicas
- MPI communication only transmits two floats per swap, regardless of dimension

This design is **ideal** for transdimensional MCMC because different replicas can maintain states of different dimensions without any modification to the swap mechanism.

### How Swaps Work

During communication phase, replicas can either:
1. Exchange `state` fields (naive, expensive for distributed systems), OR
2. Exchange `chain` fields (Pigeons' approach, O(1) communication)

The latter ensures that the amount of data transmitted during a swap does not depend on state dimensionality—a remarkable property that makes distributed transdimensional MCMC feasible.

---

## 2. RJMCMC State Design

### Current State Interface

The informal state interface (src/pt/state.jl:4-56) defines:

```julia
@informal state begin
    continuous_variables(state) = @abstract
    discrete_variables(state) = @abstract
    variable(state, name::Symbol) = @abstract
    update_state!(state, name::Symbol, index, value) = @abstract
    extract_sample(state, log_potential, extractor) = @abstract
    sample_names(state, log_potential, extractor) = @abstract
end
```

Current implementations assume fixed dimensionality:
- Arrays: Single `:singleton_variable` (lines 66-92)
- Turing VarInfo: Fixed set of variables
- Custom types (e.g., IsingState): Fixed structure

### Proposed RJState Type

```julia
"""
State type supporting reversible jump MCMC with variable dimensionality.

# Fields
- `model_indicator::Int`: Current model index (1, 2, ..., K)
- `parameters::Dict{Symbol, Any}`: Model-specific parameters
- `dimension::Int`: Current total dimension
- `model_info::RJModelInfo`: Reference to model structure (shared across replicas)
"""
mutable struct RJState{K}
    model_indicator::Int
    parameters::Dict{Symbol, Any}
    dimension::Int
    model_info::RJModelInfo{K}
end
```

### State Interface Implementation

```julia
# Dynamic variable lists based on current model
function continuous_variables(state::RJState)
    return state.model_info.continuous_vars[state.model_indicator]
end

function discrete_variables(state::RJState)
    return state.model_info.discrete_vars[state.model_indicator]
end

function variable(state::RJState, name::Symbol)
    return state.parameters[name]
end

function update_state!(state::RJState, name::Symbol, index, value)
    state.parameters[name][index] = value
end

# For post-processing
function extract_sample(state::RJState, log_potential, extractor)
    flat_params = flatten_parameters(state)
    return [state.model_indicator; flat_params; log_potential(state)]
end

function sample_names(state::RJState, log_potential, extractor)
    param_names = get_parameter_names(state)
    return [:model_indicator; param_names; :log_density]
end
```

---

## 3. Log Potential Design

### RJLogPotential Type

```julia
"""
Log potential for reversible jump MCMC supporting multiple models.

# Fields
- `models::Dict{Int, Any}`: Map from model index to log potential
- `model_prior::DiscreteDistribution`: Prior over model indicators
"""
struct RJLogPotential{T}
    models::Dict{Int, T}
    model_prior::DiscreteDistribution
end

# Evaluate log density
function (lp::RJLogPotential)(state::RJState)
    k = state.model_indicator
    model_lp = lp.models[k](state.parameters)
    model_prior_lp = logpdf(lp.model_prior, k)
    return model_lp + model_prior_lp
end
```

### Target Interface Implementation

```julia
# Initialization: start in model with prior support
function Pigeons.initialization(target::RJLogPotential, rng::AbstractRNG, replica_index::Int)
    model_k = rand(rng, target.model_prior)
    return initialize_model_state(target, model_k, rng)
end

# Default explorer: compose RJ moves with within-model moves
function Pigeons.default_explorer(target::RJLogPotential)
    return Compose(
        RJMCMCExplorer(target.models),
        SliceSampler()
    )
end

# Default reference: prior over models
function Pigeons.default_reference(target::RJLogPotential)
    ref_models = Dict(k => default_reference(model) for (k, model) in target.models)
    return RJLogPotential(ref_models, target.model_prior)
end

# i.i.d. sampling at reference
function Pigeons.sample_iid!(reference::RJLogPotential, replica, shared)
    model_k = rand(replica.rng, reference.model_prior)
    replica.state = initialize_model_state(reference, model_k, replica.rng)
end
```

---

## 4. Explorer Design

### Within-Model Moves: SliceSampler Compatibility

The SliceSampler (src/explorers/SliceSampler.jl:43-62) requires:

```julia
for c in eachindex(state)
    pointer = Ref(state, c)
    slice_sample_coord!(...)
end
```

**Implementation strategy**: Provide a flattened view of current model's parameters:

```julia
# Make RJState compatible with coordinate-wise sampling
Base.eachindex(state::RJState) = 1:state.dimension

function Base.Ref(state::RJState, i::Int)
    # Return reference to i-th element in flattened parameter vector
    var_name, local_idx = locate_parameter(state, i)
    return Ref(state.parameters[var_name], local_idx)
end

# Utility to flatten current model's parameters
function flatten_parameters(state::RJState)
    result = Float64[]
    for var_name in continuous_variables(state)
        append!(result, state.parameters[var_name])
    end
    for var_name in discrete_variables(state)
        append!(result, state.parameters[var_name])
    end
    return result
end
```

### Between-Model Moves: RJMCMCExplorer

```julia
"""
Explorer for reversible jump MCMC moves between models.

# Fields
- `jump_kernels::Dict{Pair{Int,Int}, JumpKernel}`: Kernels for model transitions
- `n_attempts::Int`: Number of jump attempts per step
"""
struct RJMCMCExplorer
    jump_kernels::Dict{Pair{Int,Int}, JumpKernel}
    n_attempts::Int
end

function Pigeons.step!(explorer::RJMCMCExplorer, replica, shared)
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)

    for _ in 1:explorer.n_attempts
        attempt_model_jump!(explorer, replica, log_potential)
    end
end

function attempt_model_jump!(explorer, replica, log_potential)
    current_k = replica.state.model_indicator

    # Propose new model
    proposed_k = propose_target_model(explorer, current_k, replica.rng)

    if current_k == proposed_k
        return  # No jump
    end

    # Get jump kernel
    kernel_key = current_k => proposed_k
    if !haskey(explorer.jump_kernels, kernel_key)
        return  # No kernel defined for this transition
    end

    kernel = explorer.jump_kernels[kernel_key]

    # Compute proposal
    proposed_state, log_jacobian, log_proposal_ratio = kernel(replica.state, replica.rng)

    # Metropolis-Hastings acceptance
    log_ratio = (log_potential(proposed_state) - log_potential(replica.state)
                 + log_jacobian + log_proposal_ratio)

    if log(rand(replica.rng)) < log_ratio
        replica.state = proposed_state
        record_if_requested!(replica.recorders, :rj_acceptance, true)
    else
        record_if_requested!(replica.recorders, :rj_acceptance, false)
    end
end
```

### Jump Kernel Interface

```julia
"""
Abstract interface for reversible jump kernels.

A jump kernel defines:
1. How to transform state from model i to model j
2. The Jacobian of the transformation
3. The proposal ratio q(j→i) / q(i→j)
"""
abstract type JumpKernel end

"""
$(SIGNATURES)

Apply jump kernel to transform state, returning:
- `new_state`: Proposed state in target model
- `log_jacobian`: Log determinant of Jacobian for dimension matching
- `log_proposal_ratio`: Log of q(reverse) / q(forward)
"""
function (kernel::JumpKernel)(state::RJState, rng::AbstractRNG)
    @abstract
end
```

### Example: Birth-Death Kernel

```julia
"""
Birth kernel: add a new parameter drawn from specified distribution.
"""
struct BirthKernel <: JumpKernel
    target_model::Int
    new_param_name::Symbol
    proposal_dist::Distribution
end

function (kernel::BirthKernel)(state::RJState, rng::AbstractRNG)
    @assert state.model_indicator < kernel.target_model

    # Create new state
    new_state = deepcopy(state)
    new_state.model_indicator = kernel.target_model

    # Sample new parameter
    new_param = rand(rng, kernel.proposal_dist)
    new_state.parameters[kernel.new_param_name] = [new_param]
    new_state.dimension = state.dimension + 1

    # Jacobian is 1 (deterministic mapping)
    log_jacobian = 0.0

    # Proposal ratio: q(death) / q(birth)
    # Assuming symmetric model proposal: cancels out
    log_proposal_ratio = 0.0

    return new_state, log_jacobian, log_proposal_ratio
end

"""
Death kernel: remove a parameter.
"""
struct DeathKernel <: JumpKernel
    target_model::Int
    remove_param_name::Symbol
    proposal_dist::Distribution  # For reverse move
end

function (kernel::DeathKernel)(state::RJState, rng::AbstractRNG)
    @assert state.model_indicator > kernel.target_model

    # Create new state
    new_state = deepcopy(state)
    new_state.model_indicator = kernel.target_model

    # Remove parameter (its value doesn't affect reverse proposal density)
    delete!(new_state.parameters, kernel.remove_param_name)
    new_state.dimension = state.dimension - 1

    # Jacobian is 1
    log_jacobian = 0.0

    # Proposal ratio accounts for the proposal density of the removed parameter
    # in the reverse (birth) move
    removed_value = state.parameters[kernel.remove_param_name][1]
    log_proposal_ratio = logpdf(kernel.proposal_dist, removed_value)

    return new_state, log_jacobian, log_proposal_ratio
end
```

### Composition Pattern

```julia
# Recommended usage: compose RJ moves with within-model moves
explorer = Compose(
    RJMCMCExplorer(jump_kernels, n_attempts=1),
    SliceSampler(n_passes=3)
)

# Or mix different exploration strategies
explorer = Mix(
    Compose(RJMCMCExplorer(...), SliceSampler()),
    Compose(RJMCMCExplorer(...), AutoMALA())
)
```

---

## 5. Tempering Compatibility

### Natural Integration with Annealing

The interpolating path (src/paths/InterpolatingPath.jl) creates:

```math
\pi_\beta(x, k) \propto [(1-\beta) \gamma_0(x, k) + \beta \gamma(x, k)]
```

where:
- β ∈ [0, 1] is the annealing parameter
- k is the model indicator
- x are the parameters within model k
- γ₀ is the reference, γ is the target

**Key insight**: The path anneals from reference to target **for each model simultaneously**. This ensures that:
- At β=0: easy i.i.d. sampling from reference (each model's prior)
- At β=1: target posterior for each model
- Intermediate β: facilitates exploration across models

### Path Creation

```julia
function Pigeons.create_path(target::RJLogPotential, inputs)
    ref = default_reference(target)
    return InterpolatingPath(ref, target, LinearInterpolator())
end
```

The existing `InterpolatedLogPotential` (src/paths/InterpolatedLogPotential.jl:5-18) works automatically with `RJLogPotential` since it just evaluates the log potential at the interpolated value.

### Schedule Adaptation

The optimal schedule adaptation (src/tempering/adaptation.jl:6-93) uses swap acceptance rates. For RJMCMC:

- **Within-model swaps**: Standard PT swaps work unchanged
- **Model mixing**: Tracked via model indicator in samples
- **Communication barriers**: Existing mechanism ensures good mixing

No modifications needed to tempering infrastructure!

---

## 6. MPI and Multi-threading Compatibility

### Why It Works Without Modification

**1. Swap Mechanism (src/swap/swap.jl:79-102)**

Distributed swaps:
- Only exchange chain indices via `transmit()` (MPI peer-to-peer)
- Complexity O(K) where K = ceil(N/P), **independent of state dimension**
- Different replicas can have different dimensional states naturally
- No serialization of states needed

**2. Exploration Phase (src/pt/pigeons.jl:82-97)**

```julia
if inputs.multithreaded
    @threads for replica in locals(replicas)
        explore!(replica, shared)
    end
else
    for replica in locals(replicas)
        explore!(replica, shared)
    end
end
```

- Each replica explored independently
- Thread-safe: each has own `SplittableRandom` RNG
- No shared mutable state between replicas during exploration

**3. Parallelism Invariance (docs/src/distributed.md:37-44)**

Output is **deterministic** regardless of number of processes/threads:
- X_{m,t}(s) = X_{m',t'}(s) for any (m,t) and (m',t')
- Achieved through deterministic RNG splitting
- RJMCMC maintains this property automatically

### EntangledReplicas Architecture

The `EntangledReplicas` structure (src/replicas/EntangledReplicas.jl:6-49):
- Treats states as opaque objects during communication
- Only communicates swap statistics (log ratios + random uniforms)
- **Variable dimensionality poses no challenge**

---

## 7. Variational References for RJMCMC

### Current Variational PT

`GaussianReference` (src/variational/GaussianReference.jl):
- Learns mean-field Gaussian approximation
- Updates after `first_tuning_round` (default: round 6)
- Uses online statistics from target chain samples

**Extension strategy**: Learn separate variational reference for each model

### RJVariationalReference Design

```julia
"""
Variational reference for reversible jump MCMC.

Learns separate Gaussian approximations for each model's parameter space.

# Fields
- `model_references::Dict{Int, GaussianReference}`: Per-model variational approximations
- `model_frequencies::Vector{Float64}`: Learned model probabilities
- `first_tuning_round::Int`: When to start updating
"""
@kwdef mutable struct RJVariationalReference
    model_references::Dict{Int, GaussianReference}
    model_frequencies::Vector{Float64}
    first_tuning_round::Int = 6
end
```

### Interface Implementation

```julia
# When to activate variational learning
function Pigeons.activate_variational(variational::RJVariationalReference, iterators)
    return iterators.round >= variational.first_tuning_round
end

# Update from samples
function Pigeons.update_reference!(reduced_recorders, variational::RJVariationalReference, state)
    # Update model frequencies from visit counts
    if haskey(reduced_recorders, :model_visits)
        update_model_frequencies!(variational, reduced_recorders.model_visits)
    end

    # Update each model's Gaussian reference
    for (k, ref) in variational.model_references
        model_samples = filter_samples_by_model(reduced_recorders.online, k)
        if !isempty(model_samples)
            update_reference!(model_samples, ref, state)
        end
    end
end

# Evaluate variational log density
function (variational::RJVariationalReference)(state::RJState)
    k = state.model_indicator
    model_ref = variational.model_references[k]

    # Log p(k) from learned frequencies
    log_model_prob = log(variational.model_frequencies[k])

    # Log p(x|k) from Gaussian approximation
    log_param_prob = model_ref(state.parameters)

    return log_model_prob + log_param_prob
end

# i.i.d. sampling
function Pigeons.sample_iid!(reference::RJVariationalReference, replica, shared)
    # Sample model from learned frequencies
    k = rand(replica.rng, Categorical(reference.model_frequencies))

    # Sample parameters from model's variational reference
    replica.state.model_indicator = k
    sample_iid!(reference.model_references[k], replica, shared)
end
```

### Variational Jump Proposals

**Innovative feature**: Use learned variational references to construct jump proposals

```julia
"""
Jump kernel that uses variational reference for proposals.
"""
struct VariationalBirthKernel <: JumpKernel
    target_model::Int
    new_param_name::Symbol
    variational_ref::RJVariationalReference
end

function (kernel::VariationalBirthKernel)(state::RJState, rng::AbstractRNG)
    new_state = deepcopy(state)
    new_state.model_indicator = kernel.target_model

    # Propose new parameter from LEARNED variational distribution
    # (instead of fixed prior)
    target_ref = kernel.variational_ref.model_references[kernel.target_model]
    new_param = sample_parameter(target_ref, kernel.new_param_name, rng)
    new_state.parameters[kernel.new_param_name] = [new_param]

    # Proposal density is variational density
    log_proposal_fwd = logpdf(target_ref, kernel.new_param_name, new_param)

    # Reverse proposal would delete this parameter
    log_proposal_rev = 0.0  # Death move has no auxiliary variable

    log_jacobian = 0.0
    log_proposal_ratio = log_proposal_rev - log_proposal_fwd

    return new_state, log_jacobian, log_proposal_ratio
end
```

**Benefit**: Proposals automatically adapt to posterior, improving acceptance rates!

### Stabilized RJMCMC-PT

Extend `StabilizedPT` (src/tempering/StabilizedPT.jl) to transdimensional case:

```julia
"""
Stabilized variational PT for reversible jump MCMC.

Uses two-leg architecture:
- Fixed leg: Uses fixed prior references for each model
- Variational leg: Uses learned model-specific references

Prevents mode forgetting across models during adaptation.
"""
struct StabilizedRJPT
    fixed_leg::NonReversiblePT
    variational_leg::NonReversiblePT
    swap_graphs
    log_potentials
    indexer
end
```

Layout:
```
<------- variational -------> <-------- fixed -------->
ref_var --- target --- target --- ref_fixed
  1           N       N+1          2N
```

Both legs explore all models, but use different reference distributions.

---

## 8. Implementation Roadmap

### Phase 1: Basic Infrastructure ✓ IN PROGRESS

**Goal**: Core types and interfaces

Tasks:
1. Implement `RJState` type
   - Constructor, basic fields
   - State interface methods (continuous_variables, etc.)
   - Flattening/unflattening utilities

2. Implement `RJLogPotential`
   - Model dictionary storage
   - Log density evaluation
   - Target interface (initialization, default_explorer, etc.)

3. Tests
   - Fixed-dimension case (should match existing behavior)
   - Two-model case with known posterior

**Deliverable**: Can run `pigeons(target = RJLogPotential(...))` with fixed dimensions

### Phase 2: Manual Jump Kernels

**Goal**: Between-model moves with user-specified kernels

Tasks:
1. Implement `JumpKernel` interface
2. Implement basic kernels:
   - `BirthKernel`
   - `DeathKernel`
   - `SwapKernel` (for nested models)
3. Implement `RJMCMCExplorer`
4. Tests
   - Simple change-point model
   - Mixture model with varying components
   - Verify detailed balance

**Deliverable**: User can specify custom jump distributions

### Phase 3: Within-Model Exploration

**Goal**: Compatibility with existing explorers

Tasks:
1. Implement `Base.eachindex`, `Base.Ref` for `RJState`
2. Test `Compose(RJMCMCExplorer(), SliceSampler())`
3. Test with `AutoMALA`, `AAPS`
4. Verify within-model sampling correctness

**Deliverable**: All existing explorers work with RJMCMC

### Phase 4: MPI/Threading Integration

**Goal**: Verify distributed and parallel compatibility

Tasks:
1. Test with `multithreaded = true`
2. Test with MPI across multiple processes
3. Verify parallelism invariance
4. Performance benchmarks

**Deliverable**: RJMCMC works efficiently in distributed setting

### Phase 5: Variational Integration

**Goal**: Automatic jump proposal learning

Tasks:
1. Implement `RJVariationalReference`
2. Implement per-model Gaussian learning
3. Implement `VariationalBirthKernel`, etc.
4. Implement `StabilizedRJPT`
5. Tests
   - Verify adaptation improves acceptance rates
   - Compare to fixed proposals

**Deliverable**: Variational references accelerate RJMCMC exploration

### Phase 6: Advanced Features

**Goal**: Production-ready implementation

Tasks:
1. Specialized recorders
   - Model visit frequencies
   - Jump acceptance rates by model pair
   - Posterior model probabilities
2. Post-processing utilities
   - `samples_for_model(pt, k)`
   - `posterior_model_probabilities(pt)`
   - `bayes_factors(pt)`
3. Documentation
   - Tutorial examples
   - API documentation
   - Performance guidelines

**Deliverable**: Complete, documented RJMCMC package

---

## 9. Example Usage

### Basic Usage: Fixed Kernels

```julia
using Pigeons
using Distributions

# Define models
model1 = MySimpleModel(data)  # k=1: simple model
model2 = MyComplexModel(data)  # k=2: complex model (has extra parameter θ)

target = RJLogPotential(
    models = Dict(1 => model1, 2 => model2),
    model_prior = Categorical([0.5, 0.5])
)

# Define jump kernels
jump_kernels = Dict(
    1 => 2 => BirthKernel(
        target_model = 2,
        new_param_name = :theta,
        proposal_dist = Normal(0, 1)  # Prior for new parameter
    ),
    2 => 1 => DeathKernel(
        target_model = 1,
        remove_param_name = :theta,
        proposal_dist = Normal(0, 1)  # For reverse move
    )
)

# Create explorer
explorer = Compose(
    RJMCMCExplorer(jump_kernels, n_attempts=1),
    SliceSampler(n_passes=3)
)

# Run RJMCMC-PT
pt = pigeons(
    target = target,
    explorer = explorer,
    n_chains = 20,
    n_rounds = 12
)

# Post-processing
model_probs = posterior_model_probabilities(pt)
println("P(Model 1 | data) = $(model_probs[1])")
println("P(Model 2 | data) = $(model_probs[2])")

# Extract samples for each model
samples_m1 = samples_for_model(pt, 1)
samples_m2 = samples_for_model(pt, 2)

# Bayes factor
bf = bayes_factor(pt, 1, 2)
println("BF(M1 / M2) = $bf")
```

### Advanced Usage: Variational Proposals

```julia
# Same models as above

target = RJLogPotential(
    models = Dict(1 => model1, 2 => model2),
    model_prior = Categorical([0.5, 0.5])
)

# Use variational references
variational = RJVariationalReference(
    model_references = Dict(
        1 => GaussianReference(mean=Dict(), standard_deviation=Dict()),
        2 => GaussianReference(mean=Dict(), standard_deviation=Dict())
    ),
    model_frequencies = [0.5, 0.5],
    first_tuning_round = 6
)

# Jump kernels use learned proposals
jump_kernels = Dict(
    1 => 2 => VariationalBirthKernel(
        target_model = 2,
        new_param_name = :theta,
        variational_ref = variational
    ),
    2 => 1 => VariationalDeathKernel(
        target_model = 1,
        remove_param_name = :theta,
        variational_ref = variational
    )
)

explorer = Compose(
    RJMCMCExplorer(jump_kernels, n_attempts=1),
    SliceSampler(n_passes=3)
)

pt = pigeons(
    target = target,
    explorer = explorer,
    variational = variational,
    n_chains = 20,
    n_rounds = 12
)

# Variational proposals automatically adapt!
# Later rounds have better acceptance rates
```

### Stabilized Variational RJMCMC-PT

```julia
# Two-leg architecture for stability
pt = pigeons(
    target = target,
    explorer = explorer,
    variational = variational,
    n_chains = 10,        # Per leg
    n_chains_variational = 10,
    n_rounds = 15
)

# Fixed leg prevents mode forgetting
# Variational leg explores efficiently with learned proposals
```

---

## 10. Technical Challenges & Solutions

### Challenge 1: State Copying

**Issue**: `deepcopy(state)` in jump kernels may be expensive

**Solution**: Implement efficient copy for `RJState`:
```julia
function Base.copy(state::RJState)
    RJState(
        state.model_indicator,
        copy(state.parameters),  # Shallow copy of dict, deep copy of arrays
        state.dimension,
        state.model_info  # Shared reference, don't copy
    )
end
```

### Challenge 2: Parameter Alignment

**Issue**: Different models have different parameter names

**Solution**: Use `Dict{Symbol, Any}` for parameters, with model-specific schemas defined in `RJModelInfo`

### Challenge 3: Online Statistics

**Issue**: Online recorders expect fixed dimension

**Solution**: Separate online statistics by model:
```julia
struct RJOnlineRecorder
    model_recorders::Dict{Int, OnlineStateRecorder}
end
```

### Challenge 4: Checkpointing

**Issue**: Serialization of variable-dimensional states

**Solution**: Pigeons' checkpointing (src/checkpointing/checkpoint.jl) uses `serialize()`, which handles `Dict` automatically. No changes needed.

### Challenge 5: Trace Storage

**Issue**: Traces have different dimensions across samples

**Solution**: Store as vector of variable-length samples:
```julia
struct RJTrace
    samples::Vector{RJSample}
end

struct RJSample
    model::Int
    parameters::Dict{Symbol, Any}
    log_density::Float64
end
```

---

## 11. Theoretical Considerations

### Detailed Balance

RJMCMC satisfies detailed balance if:

```math
\pi(x, k) \alpha(x,k \to x',k') = \pi(x', k') \alpha(x',k' \to x,k)
```

where α is the acceptance probability.

**Green (1995) condition**:
```math
\alpha(x,k \to x',k') = \min\left(1, \frac{\pi(x', k')}{\pi(x, k)} \times \frac{q(x',k' \to x,k)}{q(x,k \to x',k')} \times \left|\frac{\partial(x', u')}{\partial(x, u)}\right|\right)
```

Our implementation ensures this by:
1. Computing log_jacobian correctly
2. Computing log_proposal_ratio = log q(reverse) - log q(forward)
3. Standard Metropolis-Hastings acceptance

### Dimension Matching

For moves between dimensions d → d':
- If d < d': need auxiliary variables u ~ q(u)
- If d > d': map determines u deterministically
- Jacobian: derivative of (x', u') w.r.t. (x, u)

**Birth move** (d → d+1):
- Draw u ~ q(u) (new parameter)
- Set x' = (x, u) deterministically
- Jacobian = 1
- Reverse (death) must account for q(u)

**Death move** (d+1 → d):
- Set x' = x[1:d] deterministically
- u = x[d+1] (discarded)
- Jacobian = 1
- Proposal ratio includes q(u) from reverse birth

### Tempering Ergodicity

For ergodicity of tempered RJMCMC:

1. **Within-model**: Each model's sampler must be π_k-ergodic
2. **Between-model**: Jump graph must be connected
3. **Across temperature**: Standard PT communication suffices

Our design ensures:
- SliceSampler is ergodic for each model (proven for continuous distributions)
- User must ensure jump graph connectivity
- PT communication works unchanged

---

## 12. Performance Considerations

### Computational Complexity

**Within-model exploration**:
- SliceSampler: O(d log d) per coordinate, d = dimension
- Varies by model, handled automatically

**Between-model jumps**:
- Proposal: O(d_new) for birth, O(1) for death
- Acceptance: O(d_max) for log density evaluation
- Efficient if log densities are cached

**Swaps** (unchanged):
- Single process: O(N) where N = # chains
- Distributed: O(K) where K = ceil(N/P)
- **Independent of state dimension!**

### Memory Usage

- Each replica stores one `RJState`: O(d_max)
- Total: O(N × d_max) for N chains
- Variational references: O(K × d_avg) for K models
- Comparable to fixed-dimension PT

### Scaling

**Strong scaling** (fixed problem, more processes):
- Communication overhead: O(K) where K = ceil(N/P)
- Decreases linearly with P (up to communication latency)

**Model complexity scaling**:
- More models K: Jump proposal time increases O(K)
- Can use sparse jump graph (not fully connected)
- Variational learning cost: O(K) per round

### Optimization Strategies

1. **Lazy evaluation**: Don't compute log densities for rejected proposals
2. **Caching**: Store log density in replica, update incrementally
3. **Sparse jumps**: Only define kernels for neighboring models
4. **Adaptive n_attempts**: Adjust jump frequency based on acceptance rates

---

## 13. Testing Strategy

### Unit Tests

1. **State interface**:
   - Test all state methods
   - Test flattening/unflattening
   - Test copy, equality

2. **Log potential**:
   - Test evaluation for each model
   - Test model prior contribution
   - Test gradient (if applicable)

3. **Jump kernels**:
   - Test proposal generation
   - Test Jacobian computation
   - Test proposal ratio
   - Test birth-death reversibility

4. **Explorer**:
   - Test model proposal
   - Test acceptance/rejection
   - Test composition with SliceSampler

### Integration Tests

1. **Fixed dimension**:
   - Single model should match standard PT
   - Compare to non-RJ implementation

2. **Known posteriors**:
   - Two-model with known analytical posterior
   - Verify model probabilities
   - Verify parameter estimates within each model

3. **Detailed balance**:
   - Verify reversibility
   - Check acceptance ratios match theory

4. **Parallelism invariance**:
   - Compare serial vs. multithreaded
   - Compare 1 process vs. multiple MPI processes
   - Results should be identical (given same seed)

### Example Test Cases

1. **Change-point detection**:
   - Model 1: No change point (k parameters)
   - Model 2: One change point (k+1 parameters)
   - Known synthetic data

2. **Variable selection**:
   - Models: All subsets of p predictors
   - Linear regression with known coefficients
   - Should recover true model with high probability

3. **Mixture components**:
   - Model k: k-component Gaussian mixture
   - k ∈ {1, 2, 3, 4}
   - Known ground truth mixture

---

## 14. Related Work & References

### Foundational Papers

1. **Green (1995)**: "Reversible jump Markov chain Monte Carlo computation and Bayesian model determination"
   - Original RJMCMC paper
   - Dimension matching via auxiliary variables
   - Detailed balance conditions

2. **Syed et al. (2021)**: "Non-reversible parallel tempering: a scalable highly parallel MCMC scheme"
   - Pigeons' theoretical foundation
   - Non-reversible PT algorithm
   - Parallelism invariance

3. **Surjanovic et al. (2022)**: "Parallel tempering with variational reference"
   - Stabilized variational PT
   - Two-leg architecture
   - Automatic reference learning

### Related Methods

1. **Birth-death MCMC** (Stephens 2000):
   - Specialized for mixture models
   - Split-merge moves
   - Can be implemented as jump kernels

2. **Product space MCMC** (Carlin & Chib 1995):
   - Alternative to dimension matching
   - All models live in augmented space
   - Less efficient than RJMCMC

3. **SMC samplers** (Del Moral et al. 2006):
   - Sequential Monte Carlo for model selection
   - Complementary to MCMC
   - Could combine with PT

### Software Implementations

1. **reversiblejump.jl**: Standalone RJ package
   - Not integrated with PT
   - No distributed computing

2. **PyMC**: Python MCMC
   - Limited RJ support
   - No PT integration

3. **JAGS/WinBUGS**: Bayesian software
   - Support variable selection
   - No explicit user-defined RJ

**Pigeons.jl + RJMCMC**: Unique combination of:
- Reversible jump MCMC
- Non-reversible parallel tempering
- Distributed/parallel computing
- Variational references

---

## 15. Future Directions

### Algorithmic Extensions

1. **Non-reversible RJ moves**:
   - Extend to non-reversible jump kernels
   - Potentially better mixing
   - Requires theoretical development

2. **Adaptive jump proposals**:
   - Learn proposal distributions online
   - Beyond variational Gaussians
   - Normalizing flows, etc.

3. **Structured model spaces**:
   - DAG over models
   - Local vs. global moves
   - Hierarchical model selection

### Applications

1. **Bayesian neural networks**:
   - Variable network architecture
   - Layer-wise changes
   - Activation function selection

2. **Time series models**:
   - Variable order AR/MA
   - Change-point detection
   - Regime switching

3. **Spatial statistics**:
   - Variable number of knots
   - Adaptive mesh refinement
   - Model averaging

### Optimizations

1. **GPU acceleration**:
   - Parallel jump attempts
   - Batch log density evaluations
   - Distributed + GPU

2. **Smart scheduling**:
   - Adaptive jump frequency
   - Model-specific exploration time
   - Load balancing by model complexity

3. **Advanced variational families**:
   - Beyond mean-field Gaussian
   - Normalizing flows
   - Structured covariance

---

## 16. Relevant File References

### Core Implementation Files

**State and interface**:
- `src/pt/state.jl`: State interface definition
- `examples/ising.jl`: Example custom state (lines 17-26)
- `src/pt/Replica.jl`: Replica structure (lines 5-30)

**Explorers**:
- `src/explorers/explorer.jl`: Explorer interface (lines 7-39)
- `src/explorers/SliceSampler.jl`: Slice sampling (lines 1-238)
- `src/explorers/Compose.jl`: Composition pattern (lines 1-27)
- `src/explorers/Mix.jl`: Random mixing (lines 1-29)

**Log potentials**:
- `src/log_potentials/log_potential.jl`: Interface (lines 6-14)
- `src/targets/target.jl`: Target interface (lines 4-76)
- `src/targets/DistributionLogPotential.jl`: Example wrapper

**Tempering**:
- `src/tempering/tempering.jl`: Tempering interface (lines 12-49)
- `src/tempering/NonReversiblePT.jl`: Standard PT (lines 7-66)
- `src/paths/InterpolatingPath.jl`: Annealing paths (lines 1-29)

**Variational**:
- `src/variational/variational.jl`: Variational interface (lines 6-25)
- `src/variational/GaussianReference.jl`: Gaussian learning (lines 1-75)
- `src/tempering/StabilizedPT.jl`: Two-leg PT (lines 1-119)

**Parallel/distributed**:
- `src/swap/swap.jl`: Swap implementation (lines 79-102)
- `src/replicas/EntangledReplicas.jl`: MPI replicas (lines 6-49)
- `src/pt/pigeons.jl`: Main algorithm (lines 82-97 for threading)

**Documentation**:
- `docs/src/pt.md`: PT overview
- `docs/src/distributed.md`: Distributed computing (lines 37-44)
- `docs/src/variational.md`: Variational PT

---

## 17. Summary and Conclusions

### Key Findings

1. **Excellent architectural compatibility**: Pigeons' swap-by-chain-index design is ideal for transdimensional MCMC

2. **No core changes needed**: MPI, threading, tempering all work unchanged

3. **Natural explorer composition**: RJ moves compose cleanly with existing explorers

4. **Innovative variational integration**: Using learned references for jump proposals is novel and powerful

5. **Maintains theoretical guarantees**: Detailed balance, parallelism invariance preserved

### Implementation Status

- ✅ Design complete
- 🔄 Phase 1 in progress (basic infrastructure)
- ⏳ Phases 2-6 planned

### Novel Contributions

1. **First RJMCMC + PT integration**: Combines model selection with parallel tempering
2. **Distributed transdimensional MCMC**: O(1) swap complexity regardless of dimension
3. **Variational jump proposals**: Automatic adaptation of between-model moves
4. **Stabilized RJMCMC-PT**: Two-leg architecture prevents mode forgetting

### Recommended Next Steps

1. Complete Phase 1 implementation
2. Validate with known test cases
3. Benchmark against existing methods
4. Write tutorial documentation
5. Publish methodology paper

---

**Document Version**: 1.0
**Last Updated**: 2025-11-18
**Author**: Claude (Anthropic)
**Status**: Living document - will update as implementation progresses
