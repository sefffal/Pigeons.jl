"""
Abstract interface for reversible jump kernels.

A jump kernel defines how to transform state from model i to model j,
including:
1. The transformation itself
2. The Jacobian of the transformation (for dimension matching)
3. The proposal ratio q(j→i) / q(i→j)

# Interface

Implement the function call operator:
```julia
(kernel::MyKernel)(state::RJState, rng::AbstractRNG) -> (new_state, log_jacobian, log_proposal_ratio)
```

Returns:
- `new_state::RJState`: Proposed state in target model
- `log_jacobian::Float64`: Log determinant of Jacobian for dimension matching
- `log_proposal_ratio::Float64`: Log of q(reverse) / q(forward)
"""
abstract type JumpKernel end

# Birth kernel: add new parameters

"""
Birth kernel: add new parameter(s) to transition to a higher-dimensional model.

# Fields
- `target_model::Int`: Model index to jump to
- `new_params::Dict{Symbol, Distribution}`: Proposal distributions for new parameters
- `deterministic::Bool`: If true, new params are set to distribution mean (default: false)

# Example
```julia
using Distributions

# Birth move from model 1 to model 2, adding parameter :theta
kernel = BirthKernel(
    target_model = 2,
    new_params = Dict(:theta => Normal(0, 1))
)
```
"""
struct BirthKernel <: JumpKernel
    target_model::Int
    new_params::Dict{Symbol, Distributions.Distribution}
    deterministic::Bool
end

function BirthKernel(; target_model::Int, new_params::Dict{Symbol, <:Distributions.Distribution}, deterministic::Bool=false)
    return BirthKernel(target_model, new_params, deterministic)
end

"""
$SIGNATURES

Apply birth kernel: add new parameters sampled from proposal distributions.
"""
function (kernel::BirthKernel)(state::RJState, rng::AbstractRNG)
    @assert state.model_indicator != kernel.target_model "Already in target model"

    # Create new state
    new_state = copy(state)
    new_state.model_indicator = kernel.target_model

    # Sample or set new parameters
    log_proposal_fwd = 0.0

    for (param_name, proposal_dist) in kernel.new_params
        if kernel.deterministic
            # Use mean for deterministic mapping
            new_val = mean(proposal_dist)
        else
            # Sample from proposal
            new_val = rand(rng, proposal_dist)
            log_proposal_fwd += logpdf(proposal_dist, new_val)
        end

        # Add to parameters
        new_state.parameters[param_name] = [new_val]
    end

    # Update dimension
    new_state.dimension = new_state.model_info.dimensions[kernel.target_model]

    # Jacobian is 1 for simple birth moves (deterministic mapping of existing params)
    log_jacobian = 0.0

    # Proposal ratio: q(reverse delete) / q(forward birth)
    # Reverse move (death) has no auxiliary random variables, so q(reverse) = 1
    # Forward move sampled from proposal, so q(forward) = product of proposal densities
    log_proposal_ratio = -log_proposal_fwd

    return new_state, log_jacobian, log_proposal_ratio
end

# Death kernel: remove parameters

"""
Death kernel: remove parameter(s) to transition to a lower-dimensional model.

# Fields
- `target_model::Int`: Model index to jump to
- `remove_params::Vector{Symbol}`: Parameters to remove
- `reverse_proposals::Dict{Symbol, Distribution}`: Proposal distributions for reverse birth move

# Example
```julia
# Death move from model 2 to model 1, removing parameter :theta
kernel = DeathKernel(
    target_model = 1,
    remove_params = [:theta],
    reverse_proposals = Dict(:theta => Normal(0, 1))
)
```
"""
struct DeathKernel <: JumpKernel
    target_model::Int
    remove_params::Vector{Symbol}
    reverse_proposals::Dict{Symbol, Distributions.Distribution}
end

function DeathKernel(; target_model::Int, remove_params::Vector{Symbol}, reverse_proposals::Dict{Symbol, <:Distributions.Distribution})
    return DeathKernel(target_model, remove_params, reverse_proposals)
end

"""
$SIGNATURES

Apply death kernel: remove specified parameters.
"""
function (kernel::DeathKernel)(state::RJState, rng::AbstractRNG)
    @assert state.model_indicator != kernel.target_model "Already in target model"

    # Create new state
    new_state = copy(state)
    new_state.model_indicator = kernel.target_model

    # Compute proposal density for removed parameters (for reverse move)
    log_proposal_rev = 0.0

    for param_name in kernel.remove_params
        if haskey(state.parameters, param_name) && haskey(kernel.reverse_proposals, param_name)
            removed_val = state.parameters[param_name][1]
            log_proposal_rev += logpdf(kernel.reverse_proposals[param_name], removed_val)
        end

        # Remove from parameters
        delete!(new_state.parameters, param_name)
    end

    # Update dimension
    new_state.dimension = new_state.model_info.dimensions[kernel.target_model]

    # Jacobian is 1
    log_jacobian = 0.0

    # Proposal ratio: q(reverse birth) / q(forward death)
    # Forward move (death) is deterministic: q(forward) = 1
    # Reverse move (birth) samples from proposal: q(reverse) = product of proposal densities
    log_proposal_ratio = log_proposal_rev

    return new_state, log_jacobian, log_proposal_ratio
end

# Utility: create matching birth-death pair

"""
$SIGNATURES

Create a matching pair of birth and death kernels for reversible jumps.

# Arguments
- `model_low::Int`: Lower-dimensional model index
- `model_high::Int`: Higher-dimensional model index
- `new_params::Dict{Symbol, Distribution}`: Parameters to add/remove and their proposal distributions
- `deterministic::Bool`: Use deterministic proposals (default: false)

# Returns
- `Pair{Int, JumpKernel}`: Birth kernel (low → high)
- `Pair{Int, JumpKernel}`: Death kernel (high → low)

# Example
```julia
birth, death = create_birth_death_pair(
    model_low = 1,
    model_high = 2,
    new_params = Dict(:theta => Normal(0, 1))
)

jump_kernels = Dict(birth, death)
```
"""
function create_birth_death_pair(;
    model_low::Int,
    model_high::Int,
    new_params::Dict{Symbol, <:Distributions.Distribution},
    deterministic::Bool=false
)
    birth = model_low => BirthKernel(
        target_model = model_high,
        new_params = new_params,
        deterministic = deterministic
    )

    death = model_high => DeathKernel(
        target_model = model_low,
        remove_params = collect(keys(new_params)),
        reverse_proposals = new_params
    )

    return birth, death
end
