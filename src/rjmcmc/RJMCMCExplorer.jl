"""
Explorer for reversible jump MCMC moves between models.

Proposes transitions between models using specified jump kernels and
accepts/rejects based on the Metropolis-Hastings ratio.

# Fields
- `jump_kernels::Dict{Pair{Int,Int}, JumpKernel}`: Jump kernels for model transitions
- `n_attempts::Int`: Number of jump attempts per step (default: 1)
- `uniform_model_proposal::Bool`: Propose target model uniformly vs. from available kernels (default: true)

# Example
```julia
using Distributions

# Create jump kernels
birth, death = create_birth_death_pair(
    model_low = 1,
    model_high = 2,
    new_params = Dict(:theta => Normal(0, 1))
)

jump_kernels = Dict(birth, death)

# Create explorer
explorer = RJMCMCExplorer(
    jump_kernels = jump_kernels,
    n_attempts = 1
)

# Compose with within-model explorer
full_explorer = Compose(explorer, SliceSampler())
```
"""
struct RJMCMCExplorer
    jump_kernels::Dict{Pair{Int,Int}, JumpKernel}
    n_attempts::Int
    uniform_model_proposal::Bool
end

function RJMCMCExplorer(;
    jump_kernels::Dict{Pair{Int,Int}, <:JumpKernel},
    n_attempts::Int=1,
    uniform_model_proposal::Bool=true
)
    return RJMCMCExplorer(jump_kernels, n_attempts, uniform_model_proposal)
end

"""
$SIGNATURES

Perform RJMCMC exploration step: attempt model jumps.
"""
function Pigeons.step!(explorer::RJMCMCExplorer, replica, shared)
    # Get log potential for current chain
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)

    # Attempt jumps
    for _ in 1:explorer.n_attempts
        attempt_model_jump!(explorer, replica, log_potential)
    end
end

"""
$SIGNATURES

Attempt a single model jump with Metropolis-Hastings acceptance.
"""
function attempt_model_jump!(explorer::RJMCMCExplorer, replica, log_potential)
    current_k = replica.state.model_indicator

    # Propose target model
    proposed_k = propose_target_model(explorer, current_k, replica.rng)

    if current_k == proposed_k
        return  # No jump proposed
    end

    # Check if kernel exists for this transition
    kernel_key = current_k => proposed_k
    if !haskey(explorer.jump_kernels, kernel_key)
        return  # No kernel defined for this transition
    end

    # Get jump kernel
    kernel = explorer.jump_kernels[kernel_key]

    # Current log density
    log_density_current = log_potential(replica.state)

    # Propose new state
    proposed_state, log_jacobian, log_proposal_ratio = kernel(replica.state, replica.rng)

    # Proposed log density
    log_density_proposed = log_potential(proposed_state)

    # Metropolis-Hastings ratio
    # log α = log π(x', k') - log π(x, k) + log |J| + log [q(x',k' → x,k) / q(x,k → x',k')]
    log_acceptance_ratio = (
        log_density_proposed - log_density_current +
        log_jacobian +
        log_proposal_ratio
    )

    # Accept/reject
    if log(rand(replica.rng)) < log_acceptance_ratio
        replica.state = proposed_state
        Pigeons.@record_if_requested! replica.recorders :rj_acceptance true
        Pigeons.@record_if_requested! replica.recorders :rj_jumps (current_k, proposed_k)
    else
        Pigeons.@record_if_requested! replica.recorders :rj_acceptance false
    end
end

"""
$SIGNATURES

Propose a target model for the jump.

If `uniform_model_proposal` is true, proposes uniformly from models with available kernels.
Otherwise, randomly selects from available kernels for current model.
"""
function propose_target_model(explorer::RJMCMCExplorer, current_k::Int, rng::AbstractRNG)
    # Find available transitions from current model
    available_targets = Int[]

    for (source, target) in keys(explorer.jump_kernels)
        if source == current_k
            push!(available_targets, target)
        end
    end

    if isempty(available_targets)
        return current_k  # No transitions available, stay in current model
    end

    if explorer.uniform_model_proposal
        # Uniformly select from available targets
        return rand(rng, available_targets)
    else
        # For now, also uniform (could weight by model prior in future)
        return rand(rng, available_targets)
    end
end

# Explorer interface implementations

"""
$SIGNATURES

Recorder builders for RJMCMC explorer statistics.
"""
function Pigeons.explorer_recorder_builders(explorer::RJMCMCExplorer)
    return Function[
        () -> (rj_acceptance = Pigeons.Mean()),
        () -> (rj_jumps = Pigeons.GroupBy(Pigeons.Sum()))
    ]
end

"""
$SIGNATURES

Adaptation for RJMCMC explorer (currently no adaptation implemented).
"""
function Pigeons.adapt_explorer(
    explorer::RJMCMCExplorer,
    reduced_recorders,
    current_pt,
    new_tempering
)
    # For now, no adaptation
    # Future: could adapt proposal distributions, jump attempt frequency, etc.
    return explorer
end
