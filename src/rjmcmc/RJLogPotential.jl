"""
Log potential for reversible jump MCMC supporting multiple models.

# Fields
- `models::Dict{Int, Any}`: Map from model index to log potential for that model
- `model_prior::Distributions.DiscreteUnivariateDistribution`: Prior over model indicators
- `model_info::RJModelInfo`: Shared model structure information

# Example
```julia
using Distributions

# Define individual model log potentials
model1_lp(params) = logpdf(Normal(0, 1), params[:x][1])
model2_lp(params) = logpdf(Normal(0, 1), params[:x][1]) + logpdf(Normal(0, 1), params[:y][1])

# Create RJMCMC target
model_specs = Dict(
    1 => (continuous = [:x], discrete = Symbol[]),
    2 => (continuous = [:x, :y], discrete = Symbol[])
)

target = RJLogPotential(
    models = Dict(1 => model1_lp, 2 => model2_lp),
    model_prior = Categorical([0.5, 0.5]),
    model_info = RJModelInfo(model_specs)
)
```
"""
struct RJLogPotential{T, D <: Distributions.DiscreteUnivariateDistribution}
    models::Dict{Int, T}
    model_prior::D
    model_info::RJModelInfo
end

"""
$SIGNATURES

Convenience constructor that creates RJLogPotential from model specifications.

# Arguments
- `models::Dict{Int, Any}`: Log potential functions for each model
- `model_prior`: Prior distribution over model indices
- `model_specs::Dict{Int, NamedTuple}`: Variable specifications for each model

# Example
```julia
model_specs = Dict(
    1 => (continuous = [:x], discrete = Symbol[]),
    2 => (continuous = [:x, :y], discrete = Symbol[])
)

target = RJLogPotential(
    models = Dict(1 => model1_lp, 2 => model2_lp),
    model_prior = Categorical([0.5, 0.5]),
    model_specs = model_specs
)
```
"""
function RJLogPotential(;
    models::Dict{Int, T},
    model_prior::D,
    model_specs::Dict{Int, <:NamedTuple}
) where {T, D <: Distributions.DiscreteUnivariateDistribution}
    model_info = RJModelInfo(model_specs)
    return RJLogPotential(models, model_prior, model_info)
end

"""
$SIGNATURES

Evaluate the log potential at a given RJState.

Computes: log π(x, k) = log π(x | k) + log π(k)
where k is the model indicator and x are the parameters.
"""
function (lp::RJLogPotential)(state::RJState)
    k = state.model_indicator

    # Get log potential for this model
    model_lp = lp.models[k]

    # Evaluate log π(x | k)
    log_params = model_lp(state.parameters)

    # Add log π(k)
    log_model = logpdf(lp.model_prior, k)

    return log_params + log_model
end

# Target interface implementation

"""
$SIGNATURES

Initialize a state for the RJMCMC target.

Randomly samples a model from the prior and initializes parameters to zeros.
"""
function Pigeons.initialization(target::RJLogPotential, rng::AbstractRNG, replica_index::Int)
    # Sample model from prior
    k = rand(rng, target.model_prior)

    # Get the corresponding model's log potential
    model_lp = target.models[k]

    # Initialize parameters
    # Try to call initialization on the model if it's a Pigeons target
    if applicable(Pigeons.initialization, model_lp, rng, replica_index)
        model_state = Pigeons.initialization(model_lp, rng, replica_index)
        # Convert to dict format
        parameters = convert_to_parameter_dict(model_state, target.model_info, k)
    else
        # Default: initialize to zeros
        parameters = initialize_zero_parameters(target.model_info, k)
    end

    dimension = target.model_info.dimensions[k]

    return RJState(k, parameters, dimension, target.model_info)
end

"""
$SIGNATURES

Convert various state types to parameter dictionary format.
"""
function convert_to_parameter_dict(state, model_info::RJModelInfo, k::Int)
    parameters = Dict{Symbol, Vector{Float64}}()

    if state isa AbstractVector
        # If it's a vector, map to continuous variables in order
        cont_vars = model_info.continuous_vars[k]
        idx = 1
        for var_name in cont_vars
            parameters[var_name] = [Float64(state[idx])]
            idx += 1
        end
    elseif state isa Dict
        # Already in dict format
        for (key, val) in state
            parameters[key] = val isa Vector ? Float64.(val) : [Float64(val)]
        end
    else
        # Try to extract via variable interface
        for var_name in model_info.continuous_vars[k]
            if applicable(Pigeons.variable, state, var_name)
                val = Pigeons.variable(state, var_name)
                parameters[var_name] = val isa Vector ? Float64.(val) : [Float64(val)]
            end
        end
        for var_name in model_info.discrete_vars[k]
            if applicable(Pigeons.variable, state, var_name)
                val = Pigeons.variable(state, var_name)
                parameters[var_name] = val isa Vector ? Float64.(val) : [Float64(val)]
            end
        end
    end

    return parameters
end

"""
$SIGNATURES

Initialize parameters to zeros for a given model.
"""
function initialize_zero_parameters(model_info::RJModelInfo, k::Int)
    parameters = Dict{Symbol, Vector{Float64}}()

    for var_name in model_info.continuous_vars[k]
        parameters[var_name] = [0.0]
    end

    for var_name in model_info.discrete_vars[k]
        parameters[var_name] = [0.0]
    end

    return parameters
end

"""
$SIGNATURES

Default explorer for RJMCMC: compose jump moves with slice sampler.
"""
function Pigeons.default_explorer(target::RJLogPotential)
    # For now, just use SliceSampler
    # Once RJMCMCExplorer is implemented, use:
    # return Compose(RJMCMCExplorer(...), SliceSampler())
    return SliceSampler()
end

"""
$SIGNATURES

Default reference for RJMCMC: same model space with reference log potentials.
"""
function Pigeons.default_reference(target::RJLogPotential)
    # Create reference for each model
    ref_models = Dict{Int, Any}()

    for (k, model) in target.models
        # Try to get default reference for this model
        if applicable(Pigeons.default_reference, model)
            ref_models[k] = Pigeons.default_reference(model)
        else
            # Fallback: use the model itself (assumes it's already a simple reference)
            ref_models[k] = model
        end
    end

    return RJLogPotential(ref_models, target.model_prior, target.model_info)
end

"""
$SIGNATURES

Sample i.i.d. from the reference distribution.
"""
function Pigeons.sample_iid!(reference::RJLogPotential, replica, shared)
    # Sample model from prior
    k = rand(replica.rng, reference.model_prior)

    # Get reference for this model
    ref_model = reference.models[k]

    # Try to sample from model's reference
    if applicable(Pigeons.sample_iid!, ref_model, replica, shared)
        # Save current state to restore model_info
        old_model_info = replica.state.model_info

        # Sample from model's reference (might modify replica.state)
        Pigeons.sample_iid!(ref_model, replica, shared)

        # Convert to RJState
        if !(replica.state isa RJState)
            parameters = convert_to_parameter_dict(replica.state, old_model_info, k)
            dimension = old_model_info.dimensions[k]
            replica.state = RJState(k, parameters, dimension, old_model_info)
        else
            # Already RJState, just update model if needed
            replica.state.model_indicator = k
        end
    else
        # Fallback: initialize to zeros
        parameters = initialize_zero_parameters(reference.model_info, k)
        dimension = reference.model_info.dimensions[k]
        replica.state = RJState(k, parameters, dimension, reference.model_info)
    end
end
