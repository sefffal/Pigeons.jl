"""
State type supporting reversible jump MCMC with variable dimensionality.

# Fields
- `model_indicator::Int`: Current model index (1, 2, ..., K)
- `parameters::Dict{Symbol, Vector{Float64}}`: Model-specific parameters
- `dimension::Int`: Current total dimension
- `model_info::RJModelInfo`: Reference to model structure (shared across replicas)
"""
mutable struct RJState
    model_indicator::Int
    parameters::Dict{Symbol, Vector{Float64}}
    dimension::Int
    model_info::RJModelInfo
end

"""
Information about the model space structure, shared across all replicas.

# Fields
- `n_models::Int`: Total number of models
- `continuous_vars::Dict{Int, Vector{Symbol}}`: Continuous variables for each model
- `discrete_vars::Dict{Int, Vector{Symbol}}`: Discrete variables for each model
- `dimensions::Dict{Int, Int}`: Total dimension for each model
"""
struct RJModelInfo
    n_models::Int
    continuous_vars::Dict{Int, Vector{Symbol}}
    discrete_vars::Dict{Int, Vector{Symbol}}
    dimensions::Dict{Int, Int}
end

"""
$SIGNATURES

Create a RJModelInfo from a dictionary mapping model index to variable specifications.

# Example
```julia
model_specs = Dict(
    1 => (continuous = [:x, :y], discrete = Symbol[]),
    2 => (continuous = [:x, :y, :z], discrete = Symbol[])
)
info = RJModelInfo(model_specs)
```
"""
function RJModelInfo(model_specs::Dict{Int, <:NamedTuple})
    n_models = length(model_specs)
    continuous_vars = Dict{Int, Vector{Symbol}}()
    discrete_vars = Dict{Int, Vector{Symbol}}()
    dimensions = Dict{Int, Int}()

    for (k, spec) in model_specs
        continuous_vars[k] = collect(spec.continuous)
        discrete_vars[k] = get(spec, :discrete, Symbol[])
        dimensions[k] = length(continuous_vars[k]) + length(discrete_vars[k])
    end

    return RJModelInfo(n_models, continuous_vars, discrete_vars, dimensions)
end

# State interface implementation

"""
$SIGNATURES

The names of continuous variables in the current model.
"""
Pigeons.continuous_variables(state::RJState) =
    state.model_info.continuous_vars[state.model_indicator]

"""
$SIGNATURES

The names of discrete variables in the current model.
"""
Pigeons.discrete_variables(state::RJState) =
    state.model_info.discrete_vars[state.model_indicator]

"""
$SIGNATURES

Access the storage for a specific variable in the state.
"""
function Pigeons.variable(state::RJState, name::Symbol)
    if haskey(state.parameters, name)
        return state.parameters[name]
    else
        error("Variable $name not found in current model $(state.model_indicator)")
    end
end

"""
$SIGNATURES

Update the state's entry at symbol `name` and `index` with `value`.
"""
function Pigeons.update_state!(state::RJState, name::Symbol, index::Int, value)
    state.parameters[name][index] = value
end

"""
$SIGNATURES

Extract a sample for postprocessing, including model indicator.
"""
function Pigeons.extract_sample(state::RJState, log_potential, extractor)
    flat_params = flatten_parameters(state)
    return [Float64(state.model_indicator); flat_params; log_potential(state)]
end

"""
$SIGNATURES

Sample names for extracted samples.
"""
function Pigeons.sample_names(state::RJState, log_potential, extractor)
    param_names = get_parameter_names(state)
    return [:model_indicator; param_names; :log_density]
end

# Utility functions

"""
$SIGNATURES

Flatten all parameters of the current model into a single vector.
"""
function flatten_parameters(state::RJState)
    result = Float64[]

    # Continuous variables
    for var_name in continuous_variables(state)
        if haskey(state.parameters, var_name)
            append!(result, state.parameters[var_name])
        end
    end

    # Discrete variables
    for var_name in discrete_variables(state)
        if haskey(state.parameters, var_name)
            append!(result, Float64.(state.parameters[var_name]))
        end
    end

    return result
end

"""
$SIGNATURES

Get parameter names for the current model in flattened order.
"""
function get_parameter_names(state::RJState)
    names = Symbol[]

    # Continuous variables
    for var_name in continuous_variables(state)
        if haskey(state.parameters, var_name)
            n = length(state.parameters[var_name])
            if n == 1
                push!(names, var_name)
            else
                for i in 1:n
                    push!(names, Symbol("$(var_name)[$i]"))
                end
            end
        end
    end

    # Discrete variables
    for var_name in discrete_variables(state)
        if haskey(state.parameters, var_name)
            n = length(state.parameters[var_name])
            if n == 1
                push!(names, var_name)
            else
                for i in 1:n
                    push!(names, Symbol("$(var_name)[$i]"))
                end
            end
        end
    end

    return names
end

"""
$SIGNATURES

Efficient copy of RJState.
"""
function Base.copy(state::RJState)
    # Deep copy parameters dict and arrays, but share model_info (immutable reference)
    new_params = Dict{Symbol, Vector{Float64}}()
    for (k, v) in state.parameters
        new_params[k] = copy(v)
    end

    return RJState(
        state.model_indicator,
        new_params,
        state.dimension,
        state.model_info  # Shared reference
    )
end

# SliceSampler compatibility via Ref interface

"""
$SIGNATURES

Number of parameters in current model (for iteration).
"""
Base.length(state::RJState) = state.dimension

"""
$SIGNATURES

Iterate over indices of current model's parameters.
"""
Base.eachindex(state::RJState) = 1:state.dimension

"""
$SIGNATURES

Get a parameter at a given flattened index.
"""
function Base.getindex(state::RJState, i::Int)
    var_name, local_idx = locate_parameter(state, i)
    return state.parameters[var_name][local_idx]
end

"""
$SIGNATURES

Set a parameter at a given flattened index.
"""
function Base.setindex!(state::RJState, value, i::Int)
    var_name, local_idx = locate_parameter(state, i)
    state.parameters[var_name][local_idx] = value
    return value
end

"""
$SIGNATURES

Create a reference to a specific parameter for in-place modification.
"""
function Base.Ref(state::RJState, i::Int)
    var_name, local_idx = locate_parameter(state, i)
    return Ref(state.parameters[var_name], local_idx)
end

"""
$SIGNATURES

Locate which variable and index corresponds to flattened index i.

Returns `(var_name::Symbol, local_index::Int)`.
"""
function locate_parameter(state::RJState, i::Int)
    @assert 1 <= i <= state.dimension "Index $i out of bounds for dimension $(state.dimension)"

    current_offset = 0

    # Search through continuous variables
    for var_name in continuous_variables(state)
        if haskey(state.parameters, var_name)
            var_length = length(state.parameters[var_name])
            if i <= current_offset + var_length
                local_idx = i - current_offset
                return (var_name, local_idx)
            end
            current_offset += var_length
        end
    end

    # Search through discrete variables
    for var_name in discrete_variables(state)
        if haskey(state.parameters, var_name)
            var_length = length(state.parameters[var_name])
            if i <= current_offset + var_length
                local_idx = i - current_offset
                return (var_name, local_idx)
            end
            current_offset += var_length
        end
    end

    error("Could not locate parameter at index $i")
end
