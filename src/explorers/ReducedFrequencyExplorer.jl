"""
An explorer wrapper that updates different variables at different frequencies.

This explorer wraps a base explorer (typically SliceSampler) and updates different
variables at different rates. Variables with higher update frequencies are sampled
every iteration, while variables with lower frequencies are updated less often.

This is useful for models with nuisance parameters that have minimal effect on the
posterior but are computationally expensive to sample.

# Fields
- `base_explorer`: The underlying explorer to wrap (e.g., SliceSampler())
- `update_frequency::Vector{Int}`: Update frequency for each variable.
  A frequency of 1 means update every iteration, 10 means update every 10 iterations, etc.
- `iteration_counts::Dict{Int, Int}`: Internal state tracking iterations per replica

# Example
```julia
# Update variables 1-3 every iteration, variable 4 every 10 iterations
explorer = ReducedFrequencyExplorer(
    SliceSampler(),
    update_frequency = [1, 1, 1, 10]
)

pt = pigeons(
    target = my_target,
    explorer = explorer,
    n_rounds = 10
)
```

# Implementation Notes
- The explorer tracks iteration counts per replica to handle parallel tempering correctly
- Variables are updated when `iteration_count % frequency == 0`
- The base explorer is called on all variables, then non-updated variables are restored
- Works with any state type that supports indexing and copying (typically Vector{Float64})
- Can be composed with other explorers using `Compose()`

# Reference
Designed for use cases similar to the ranking mechanism in:
```julia
transit_priorities ~ MvNormal(zeros(len_epochs), I)
transits = partialsortperm(SVector(transit_priorities), 1:n_transits, rev=true)
```
where the `transit_priorities` are nuisance variables that can be updated less frequently.
"""
mutable struct ReducedFrequencyExplorer{E}
    base_explorer::E
    update_frequency::Vector{Int}
    iteration_counts::Dict{Int, Int}
end

"""
    ReducedFrequencyExplorer(base_explorer; update_frequency::Vector{Int})

Create a reduced-frequency explorer that updates variables at different rates.

# Arguments
- `base_explorer`: The base explorer to wrap (e.g., `SliceSampler()`)
- `update_frequency`: Vector specifying update frequency for each variable.
  Length must match the number of variables in your state.
  Each element should be a positive integer where:
  - 1 = update every iteration
  - n = update every n iterations

# Example
```julia
explorer = ReducedFrequencyExplorer(
    SliceSampler(n_passes=3),
    update_frequency = fill(1, 77)  # all variables updated every iteration by default
)

# Or selectively reduce frequency for some variables:
frequencies = fill(1, 80)
frequencies[4:end] .= 10  # Update last 77 variables every 10 iterations
explorer = ReducedFrequencyExplorer(
    SliceSampler(),
    update_frequency = frequencies
)
```
"""
function ReducedFrequencyExplorer(base_explorer::E; update_frequency::Vector{Int}) where E
    # Validate update frequencies
    if any(f <= 0 for f in update_frequency)
        error("All update frequencies must be positive integers. Got: $update_frequency")
    end
    if isempty(update_frequency)
        error("update_frequency vector cannot be empty")
    end

    return ReducedFrequencyExplorer{E}(
        base_explorer,
        update_frequency,
        Dict{Int, Int}()
    )
end

"""
    step!(explorer::ReducedFrequencyExplorer, replica, shared)

Perform one exploration step with reduced-frequency updates.

This function:
1. Increments the iteration counter for this replica
2. Determines which variables should be updated based on iteration count and frequency
3. Saves the state of variables that should NOT be updated
4. Calls the base explorer's step! function (updates all variables)
5. Restores the saved values for variables that should not have been updated
"""
function step!(explorer::ReducedFrequencyExplorer, replica, shared)
    # Initialize or increment iteration count for this replica
    replica_id = replica.replica_index
    if !haskey(explorer.iteration_counts, replica_id)
        explorer.iteration_counts[replica_id] = 0
    end
    explorer.iteration_counts[replica_id] += 1
    current_iter = explorer.iteration_counts[replica_id]

    # Validate that state length matches update_frequency length
    state_length = length(replica.state)
    freq_length = length(explorer.update_frequency)
    if state_length != freq_length
        error("""
        State length ($state_length) does not match update_frequency length ($freq_length).
        Make sure your update_frequency vector has one entry per state variable.
        """)
    end

    # Determine which variables should be updated this iteration
    # Variable i is updated when current_iter % update_frequency[i] == 0
    should_update = [current_iter % freq == 0 for freq in explorer.update_frequency]

    # If all variables should be updated, just call the base explorer directly
    if all(should_update)
        step!(explorer.base_explorer, replica, shared)
        return
    end

    # Save values of variables that should NOT be updated
    saved_values = similar(replica.state)
    for i in eachindex(replica.state)
        if !should_update[i]
            saved_values[i] = replica.state[i]
        end
    end

    # Run base explorer (this will update all variables)
    step!(explorer.base_explorer, replica, shared)

    # Restore variables that should not have been updated
    for i in eachindex(replica.state)
        if !should_update[i]
            replica.state[i] = saved_values[i]
        end
    end
end

# Forward adapter function to base explorer
function adapt_explorer(explorer::ReducedFrequencyExplorer, reduced_recorders, current_pt, new_tempering)
    adapted_base = adapt_explorer(explorer.base_explorer, reduced_recorders, current_pt, new_tempering)
    return ReducedFrequencyExplorer(
        adapted_base,
        explorer.update_frequency,
        explorer.iteration_counts  # Preserve iteration counts across adaptation
    )
end

# Forward recorder builders to base explorer
explorer_recorder_builders(explorer::ReducedFrequencyExplorer) =
    explorer_recorder_builders(explorer.base_explorer)
