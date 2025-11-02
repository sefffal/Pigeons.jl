"""
An explorer wrapper that updates different variables at different frequencies and offsets.

This explorer wraps a base explorer (typically SliceSampler) and updates different
variables at different rates. Variables can be updated every N iterations, and offsets
allow creating cycling groups where variables take turns being updated.

This is useful for models with nuisance parameters that have minimal effect on the
posterior but are computationally expensive to sample.

# Fields
- `base_explorer`: The underlying explorer to wrap (e.g., SliceSampler())
- `update_frequency::Vector{Int}`: Update frequency for each variable.
  A frequency of 1 means update every iteration, 10 means update every 10 iterations, etc.
- `update_offset::Vector{Int}`: Offset within the update cycle for each variable.
  Variable i is updated when `iteration % frequency[i] == offset[i]`.
  This enables cycling through groups of variables.
- `iteration_counts::Dict{Int, Int}`: Internal state tracking iterations per replica

# Examples
```julia
# Update variables 1-3 every iteration, variable 4 every 10 iterations
explorer = BlockExplorer(
    SliceSampler(),
    update_frequency = [1, 1, 1, 10]
)

# Cycle through 3 variables, updating only one per iteration
explorer = BlockExplorer(
    SliceSampler(),
    update_frequency = [3, 3, 3],
    update_offset = [0, 1, 2]
)
# Variable 1 updates at iterations 3, 6, 9, ...
# Variable 2 updates at iterations 1, 4, 7, ...
# Variable 3 updates at iterations 2, 5, 8, ...

# Mixed: always update vars 1-2, update var 3 every 10 iterations, cycle through vars 4-6
explorer = BlockExplorer(
    SliceSampler(),
    update_frequency = [1, 1, 10, 3, 3, 3],
    update_offset = [0, 0, 0, 0, 1, 2]
)
```

# Implementation Notes
- The explorer tracks iteration counts per replica to handle parallel tempering correctly
- Variables are updated when `iteration % frequency[i] == offset[i]`
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
mutable struct BlockExplorer{E}
    base_explorer::E
    update_frequency::Vector{Int}
    update_offset::Vector{Int}
    iteration_counts::Dict{Int, Int}
end

"""
    BlockExplorer(base_explorer; update_frequency::Vector{Int}, update_offset::Vector{Int}=zeros(Int, length(update_frequency)))

Create a reduced-frequency explorer that updates variables at different rates with optional offsets.

# Arguments
- `base_explorer`: The base explorer to wrap (e.g., `SliceSampler()`)
- `update_frequency`: Vector specifying update frequency for each variable.
  Length must match the number of variables in your state.
  Each element should be a positive integer where:
  - 1 = update every iteration
  - n = update every n iterations
- `update_offset`: Vector specifying the offset within each variable's update cycle (default: all zeros).
  Variable i is updated when `iteration % frequency[i] == offset[i]`.
  Offsets must satisfy `0 <= offset[i] < frequency[i]`.

# Examples
```julia
# Basic: update variables at different frequencies
explorer = BlockExplorer(
    SliceSampler(n_passes=3),
    update_frequency = [1, 1, 1, 10]  # Vars 1-3 every iter, var 4 every 10
)

# Cycling: update one variable from a group per iteration
explorer = BlockExplorer(
    SliceSampler(),
    update_frequency = [3, 3, 3],
    update_offset = [0, 1, 2]  # Each variable updated at different points in 3-iter cycle
)

# Mixed: some always, some periodic, some cycling
frequencies = fill(1, 80)
offsets = fill(0, 80)
frequencies[4:end] .= 10  # Update last 77 variables every 10 iterations
explorer = BlockExplorer(
    SliceSampler(),
    update_frequency = frequencies,
    update_offset = offsets
)

# Two separate cycling groups with different periods
explorer = BlockExplorer(
    SliceSampler(),
    update_frequency = [5, 5, 5, 5, 5,  3, 3, 3],
    update_offset =    [0, 1, 2, 3, 4,  0, 1, 2]
)
```
"""
function BlockExplorer(base_explorer::E;
                                   update_frequency::Vector{Int},
                                   update_offset::Vector{Int}=zeros(Int, length(update_frequency))) where E
    # Validate update frequencies
    if any(f <= 0 for f in update_frequency)
        error("All update frequencies must be positive integers. Got: $update_frequency")
    end
    if isempty(update_frequency)
        error("update_frequency vector cannot be empty")
    end

    # Validate update offsets
    if length(update_offset) != length(update_frequency)
        error("update_offset length ($(length(update_offset))) must match update_frequency length ($(length(update_frequency)))")
    end
    if any(update_offset .< 0)
        error("All update offsets must be non-negative. Got: $update_offset")
    end
    if any(update_offset .>= update_frequency)
        invalid_pairs = [(i, update_offset[i], update_frequency[i]) for i in 1:length(update_offset) if update_offset[i] >= update_frequency[i]]
        error("All offsets must be less than their corresponding frequencies. Invalid: $invalid_pairs")
    end

    return BlockExplorer{E}(
        base_explorer,
        update_frequency,
        update_offset,
        Dict{Int, Int}()
    )
end

"""
    step!(explorer::BlockExplorer, replica, shared)

Perform one exploration step with reduced-frequency updates.

This function:
1. Increments the iteration counter for this replica
2. Determines which variables should be updated based on iteration count, frequency, and offset
3. Saves the state of variables that should NOT be updated
4. Calls the base explorer's step! function (updates all variables)
5. Restores the saved values for variables that should not have been updated

Variables are updated when `iteration % frequency[i] == offset[i]`, enabling both
periodic updates and cycling patterns.
"""
function step!(explorer::BlockExplorer, replica, shared)
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
    # Variable i is updated when current_iter % update_frequency[i] == update_offset[i]
    should_update = [current_iter % explorer.update_frequency[i] == explorer.update_offset[i]
                     for i in eachindex(explorer.update_frequency)]

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
function adapt_explorer(explorer::BlockExplorer, reduced_recorders, current_pt, new_tempering)
    adapted_base = adapt_explorer(explorer.base_explorer, reduced_recorders, current_pt, new_tempering)
    return BlockExplorer(
        adapted_base,
        update_frequency = explorer.update_frequency,
        update_offset = explorer.update_offset
        # Note: iteration_counts will be reset to empty Dict{Int,Int}() by constructor
        # This is intentional as adaptation happens between rounds
    )
end

# Forward recorder builders to base explorer
explorer_recorder_builders(explorer::BlockExplorer) =
    explorer_recorder_builders(explorer.base_explorer)
