"""
An explorer wrapper that updates different variables at different periods and offsets.

This explorer wraps a SliceSampler and updates different variables at different rates.
Variables can be updated every N iterations, and offsets allow creating cycling groups
where variables take turns being updated. **Only the variables that need updating are
actually sampled**, making this much more efficient than running a full explorer and
discarding some updates.

This is useful for models with nuisance parameters that have minimal effect on the
posterior but are computationally expensive to sample.

# Fields
- `slice_sampler::SliceSampler`: The underlying slice sampler configuration
- `update_period::Vector{Int}`: Update period for each variable.
  A period of 1 means update every iteration, 10 means update every 10 iterations, etc.
- `update_offset::Vector{Int}`: Offset within the update cycle for each variable.
  Variable i is updated when `scan % period[i] == offset[i]`, where `scan` is from `shared.iterators.scan`.
  This enables cycling through groups of variables.

# Examples
```julia
# Update variables 1-3 every iteration, variable 4 every 10 iterations
explorer = BlockExplorer(
    SliceSampler(),
    update_period = [1, 1, 1, 10]
)

# Cycle through 3 variables, updating only one per iteration
explorer = BlockExplorer(
    SliceSampler(),
    update_period = [3, 3, 3],
    update_offset = [0, 1, 2]
)
# Variable 1 updates at iterations 3, 6, 9, ...
# Variable 2 updates at iterations 1, 4, 7, ...
# Variable 3 updates at iterations 2, 5, 8, ...

# Mixed: always update vars 1-2, update var 3 every 10 iterations, cycle through vars 4-6
explorer = BlockExplorer(
    SliceSampler(),
    update_period = [1, 1, 10, 3, 3, 3],
    update_offset = [0, 0, 0, 0, 1, 2]
)
```

# Implementation Notes
- The explorer tracks iteration counts per replica to handle parallel tempering correctly
- Variables are updated when `iteration % period[i] == offset[i]`
- **Only variables that need updating are sampled** - no wasted computation
- Works with any state type that supports indexing (typically Vector{Float64})
- Specifically designed for SliceSampler to enable coordinate-wise control

# Reference
Designed for use cases similar to the ranking mechanism in:
```julia
transit_priorities ~ MvNormal(zeros(len_epochs), I)
transits = partialsortperm(SVector(transit_priorities), 1:n_transits, rev=true)
```
where the `transit_priorities` are nuisance variables that can be updated less frequently.
"""
struct BlockExplorer
    slice_sampler::SliceSampler
    update_period::Vector{Int}
    update_offset::Vector{Int}
end

"""
    BlockExplorer(slice_sampler::SliceSampler; update_period::Vector{Int}, update_offset::Vector{Int}=zeros(Int, length(update_period)))

Create a block explorer that updates variables at different periods with optional offsets.

# Arguments
- `slice_sampler`: The SliceSampler configuration to use
- `update_period`: Vector specifying update period for each variable.
  Length must match the number of variables in your state.
  Each element should be a positive integer where:
  - 1 = update every iteration
  - n = update every n iterations
- `update_offset`: Vector specifying the offset within each variable's update cycle (default: all zeros).
  Variable i is updated when `iteration % period[i] == offset[i]`.
  Offsets must satisfy `0 <= offset[i] < period[i]`.

# Examples
```julia
# Basic: update variables at different periods
explorer = BlockExplorer(
    SliceSampler(n_passes=3),
    update_period = [1, 1, 1, 10]  # Vars 1-3 every iter, var 4 every 10
)

# Cycling: update one variable from a group per iteration
explorer = BlockExplorer(
    SliceSampler(),
    update_period = [3, 3, 3],
    update_offset = [0, 1, 2]  # Each variable updated at different points in 3-iter cycle
)

# Mixed: some always, some periodic, some cycling
periods = fill(1, 80)
offsets = fill(0, 80)
periods[4:end] .= 10  # Update last 77 variables every 10 iterations
explorer = BlockExplorer(
    SliceSampler(),
    update_period = periods,
    update_offset = offsets
)

# Two separate cycling groups with different periods
explorer = BlockExplorer(
    SliceSampler(),
    update_period = [5, 5, 5, 5, 5,  3, 3, 3],
    update_offset =    [0, 1, 2, 3, 4,  0, 1, 2]
)
```
"""
function BlockExplorer(slice_sampler::SliceSampler;
                       update_period::Vector{Int},
                       update_offset::Vector{Int}=zeros(Int, length(update_period)))
    # Validate update periods
    if any(f <= 0 for f in update_period)
        error("All update periods must be positive integers. Got: $update_period")
    end
    if isempty(update_period)
        error("update_period vector cannot be empty")
    end

    # Validate update offsets
    if length(update_offset) != length(update_period)
        error("update_offset length ($(length(update_offset))) must match update_period length ($(length(update_period)))")
    end
    if any(update_offset .< 0)
        error("All update offsets must be non-negative. Got: $update_offset")
    end
    if any(update_offset .>= update_period)
        invalid_pairs = [(i, update_offset[i], update_period[i]) for i in 1:length(update_offset) if update_offset[i] >= update_period[i]]
        error("All offsets must be less than their corresponding periods. Invalid: $invalid_pairs")
    end

    return BlockExplorer(
        slice_sampler,
        update_period,
        update_offset
    )
end

"""
    step!(explorer::BlockExplorer, replica, shared)

Perform one exploration step with block-based updates.

This function:
1. Gets current scan number from shared.iterators.scan (thread-safe)
2. Determines which variables should be updated based on scan, period, and offset
3. For each pass in n_passes:
   - Only samples coordinates that should be updated this scan
   - Skips coordinates that shouldn't be updated (no wasted work!)
4. Returns updated state

Variables are updated when `scan % period[i] == offset[i]`, enabling both
periodic updates and cycling patterns.

# Thread Safety
This implementation is thread-safe because:
- The explorer struct is immutable (no mutable state)
- We read from shared.iterators.scan (managed by framework)
- Each replica modifies only its own state
"""
function step!(explorer::BlockExplorer, replica, shared)
    # Get current scan number (thread-safe read from framework-managed state)
    current_scan = shared.iterators.scan

    # Validate that state length matches update_period length
    state_length = length(replica.state)
    period_length = length(explorer.update_period)
    if state_length != period_length
        error("""
        State length ($state_length) does not match update_period length ($period_length).
        Make sure your update_period vector has one entry per state variable.
        """)
    end

    # Get log potential
    log_potential = find_log_potential(replica, shared.tempering, shared)
    h = explorer.slice_sampler
    state = replica.state

    # Cached log potential - reused across coordinates
    cached_lp = -Inf

    # Perform n_passes, but only update coordinates that should be updated
    for _ in 1:h.n_passes
        # Compute cached log potential if needed
        cached_lp = cached_log_potential(log_potential, replica.state, cached_lp)

        # Iterate over coordinates, but only sample those that should be updated
        for c in eachindex(state)
            # Check if this coordinate should be updated this scan
            should_update = (current_scan % explorer.update_period[c] == explorer.update_offset[c])

            if should_update
                # Sample this coordinate
                pointer = Ref(state, c)
                cached_lp = slice_sample_coord!(h, replica, pointer, log_potential, cached_lp, typeof(pointer[]))

                # Check we still have a healthy state
                if !isfinite(cached_lp)
                    error("""Got an invalid log density after updating state at index $c:
                    - log density = $cached_lp
                    - state[$c]   = $(pointer[])
                    Dumping full replica state:
                    $(replica.state)
                    """)
                end
            end
            # If should_update is false, we skip this coordinate entirely - no work done!
        end
    end
end

# Forward adapter function
function adapt_explorer(explorer::BlockExplorer, reduced_recorders, current_pt, new_tempering)
    # Adapt the underlying slice sampler (though SliceSampler doesn't adapt by default)
    adapted_slice = adapt_explorer(explorer.slice_sampler, reduced_recorders, current_pt, new_tempering)
    return BlockExplorer(
        adapted_slice,
        update_period = explorer.update_period,
        update_offset = explorer.update_offset
    )
end

# Forward recorder builders to slice sampler
explorer_recorder_builders(explorer::BlockExplorer) =
    explorer_recorder_builders(explorer.slice_sampler)
