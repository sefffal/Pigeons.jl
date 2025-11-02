# Example: Using BlockExplorer for Models with Nuisance Parameters
#
# This example demonstrates how to use the BlockExplorer to reduce
# computational cost in models with many nuisance parameters that have minimal
# effect on the posterior.

using Pigeons
using Distributions
using LinearAlgebra

# Example 1: Simple Gaussian model with some nuisance variables
# ================================================================

function example_simple_gaussian()
    println("\n=== Example 1: Simple Gaussian Model ===")

    # Target: Standard normal in 10 dimensions
    # Suppose variables 1-3 are important, variables 4-10 are nuisance
    dim = 10
    log_potential(x) = -0.5 * sum(x.^2)

    # Create explorer that updates:
    # - Variables 1-3 every iteration (frequency = 1)
    # - Variables 4-10 every 10 iterations (frequency = 10)
    frequencies = vcat(fill(1, 3), fill(10, 7))

    explorer = BlockExplorer(
        SliceSampler(n_passes=2),
        update_frequency = frequencies
    )

    pt = pigeons(
        target = log_potential,
        explorer = explorer,
        n_rounds = 8,
        n_chains = 10
    )

    samples = pt.reduced_recorders.online
    println("Mean: ", mean(samples))
    println("Standard deviation: ", std(samples))

    return pt
end

# Example 2: Transit priorities use case
# =======================================
# This mimics the original use case with ranking nuisance variables

function example_transit_priorities()
    println("\n=== Example 2: Transit Priorities Use Case ===")

    # Simulate a model with:
    # - 3 important astronomical parameters (e.g., position, velocity, parallax)
    # - 77 nuisance transit_priority parameters
    n_important = 3
    n_transits = 77
    dim = n_important + n_transits

    # Simple target that includes the ranking mechanism
    function log_potential(x)
        # Important parameters have tighter priors
        lp = -0.5 * sum(x[1:n_important].^2)

        # Transit priorities have standard normal priors
        transit_priorities = x[(n_important+1):end]
        lp += -0.5 * sum(transit_priorities.^2)

        # The ranking itself (this is what gets used in the actual model)
        # In practice, you would use: transits = partialsortperm(transit_priorities, 1:n_matched, rev=true)
        # For this example, we just ensure they're used in the likelihood
        lp += -0.01 * sum(transit_priorities)  # Weak coupling

        return lp
    end

    # Update important parameters every iteration,
    # transit priorities every 10 iterations
    frequencies = vcat(fill(1, n_important), fill(10, n_transits))

    explorer = BlockExplorer(
        SliceSampler(n_passes=3),
        update_frequency = frequencies
    )

    pt = pigeons(
        target = log_potential,
        explorer = explorer,
        n_rounds = 10,
        n_chains = 20
    )

    samples = pt.reduced_recorders.online
    println("Number of dimensions: ", length(mean(samples)))
    println("Mean of important parameters: ", mean(samples)[1:n_important])
    println("Mean of first 5 transit priorities: ", mean(samples)[(n_important+1):(n_important+5)])

    return pt
end

# Example 3: Composing with other explorers
# ==========================================

function example_composition()
    println("\n=== Example 3: Composing Explorers ===")

    dim = 6
    log_potential(x) = -0.5 * sum(x.^2)

    # Create a reduced-frequency slice sampler
    reduced_slice = BlockExplorer(
        SliceSampler(n_passes=1),
        update_frequency = [1, 1, 5, 5, 10, 10]
    )

    # Compose with another slice sampler pass
    explorer = Compose(reduced_slice, SliceSampler(n_passes=1))

    pt = pigeons(
        target = log_potential,
        explorer = explorer,
        n_rounds = 6,
        n_chains = 8
    )

    println("Composition works! Mean: ", mean(pt.reduced_recorders.online))

    return pt
end

# Example 4: Different update patterns
# =====================================

function example_varied_frequencies()
    println("\n=== Example 4: Varied Update Frequencies ===")

    dim = 8
    log_potential(x) = -0.5 * sum(x.^2)

    # Different variables updated at different rates:
    # Variables 1-2: every iteration (1)
    # Variables 3-4: every 2 iterations (2)
    # Variables 5-6: every 5 iterations (5)
    # Variables 7-8: every 10 iterations (10)
    frequencies = [1, 1, 2, 2, 5, 5, 10, 10]

    explorer = BlockExplorer(
        SliceSampler(n_passes=2),
        update_frequency = frequencies
    )

    pt = pigeons(
        target = log_potential,
        explorer = explorer,
        n_rounds = 7,
        n_chains = 12
    )

    println("Varied frequencies work! Mean: ", mean(pt.reduced_recorders.online))

    return pt
end

# Example 5: Cycling through variables
# =====================================

function example_cycling()
    println("\n=== Example 5: Cycling Through Variables ===")

    dim = 6
    log_potential(x) = -0.5 * sum(x.^2)

    # Cycle through 3 variables, updating only one per iteration
    # This is useful when you have expensive likelihood evaluations
    # and want to update variables one at a time
    explorer = BlockExplorer(
        SliceSampler(n_passes=1),
        update_frequency = [3, 3, 3, 1, 1, 1],
        update_offset = [0, 1, 2, 0, 0, 0]
    )
    # Variables 1-3 cycle (one per iteration)
    # Variables 4-6 always update
    # At iter 1: vars 2, 4, 5, 6 update
    # At iter 2: vars 3, 4, 5, 6 update
    # At iter 3: vars 1, 4, 5, 6 update
    # At iter 4: vars 2, 4, 5, 6 update (pattern repeats)

    pt = pigeons(
        target = log_potential,
        explorer = explorer,
        n_rounds = 6,
        n_chains = 8
    )

    println("Cycling works! Mean: ", mean(pt.reduced_recorders.online))

    return pt
end

# Example 6: Multiple cycling groups
# ===================================

function example_multiple_cycling_groups()
    println("\n=== Example 6: Multiple Cycling Groups ===")

    dim = 8
    log_potential(x) = -0.5 * sum(x.^2)

    # Two separate cycling groups with different periods
    # Group 1: Variables 1-5 cycle every 5 iterations (one var per iter)
    # Group 2: Variables 6-8 cycle every 3 iterations (one var per iter)
    explorer = BlockExplorer(
        SliceSampler(n_passes=1),
        update_frequency = [5, 5, 5, 5, 5,  3, 3, 3],
        update_offset =    [0, 1, 2, 3, 4,  0, 1, 2]
    )

    pt = pigeons(
        target = log_potential,
        explorer = explorer,
        n_rounds = 7,
        n_chains = 10
    )

    println("Multiple cycling groups work! Mean: ", mean(pt.reduced_recorders.online))

    return pt
end

# Example 7: Practical cycling for expensive models
# ==================================================

function example_practical_cycling()
    println("\n=== Example 7: Practical Cycling for Expensive Models ===")

    # Simulate a model where:
    # - Variables 1-3 are critical parameters (always update)
    # - Variables 4-80 are nuisance parameters
    # - Updating all nuisance params at once is expensive
    # - So we cycle through them in groups of 10

    n_critical = 3
    n_nuisance = 77
    dim = n_critical + n_nuisance

    log_potential(x) = -0.5 * sum(x.^2)

    # Create cycling groups for nuisance parameters
    # Split 77 nuisance variables into groups that cycle
    group_size = 10
    n_full_groups = div(n_nuisance, group_size)  # 7 groups
    remainder = mod(n_nuisance, group_size)      # 7 remaining

    frequencies = fill(1, n_critical)  # Critical params: always
    offsets = fill(0, n_critical)

    # Add cycling groups
    for group in 0:(n_full_groups-1)
        append!(frequencies, fill(group_size, group_size))
        append!(offsets, 0:(group_size-1))
    end

    # Add remainder with shorter cycle
    if remainder > 0
        append!(frequencies, fill(remainder, remainder))
        append!(offsets, 0:(remainder-1))
    end

    explorer = BlockExplorer(
        SliceSampler(n_passes=2),
        update_frequency = frequencies,
        update_offset = offsets
    )

    println("Created explorer with $(length(frequencies)) variables")
    println("Critical params (always updated): $n_critical")
    println("Nuisance params (cycling): $n_nuisance")
    println("Effective updates per iteration: $n_critical + ~$(div(n_nuisance, group_size))")

    pt = pigeons(
        target = log_potential,
        explorer = explorer,
        n_rounds = 8,
        n_chains = 12
    )

    println("Practical cycling works! Mean of critical params: ",
            mean(pt.reduced_recorders.online)[1:n_critical])

    return pt
end

# Main: Run all examples
# ======================

function run_all_examples()
    println("Running BlockExplorer Examples")
    println("==========================================")

    example_simple_gaussian()
    example_transit_priorities()
    example_composition()
    example_varied_frequencies()
    example_cycling()
    example_multiple_cycling_groups()
    example_practical_cycling()

    println("\n✓ All examples completed successfully!")
end

# To run these examples:
# julia --project=. examples/block_explorer_example.jl
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_examples()
end
