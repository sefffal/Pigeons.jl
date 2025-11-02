# Example: Using ReducedFrequencyExplorer for Models with Nuisance Parameters
#
# This example demonstrates how to use the ReducedFrequencyExplorer to reduce
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

    explorer = ReducedFrequencyExplorer(
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

    explorer = ReducedFrequencyExplorer(
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
    reduced_slice = ReducedFrequencyExplorer(
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

    explorer = ReducedFrequencyExplorer(
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

# Main: Run all examples
# ======================

function run_all_examples()
    println("Running ReducedFrequencyExplorer Examples")
    println("==========================================")

    example_simple_gaussian()
    example_transit_priorities()
    example_composition()
    example_varied_frequencies()

    println("\n✓ All examples completed successfully!")
end

# To run these examples:
# julia --project=. examples/reduced_frequency_explorer_example.jl
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_examples()
end
