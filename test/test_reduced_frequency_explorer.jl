import Pigeons: ReducedFrequencyExplorer, SliceSampler, Compose, step!, Replica
using Distributions
using LinearAlgebra

@testset "ReducedFrequencyExplorer - Constructor" begin
    # Test basic construction
    explorer = ReducedFrequencyExplorer(
        SliceSampler(),
        update_frequency = [1, 1, 10]
    )
    @test length(explorer.update_frequency) == 3
    @test explorer.update_frequency == [1, 1, 10]
    @test isempty(explorer.iteration_counts)

    # Test error on invalid frequencies
    @test_throws ErrorException ReducedFrequencyExplorer(
        SliceSampler(),
        update_frequency = [1, 0, 10]  # 0 is invalid
    )

    @test_throws ErrorException ReducedFrequencyExplorer(
        SliceSampler(),
        update_frequency = [1, -1, 10]  # negative is invalid
    )

    @test_throws ErrorException ReducedFrequencyExplorer(
        SliceSampler(),
        update_frequency = Int[]  # empty is invalid
    )
end

@testset "ReducedFrequencyExplorer - Frequency Logic" begin
    # Test that variables are updated at correct frequencies
    rng = SplittableRandom(42)
    log_potential(x) = -0.5 * sum(x.^2)  # Standard normal

    # Create explorer with different frequencies
    explorer = ReducedFrequencyExplorer(
        SliceSampler(n_passes=1),
        update_frequency = [1, 1, 5]  # Last variable updated every 5 iterations
    )

    state = [0.0, 0.0, 0.0]
    shared = (
        tempering = Pigeons.NonReversiblePT(
            [log_potential],
            Pigeons.Geometric(0.5)
        ),
        iterators = (scan = 1,),
        explorer = explorer,
        reports = (;)
    )

    # Track third variable across iterations
    third_values = Float64[]

    for iter in 1:10
        replica = Replica(state, 1, rng, (;), 1)
        Pigeons.step!(explorer, replica, shared)
        push!(third_values, state[3])
    end

    # Check that third variable changes at iterations 5 and 10
    # but not at 1, 2, 3, 4, 6, 7, 8, 9
    @test third_values[1] == third_values[2] == third_values[3] == third_values[4]
    @test third_values[5] != third_values[4]  # Should change at iteration 5
    @test third_values[6] == third_values[7] == third_values[8] == third_values[9]
    @test third_values[10] != third_values[9]  # Should change at iteration 10
end

@testset "ReducedFrequencyExplorer - Integration with pigeons" begin
    # Simple Gaussian target
    log_potential(x) = -0.5 * sum(x.^2)

    # Test with all variables at frequency 1 (should work like normal SliceSampler)
    explorer1 = ReducedFrequencyExplorer(
        SliceSampler(n_passes=1),
        update_frequency = [1, 1, 1, 1]
    )

    pt1 = pigeons(
        target = log_potential,
        explorer = explorer1,
        n_rounds = 5,
        n_chains = 4
    )

    # Should produce reasonable samples
    samples1 = pt1.reduced_recorders.online
    @test abs(mean(samples1)[1]) < 0.3  # Mean should be close to 0

    # Test with mixed frequencies
    explorer2 = ReducedFrequencyExplorer(
        SliceSampler(n_passes=1),
        update_frequency = [1, 1, 1, 10]
    )

    pt2 = pigeons(
        target = log_potential,
        explorer = explorer2,
        n_rounds = 5,
        n_chains = 4
    )

    samples2 = pt2.reduced_recorders.online
    @test abs(mean(samples2)[1]) < 0.5
end

@testset "ReducedFrequencyExplorer - Composability" begin
    # Test that ReducedFrequencyExplorer can be composed with other explorers
    log_potential(x) = -0.5 * sum(x.^2)

    base_explorer = SliceSampler(n_passes=1)
    reduced_explorer = ReducedFrequencyExplorer(
        base_explorer,
        update_frequency = [1, 1, 1, 5]
    )

    # Compose should work
    composed = Compose(reduced_explorer, SliceSampler(n_passes=1))

    # Should run without errors
    pt = pigeons(
        target = log_potential,
        explorer = composed,
        n_rounds = 3,
        n_chains = 2
    )

    @test pt isa Pigeons.PT
end

@testset "ReducedFrequencyExplorer - State Length Validation" begin
    # Test that state length must match update_frequency length
    rng = SplittableRandom(42)
    log_potential(x) = -0.5 * sum(x.^2)

    explorer = ReducedFrequencyExplorer(
        SliceSampler(n_passes=1),
        update_frequency = [1, 1, 10]  # 3 variables
    )

    state = [0.0, 0.0, 0.0, 0.0]  # 4 variables - mismatch!
    shared = (
        tempering = Pigeons.NonReversiblePT(
            [log_potential],
            Pigeons.Geometric(0.5)
        ),
        iterators = (scan = 1,),
        explorer = explorer,
        reports = (;)
    )

    replica = Replica(state, 1, rng, (;), 1)

    # Should error due to length mismatch
    @test_throws ErrorException Pigeons.step!(explorer, replica, shared)
end

@testset "ReducedFrequencyExplorer - Multiple Replicas" begin
    # Test that iteration counts are tracked per replica
    rng1 = SplittableRandom(1)
    rng2 = SplittableRandom(2)
    log_potential(x) = -0.5 * sum(x.^2)

    explorer = ReducedFrequencyExplorer(
        SliceSampler(n_passes=1),
        update_frequency = [1, 10]
    )

    state1 = [0.0, 0.0]
    state2 = [0.0, 0.0]

    shared = (
        tempering = Pigeons.NonReversiblePT(
            [log_potential],
            Pigeons.Geometric(0.5)
        ),
        iterators = (scan = 1,),
        explorer = explorer,
        reports = (;)
    )

    # Run replica 1 for 5 iterations
    for _ in 1:5
        replica1 = Replica(state1, 1, rng1, (;), 1)
        Pigeons.step!(explorer, replica1, shared)
    end

    # Run replica 2 for 3 iterations
    for _ in 1:3
        replica2 = Replica(state2, 1, rng2, (;), 2)
        Pigeons.step!(explorer, replica2, shared)
    end

    # Check that iteration counts are tracked separately
    @test explorer.iteration_counts[1] == 5
    @test explorer.iteration_counts[2] == 3
end

@testset "ReducedFrequencyExplorer - All Variables Skip Case" begin
    # Test edge case where all variables should be updated
    rng = SplittableRandom(42)
    log_potential(x) = -0.5 * sum(x.^2)

    explorer = ReducedFrequencyExplorer(
        SliceSampler(n_passes=1),
        update_frequency = [10, 10, 10]
    )

    state = [1.0, 2.0, 3.0]
    initial_state = copy(state)

    shared = (
        tempering = Pigeons.NonReversiblePT(
            [log_potential],
            Pigeons.Geometric(0.5)
        ),
        iterators = (scan = 1,),
        explorer = explorer,
        reports = (;)
    )

    # Iteration 1-9: no updates
    for iter in 1:9
        replica = Replica(state, 1, rng, (;), 1)
        Pigeons.step!(explorer, replica, shared)
        # State should not change
        @test state == initial_state
    end

    # Iteration 10: all variables updated
    replica = Replica(state, 1, rng, (;), 1)
    Pigeons.step!(explorer, replica, shared)
    # State should have changed (with very high probability)
    @test state != initial_state
end

@testset "ReducedFrequencyExplorer - Typical Use Case" begin
    # Simulate the actual use case: many nuisance variables
    n_important = 3
    n_nuisance = 77
    dim = n_important + n_nuisance

    # Simple target
    log_potential(x) = -0.5 * sum(x.^2)

    # Update important variables every iteration, nuisance every 10
    frequencies = vcat(fill(1, n_important), fill(10, n_nuisance))

    explorer = ReducedFrequencyExplorer(
        SliceSampler(n_passes=1),
        update_frequency = frequencies
    )

    # Should run without errors
    pt = pigeons(
        target = log_potential,
        explorer = explorer,
        n_rounds = 5,
        n_chains = 4
    )

    @test pt isa Pigeons.PT
    samples = pt.reduced_recorders.online

    # Check that we get reasonable results
    @test length(mean(samples)) == dim
    @test all(abs.(mean(samples)) .< 0.5)  # Means should be close to 0
end
