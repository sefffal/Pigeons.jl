import Pigeons: BlockExplorer, SliceSampler, step!, Replica
using Distributions
using LinearAlgebra

@testset "BlockExplorer - Constructor" begin
    # Test basic construction with default offsets
    explorer = BlockExplorer(
        SliceSampler(),
        update_period = [1, 1, 10]
    )
    @test length(explorer.update_period) == 3
    @test explorer.update_period == [1, 1, 10]
    @test explorer.update_offset == [0, 0, 0]  # Default offsets

    # Test construction with explicit offsets
    explorer2 = BlockExplorer(
        SliceSampler(),
        update_period = [3, 3, 3],
        update_offset = [0, 1, 2]
    )
    @test explorer2.update_period == [3, 3, 3]
    @test explorer2.update_offset == [0, 1, 2]

    # Test error on invalid frequencies
    @test_throws ErrorException BlockExplorer(
        SliceSampler(),
        update_period = [1, 0, 10]  # 0 is invalid
    )

    @test_throws ErrorException BlockExplorer(
        SliceSampler(),
        update_period = [1, -1, 10]  # negative is invalid
    )

    @test_throws ErrorException BlockExplorer(
        SliceSampler(),
        update_period = Int[]  # empty is invalid
    )

    # Test error on invalid offsets
    @test_throws ErrorException BlockExplorer(
        SliceSampler(),
        update_period = [3, 3, 3],
        update_offset = [0, 1]  # Length mismatch
    )

    @test_throws ErrorException BlockExplorer(
        SliceSampler(),
        update_period = [3, 3, 3],
        update_offset = [0, -1, 2]  # Negative offset
    )

    @test_throws ErrorException BlockExplorer(
        SliceSampler(),
        update_period = [3, 3, 3],
        update_offset = [0, 1, 3]  # Offset >= frequency
    )
end

@testset "BlockExplorer - Frequency Logic" begin
    # Test that variables are updated at correct frequencies
    rng = SplittableRandom(42)
    log_potential(x) = -0.5 * sum(x.^2)  # Standard normal

    # Create explorer with different frequencies
    explorer = BlockExplorer(
        SliceSampler(n_passes=1),
        update_period = [1, 1, 5]  # Last variable updated every 5 iterations
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

@testset "BlockExplorer - Integration with pigeons" begin
    # Simple Gaussian target
    log_potential(x) = -0.5 * sum(x.^2)

    # Test with all variables at frequency 1 (should work like normal SliceSampler)
    explorer1 = BlockExplorer(
        SliceSampler(n_passes=1),
        update_period = [1, 1, 1, 1]
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
    explorer2 = BlockExplorer(
        SliceSampler(n_passes=1),
        update_period = [1, 1, 1, 10]
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

@testset "BlockExplorer - State Length Validation" begin
    # Test that state length must match update_period length
    rng = SplittableRandom(42)
    log_potential(x) = -0.5 * sum(x.^2)

    explorer = BlockExplorer(
        SliceSampler(n_passes=1),
        update_period = [1, 1, 10]  # 3 variables
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

@testset "BlockExplorer - Typical Use Case" begin
    # Simulate the actual use case: many nuisance variables
    n_important = 3
    n_nuisance = 77
    dim = n_important + n_nuisance

    # Simple target
    log_potential(x) = -0.5 * sum(x.^2)

    # Update important variables every iteration, nuisance every 10
    frequencies = vcat(fill(1, n_important), fill(10, n_nuisance))

    explorer = BlockExplorer(
        SliceSampler(n_passes=1),
        update_period = frequencies
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

@testset "BlockExplorer - Cycling Behavior" begin
    # Test cycling through variables with offsets
    rng = SplittableRandom(42)
    log_potential(x) = -0.5 * sum(x.^2)

    # Cycle through 3 variables, one per iteration
    explorer = BlockExplorer(
        SliceSampler(n_passes=1),
        update_period = [3, 3, 3],
        update_offset = [0, 1, 2]
    )

    state = [1.0, 2.0, 3.0]
    shared = (
        tempering = Pigeons.NonReversiblePT(
            [log_potential],
            Pigeons.Geometric(0.5)
        ),
        iterators = (scan = 1,),
        explorer = explorer,
        reports = (;)
    )

    # Track which variables change at each iteration
    changes = []
    prev_state = copy(state)

    for iter in 1:9
        replica = Replica(state, 1, rng, (;), 1)
        Pigeons.step!(explorer, replica, shared)

        # Record which variables changed
        changed = [state[i] != prev_state[i] for i in 1:3]
        push!(changes, changed)
        prev_state = copy(state)
    end

    # Check cycling pattern:
    # iter 1: var 2 updates (iter % 3 == 1)
    # iter 2: var 3 updates (iter % 3 == 2)
    # iter 3: var 1 updates (iter % 3 == 0)
    # iter 4: var 2 updates (iter % 3 == 1)
    # etc.
    @test changes[1] == [false, true, false]  # iter 1: var 2
    @test changes[2] == [false, false, true]  # iter 2: var 3
    @test changes[3] == [true, false, false]  # iter 3: var 1
    @test changes[4] == [false, true, false]  # iter 4: var 2
    @test changes[5] == [false, false, true]  # iter 5: var 3
    @test changes[6] == [true, false, false]  # iter 6: var 1
end

@testset "BlockExplorer - Mixed Cycling and Periodic" begin
    # Test mixed update patterns
    rng = SplittableRandom(42)
    log_potential(x) = -0.5 * sum(x.^2)

    # Mixed: always update vars 1-2, update var 3 every 10 iterations,
    # cycle through vars 4-6
    explorer = BlockExplorer(
        SliceSampler(n_passes=1),
        update_period = [1, 1, 10, 3, 3, 3],
        update_offset = [0, 0, 0, 0, 1, 2]
    )

    state = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    shared = (
        tempering = Pigeons.NonReversiblePT(
            [log_potential],
            Pigeons.Geometric(0.5)
        ),
        iterators = (scan = 1,),
        explorer = explorer,
        reports = (;)
    )

    # Track changes at specific iterations
    changes_iter1 = []
    changes_iter2 = []
    changes_iter3 = []

    for iter in 1:3
        prev_state = copy(state)
        replica = Replica(state, 1, rng, (;), 1)
        Pigeons.step!(explorer, replica, shared)
        changed = [state[i] != prev_state[i] for i in 1:6]

        if iter == 1
            changes_iter1 = changed
        elseif iter == 2
            changes_iter2 = changed
        elseif iter == 3
            changes_iter3 = changed
        end
    end

    # iter 1: vars 1,2 (always), var 5 (offset 1), nothing else
    @test changes_iter1[1:2] == [true, true]  # vars 1-2 always update
    @test changes_iter1[3] == false  # var 3 only updates at iter 10
    @test changes_iter1[4] == false  # var 4 updates at iter 3 (offset 0)
    @test changes_iter1[5] == true   # var 5 updates at iter 1 (offset 1)
    @test changes_iter1[6] == false  # var 6 updates at iter 2 (offset 2)

    # iter 2: vars 1,2 (always), var 6 (offset 2)
    @test changes_iter2[1:2] == [true, true]
    @test changes_iter2[3] == false
    @test changes_iter2[4] == false
    @test changes_iter2[5] == false
    @test changes_iter2[6] == true

    # iter 3: vars 1,2 (always), var 4 (offset 0)
    @test changes_iter3[1:2] == [true, true]
    @test changes_iter3[3] == false
    @test changes_iter3[4] == true
    @test changes_iter3[5] == false
    @test changes_iter3[6] == false
end

@testset "BlockExplorer - Two Cycling Groups" begin
    # Test two separate cycling groups with different periods
    rng = SplittableRandom(42)
    log_potential(x) = -0.5 * sum(x.^2)

    explorer = BlockExplorer(
        SliceSampler(n_passes=1),
        update_period = [5, 5, 5, 5, 5,  3, 3, 3],
        update_offset =    [0, 1, 2, 3, 4,  0, 1, 2]
    )

    state = zeros(8)
    shared = (
        tempering = Pigeons.NonReversiblePT(
            [log_potential],
            Pigeons.Geometric(0.5)
        ),
        iterators = (scan = 1,),
        explorer = explorer,
        reports = (;)
    )

    # Run for 15 iterations to see both cycles
    changes_history = []
    prev_state = copy(state)

    for iter in 1:15
        replica = Replica(state, 1, rng, (;), 1)
        Pigeons.step!(explorer, replica, shared)
        changed = [state[i] != prev_state[i] for i in 1:8]
        push!(changes_history, (iter, changed))
        prev_state = copy(state)
    end

    # Check specific iterations
    # iter 1: var 2 (offset 1 in group 1), var 7 (offset 1 in group 2)
    @test changes_history[1][2][2] == true
    @test changes_history[1][2][7] == true

    # iter 3: var 4 (offset 3 in group 1), var 6 (offset 0 in group 2)
    @test changes_history[3][2][4] == true
    @test changes_history[3][2][6] == true

    # iter 5: var 1 (offset 0 in group 1), var 8 (offset 2 in group 2)
    @test changes_history[5][2][1] == true
    @test changes_history[5][2][8] == true
end

@testset "BlockExplorer - Cycling Integration Test" begin
    # Integration test with pigeons() using cycling
    log_potential(x) = -0.5 * sum(x.^2)

    # Cycle through 6 variables in pairs
    explorer = BlockExplorer(
        SliceSampler(n_passes=1),
        update_period = [3, 3, 3, 3, 3, 3],
        update_offset = [0, 0, 1, 1, 2, 2]
    )

    pt = pigeons(
        target = log_potential,
        explorer = explorer,
        n_rounds = 5,
        n_chains = 4
    )

    @test pt isa Pigeons.PT
    samples = pt.reduced_recorders.online
    @test length(mean(samples)) == 6
    @test all(abs.(mean(samples)) .< 0.5)
end
