using Test
using Pigeons
using Distributions

@testset "RJMCMC Basic Functionality" begin

    @testset "RJModelInfo Construction" begin
        model_specs = Dict(
            1 => (continuous = [:x], discrete = Symbol[]),
            2 => (continuous = [:x, :y], discrete = Symbol[])
        )

        info = RJModelInfo(model_specs)

        @test info.n_models == 2
        @test info.continuous_vars[1] == [:x]
        @test info.continuous_vars[2] == [:x, :y]
        @test info.dimensions[1] == 1
        @test info.dimensions[2] == 2
    end

    @testset "RJState Construction and Interface" begin
        model_specs = Dict(
            1 => (continuous = [:x], discrete = Symbol[]),
            2 => (continuous = [:x, :y], discrete = Symbol[])
        )
        info = RJModelInfo(model_specs)

        # Create state for model 1
        state = RJState(
            1,
            Dict(:x => [1.0]),
            1,
            info
        )

        @test state.model_indicator == 1
        @test state.dimension == 1
        @test Pigeons.continuous_variables(state) == [:x]
        @test Pigeons.discrete_variables(state) == Symbol[]
        @test Pigeons.variable(state, :x) == [1.0]

        # Test copy
        state2 = copy(state)
        @test state2.model_indicator == state.model_indicator
        @test state2.dimension == state.dimension
        @test state2.parameters[:x] == state.parameters[:x]
        @test state2.parameters[:x] !== state.parameters[:x]  # Deep copy

        # Test indexing (SliceSampler compatibility)
        @test length(state) == 1
        @test state[1] == 1.0
        state[1] = 2.0
        @test state[1] == 2.0
        @test state.parameters[:x][1] == 2.0
    end

    @testset "RJLogPotential Simple Example" begin
        # Define two simple models
        model1_lp(params) = logpdf(Normal(0, 1), params[:x][1])
        model2_lp(params) = logpdf(Normal(0, 1), params[:x][1]) + logpdf(Normal(0, 1), params[:y][1])

        model_specs = Dict(
            1 => (continuous = [:x], discrete = Symbol[]),
            2 => (continuous = [:x, :y], discrete = Symbol[])
        )

        target = RJLogPotential(
            models = Dict(1 => model1_lp, 2 => model2_lp),
            model_prior = Categorical([0.5, 0.5]),
            model_specs = model_specs
        )

        # Test evaluation for model 1
        state1 = RJState(1, Dict(:x => [0.0]), 1, target.model_info)
        lp1 = target(state1)
        @test isfinite(lp1)
        @test lp1 ≈ logpdf(Normal(0, 1), 0.0) + log(0.5)

        # Test evaluation for model 2
        state2 = RJState(2, Dict(:x => [0.0], :y => [0.0]), 2, target.model_info)
        lp2 = target(state2)
        @test isfinite(lp2)
        @test lp2 ≈ logpdf(Normal(0, 1), 0.0) + logpdf(Normal(0, 1), 0.0) + log(0.5)

        # Test initialization
        rng = SplittableRandom(1)
        init_state = Pigeons.initialization(target, rng, 1)
        @test init_state isa RJState
        @test 1 <= init_state.model_indicator <= 2
    end

    @testset "Jump Kernels" begin
        model_specs = Dict(
            1 => (continuous = [:x], discrete = Symbol[]),
            2 => (continuous = [:x, :theta], discrete = Symbol[])
        )
        info = RJModelInfo(model_specs)

        # Test birth kernel
        birth = BirthKernel(
            target_model = 2,
            new_params = Dict(:theta => Normal(0, 1))
        )

        state1 = RJState(1, Dict(:x => [1.0]), 1, info)
        rng = SplittableRandom(1)

        new_state, log_jac, log_prop_ratio = birth(state1, rng)

        @test new_state.model_indicator == 2
        @test new_state.dimension == 2
        @test haskey(new_state.parameters, :theta)
        @test log_jac == 0.0
        @test log_prop_ratio < 0.0  # Should account for proposal density

        # Test death kernel
        death = DeathKernel(
            target_model = 1,
            remove_params = [:theta],
            reverse_proposals = Dict(:theta => Normal(0, 1))
        )

        state2 = RJState(2, Dict(:x => [1.0], :theta => [0.5]), 2, info)
        new_state2, log_jac2, log_prop_ratio2 = death(state2, rng)

        @test new_state2.model_indicator == 1
        @test new_state2.dimension == 1
        @test !haskey(new_state2.parameters, :theta)
        @test log_jac2 == 0.0
        @test log_prop_ratio2 > 0.0  # Should account for reverse birth proposal

        # Test create_birth_death_pair
        birth_pair, death_pair = create_birth_death_pair(
            model_low = 1,
            model_high = 2,
            new_params = Dict(:theta => Normal(0, 1))
        )

        @test birth_pair.first == 1
        @test birth_pair.second isa BirthKernel
        @test death_pair.first == 2
        @test death_pair.second isa DeathKernel
    end

    @testset "RJMCMCExplorer Construction" begin
        birth, death = create_birth_death_pair(
            model_low = 1,
            model_high = 2,
            new_params = Dict(:theta => Normal(0, 1))
        )

        jump_kernels = Dict(birth, death)

        explorer = RJMCMCExplorer(
            jump_kernels = jump_kernels,
            n_attempts = 1
        )

        @test explorer.n_attempts == 1
        @test length(explorer.jump_kernels) == 2
    end

    @testset "Simple Two-Model RJMCMC Run" begin
        # This test just verifies the code runs without errors
        # A full statistical test would require many samples

        # Define two nested models
        model1_lp(params) = logpdf(Normal(0, 1), params[:x][1])
        model2_lp(params) = (
            logpdf(Normal(0, 1), params[:x][1]) +
            logpdf(Normal(0, 1), params[:theta][1])
        )

        model_specs = Dict(
            1 => (continuous = [:x], discrete = Symbol[]),
            2 => (continuous = [:x, :theta], discrete = Symbol[])
        )

        target = RJLogPotential(
            models = Dict(1 => model1_lp, 2 => model2_lp),
            model_prior = Categorical([0.5, 0.5]),
            model_specs = model_specs
        )

        # Create jump kernels
        birth, death = create_birth_death_pair(
            model_low = 1,
            model_high = 2,
            new_params = Dict(:theta => Normal(0, 1))
        )

        # Compose RJMCMC with SliceSampler
        explorer = Compose(
            RJMCMCExplorer(jump_kernels = Dict(birth, death)),
            SliceSampler(n_passes = 1)
        )

        # Run a very short test (just to check it doesn't crash)
        pt = pigeons(
            target = target,
            explorer = explorer,
            n_chains = 4,
            n_rounds = 3,
            record = []  # Don't record anything for this basic test
        )

        @test pt isa PT
    end

end
