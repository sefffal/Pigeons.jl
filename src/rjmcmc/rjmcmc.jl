# Reversible Jump MCMC for Pigeons.jl
#
# This module provides infrastructure for transdimensional MCMC where
# the model space itself is explored alongside parameters.
#
# Key components:
# - RJState: Variable-dimensional state type
# - RJLogPotential: Log potential over model space
# - JumpKernel: Interface for between-model moves
# - RJMCMCExplorer: Explorer for model jumps
#
# For design documentation, see: docs/research/rjmcmc-design.md

include("RJState.jl")
include("RJLogPotential.jl")
include("JumpKernel.jl")
include("RJMCMCExplorer.jl")

# Export main types and functions
export RJState, RJModelInfo, RJLogPotential
export JumpKernel, BirthKernel, DeathKernel, create_birth_death_pair
export RJMCMCExplorer
