"""
# NeuralPOMDPPolicy

Neural network approximation of POMDP policies.

Combines classical POMDP theory with modern deep reinforcement learning:
- Belief state encoding via neural networks
- Policy approximation with policy gradients
- Experience replay for sample efficiency
- Comparison with classical baselines (QMDP, SARSOP)

## Main Components

- `BeliefEncoder`: Encode belief distributions into fixed-size vectors
- `PolicyNetwork`: Map belief embeddings to action probabilities
- `ExperienceReplay`: Store and sample training experiences
- `train_neural_policy`: Main training loop
- `evaluate_policy`: Evaluate trained policies

## Example

```julia
using NeuralPOMDPPolicy
using POMDPs

# Create POMDP
pomdp = TigerPOMDP()

# Train neural policy
policy = train_neural_policy(
    pomdp,
    episodes=1000,
    learning_rate=0.001
)

# Evaluate
avg_reward = evaluate_policy(pomdp, policy, n_episodes=100)
```
"""
module NeuralPOMDPPolicy

using Flux
using POMDPs
using POMDPTools
using Random
using Statistics
using StatsBase
using BSON

# Export main types
export BeliefEncoder, PolicyNetwork, NeuralPolicy
export ExperienceReplay, Experience

# Export training functions
export train_neural_policy, train_step!
export collect_episode, update_policy!
export add_experience!, sample_batch, unpack_batch

# Export evaluation functions
export evaluate_policy, compare_policies
export compute_metrics

# Export environments
export TigerPOMDP, LightDarkPOMDP
export create_tiger_updater

# Export utilities
export save_policy, load_policy
export plot_training_curves, plot_belief_space

# Include submodules
include("networks/belief_encoder.jl")
include("networks/policy_network.jl")
include("training/experience_replay.jl")
include("training/trainer.jl")
include("evaluation/metrics.jl")
include("evaluation/comparison.jl")
include("environments/tiger.jl")
include("environments/lightdark.jl")

end # module
