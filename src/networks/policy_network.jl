"""
# Policy Network

Maps belief state embeddings to action probabilities.

## The Big Picture

After the belief encoder compresses a belief state into a compact embedding,
the policy network decides what action to take.

Think of it like this:
1. Belief Encoder: "Here's the situation in 16 numbers"
2. Policy Network: "Given that situation, here's what you should do"

## Real-World Analogy

Imagine you're driving and trying to decide whether to turn left or right:
- Your brain processes sensory information (belief encoder)
- Your brain decides which way to turn (policy network)
- The decision is probabilistic: "70% confident I should turn left"

## Why Probabilistic?

We output probabilities, not hard decisions, because:
1. Exploration: Sometimes we want to try suboptimal actions to learn
2. Uncertainty: When unsure, we want to express that uncertainty
3. Learning: Gradients flow better through soft probabilities than hard decisions
"""

using Flux
using Statistics

"""
    PolicyNetwork

Neural network that maps belief embeddings to action probabilities.

# Architecture
- Input: Belief embedding (typically 16-dim)
- Hidden layer 1: 32 neurons with ReLU
- Hidden layer 2: 16 neurons with ReLU
- Output: Action probabilities (softmax)

# Why This Architecture?

The network is relatively small because:
1. The belief encoder already did the hard work of compression
2. Policy decisions are often simpler than belief encoding
3. Smaller networks train faster and generalize better

# Fields
- `network::Chain`: The neural network
- `embedding_dim::Int`: Size of input embedding
- `n_actions::Int`: Number of possible actions

# Example
```julia
policy = PolicyNetwork(embedding_dim=16, n_actions=3)
embedding = rand(16)
action_probs = policy(embedding)  # Returns [0.2, 0.5, 0.3] (sums to 1.0)
```
"""
struct PolicyNetwork
    network::Chain
    embedding_dim::Int
    n_actions::Int
end

"""
    PolicyNetwork(; embedding_dim::Int, n_actions::Int, hidden_dims::Vector{Int}=[32, 16])

Create a policy network with specified architecture.

# Arguments
- `embedding_dim`: Size of input belief embedding
- `n_actions`: Number of possible actions
- `hidden_dims`: Sizes of hidden layers (default: [32, 16])

# Returns
- `PolicyNetwork`: Initialized policy network

# Error Handling
- Validates dimensions are positive
- Ensures n_actions ≥ 2 (need at least 2 actions for a decision)
- Provides helpful error messages

# Example
```julia
# For a POMDP with 3 actions (e.g., "left", "right", "listen")
policy = PolicyNetwork(embedding_dim=16, n_actions=3)

# Custom architecture
policy = PolicyNetwork(embedding_dim=32, n_actions=5, hidden_dims=[64, 32])
```
"""
function PolicyNetwork(;
    embedding_dim::Int,
    n_actions::Int,
    hidden_dims::Vector{Int}=[32, 16]
)
    try
        # Input validation
        @assert embedding_dim > 0 "embedding_dim must be positive, got $embedding_dim"
        @assert n_actions >= 2 "n_actions must be ≥ 2 (need at least 2 actions), got $n_actions"
        @assert all(d -> d > 0, hidden_dims) "All hidden dimensions must be positive"
        
        # Build network layers
        layers = []
        
        # Input → First hidden layer
        push!(layers, Dense(embedding_dim, hidden_dims[1], relu))
        
        # Hidden layers
        for i in 1:(length(hidden_dims)-1)
            push!(layers, Dense(hidden_dims[i], hidden_dims[i+1], relu))
        end
        
        # Last hidden → Output with softmax
        # Softmax ensures outputs are valid probabilities (sum to 1.0)
        push!(layers, Dense(hidden_dims[end], n_actions))
        push!(layers, softmax)
        
        network = Chain(layers...)
        
        return PolicyNetwork(network, embedding_dim, n_actions)
        
    catch e
        if isa(e, AssertionError)
            error("PolicyNetwork creation failed: $(e.msg)")
        else
            error("Unexpected error creating PolicyNetwork: $e")
        end
    end
end

"""
    (policy::PolicyNetwork)(embedding::AbstractVector)

Compute action probabilities given a belief embedding.

# How It Works

1. Takes belief embedding (compact representation of situation)
2. Passes through neural network layers
3. Applies softmax to get valid probabilities
4. Returns probability for each action

# Arguments
- `embedding`: Belief embedding vector

# Returns
- Action probability vector (sums to 1.0)

# Error Handling
- Validates embedding dimension
- Checks output is valid probability distribution
- Provides clear error messages

# Example
```julia
policy = PolicyNetwork(embedding_dim=16, n_actions=3)
embedding = rand(16)
probs = policy(embedding)
# probs might be [0.15, 0.60, 0.25] - "60% confident in action 2"
```
"""
function (policy::PolicyNetwork)(embedding::AbstractVector)
    try
        # Validate input dimension
        if length(embedding) != policy.embedding_dim
            error("Embedding dimension mismatch: expected $(policy.embedding_dim), got $(length(embedding))")
        end
        
        # Compute action probabilities
        action_probs = policy.network(embedding)
        
        # Sanity check: probabilities should sum to 1.0
        prob_sum = sum(action_probs)
        if abs(prob_sum - 1.0) > 1e-5
            @warn "Action probabilities don't sum to 1.0 (sum = $prob_sum). " *
                  "This might indicate a numerical issue."
        end
        
        # Sanity check: no negative probabilities
        if any(action_probs .< 0)
            error("Policy network produced negative probabilities: $action_probs")
        end
        
        return action_probs
        
    catch e
        if isa(e, DimensionMismatch)
            error("Policy forward pass failed due to dimension mismatch: $e")
        else
            rethrow(e)
        end
    end
end

"""
    sample_action(policy::PolicyNetwork, embedding::AbstractVector; rng=Random.GLOBAL_RNG)

Sample an action from the policy's probability distribution.

# Why Sample Instead of Argmax?

During training, we want to explore different actions, not always pick
the most likely one. Think of it like trying different routes to work:
- Sometimes take the "best" route (70% probability)
- Sometimes try alternatives (30% probability)
- This exploration helps us learn better policies

# Arguments
- `policy`: The policy network
- `embedding`: Belief embedding
- `rng`: Random number generator (for reproducibility)

# Returns
- `action::Int`: Sampled action index (1 to n_actions)

# Example
```julia
policy = PolicyNetwork(embedding_dim=16, n_actions=3)
embedding = rand(16)

# Sample 100 actions and see the distribution
actions = [sample_action(policy, embedding) for _ in 1:100]
# Might get: 15 action-1's, 60 action-2's, 25 action-3's
# Roughly matching the policy's probabilities
```
"""
function sample_action(policy::PolicyNetwork, embedding::AbstractVector; rng=Random.GLOBAL_RNG)
    try
        # Get action probabilities
        probs = policy(embedding)
        
        # Sample from categorical distribution
        # This is like spinning a weighted roulette wheel
        action = StatsBase.sample(rng, 1:policy.n_actions, Weights(probs))
        
        return action
        
    catch e
        error("Action sampling failed: $e")
    end
end

"""
    greedy_action(policy::PolicyNetwork, embedding::AbstractVector)

Select the most likely action (no exploration).

# When to Use This

- During evaluation (not training)
- When you want the "best" action according to current policy
- When exploration is not needed

# Arguments
- `policy`: The policy network
- `embedding`: Belief embedding

# Returns
- `action::Int`: Action with highest probability

# Example
```julia
policy = PolicyNetwork(embedding_dim=16, n_actions=3)
embedding = rand(16)
action = greedy_action(policy, embedding)
# Returns the action with highest probability (no randomness)
```
"""
function greedy_action(policy::PolicyNetwork, embedding::AbstractVector)
    try
        probs = policy(embedding)
        return argmax(probs)
    catch e
        error("Greedy action selection failed: $e")
    end
end

"""
    NeuralPolicy

Complete neural policy combining belief encoder and policy network.

This is the full pipeline:
Belief State → Belief Encoder → Embedding → Policy Network → Action

Think of it as a complete "brain" for decision-making under uncertainty.

# Fields
- `encoder::BeliefEncoder`: Encodes beliefs into embeddings
- `policy::PolicyNetwork`: Maps embeddings to actions
- `training_mode::Bool`: Whether in training mode (affects exploration)

# Example
```julia
neural_policy = NeuralPolicy(
    encoder=BeliefEncoder(belief_dim=10, embedding_dim=16),
    policy=PolicyNetwork(embedding_dim=16, n_actions=3)
)

belief = normalize_belief(rand(10))
action = neural_policy(belief)  # Returns action index
```
"""
mutable struct NeuralPolicy
    encoder::BeliefEncoder
    policy::PolicyNetwork
    training_mode::Bool
end

"""
    NeuralPolicy(; belief_dim::Int, n_actions::Int, embedding_dim::Int=16)

Create a complete neural policy with default architecture.

# Arguments
- `belief_dim`: Size of belief state
- `n_actions`: Number of possible actions
- `embedding_dim`: Size of intermediate embedding (default: 16)

# Returns
- `NeuralPolicy`: Complete policy ready for training

# Example
```julia
# For Tiger POMDP: 2 states, 3 actions
policy = NeuralPolicy(belief_dim=2, n_actions=3)
```
"""
function NeuralPolicy(; belief_dim::Int, n_actions::Int, embedding_dim::Int=16)
    try
        encoder = BeliefEncoder(belief_dim=belief_dim, embedding_dim=embedding_dim)
        policy = PolicyNetwork(embedding_dim=embedding_dim, n_actions=n_actions)
        return NeuralPolicy(encoder, policy, true)
    catch e
        error("NeuralPolicy creation failed: $e")
    end
end

"""
    (neural_policy::NeuralPolicy)(belief::AbstractVector; sample::Bool=true, return_probs::Bool=false)

Execute the full policy: belief → embedding → action probabilities.

# Arguments
- `belief`: Belief state vector
- `sample`: If true, sample action; if false, take greedy action (only used if return_probs=false)
- `return_probs`: If true, return probability vector; if false, return action index

# Returns
- If `return_probs=true`: Action probability vector
- If `return_probs=false`: Selected action index

# Example
```julia
policy = NeuralPolicy(belief_dim=10, n_actions=3)
belief = normalize_belief(rand(10))

# Get probabilities (for testing)
probs = policy(belief, return_probs=true)

# Get action (for execution)
action = policy(belief, sample=true, return_probs=false)
```
"""
function (neural_policy::NeuralPolicy)(belief::AbstractVector; sample::Bool=true, return_probs::Bool=false)
    try
        # Encode belief
        embedding = neural_policy.encoder(belief)
        
        # Get action probabilities
        probs = neural_policy.policy(embedding)
        
        # Return probabilities or action
        if return_probs
            return probs
        else
            # Select action
            if sample && neural_policy.training_mode
                action = sample_action(neural_policy.policy, embedding)
            else
                action = greedy_action(neural_policy.policy, embedding)
            end
            return action
        end
        
    catch e
        error("Neural policy execution failed: $e")
    end
end

"""
    test_policy_network()

Test policy network functionality.

Run this to verify everything works before training.
"""
function test_policy_network()
    println("Testing PolicyNetwork...")
    
    try
        # Test 1: Basic creation
        policy = PolicyNetwork(embedding_dim=16, n_actions=3)
        println("✓ Policy network created")
        
        # Test 2: Forward pass
        embedding = rand(16)
        probs = policy(embedding)
        @assert length(probs) == 3 "Wrong number of action probabilities"
        @assert abs(sum(probs) - 1.0) < 1e-5 "Probabilities don't sum to 1.0"
        println("✓ Forward pass works")
        
        # Test 3: Action sampling
        action = sample_action(policy, embedding)
        @assert 1 <= action <= 3 "Invalid action sampled"
        println("✓ Action sampling works")
        
        # Test 4: Greedy action
        action = greedy_action(policy, embedding)
        @assert 1 <= action <= 3 "Invalid greedy action"
        println("✓ Greedy action works")
        
        # Test 5: Complete neural policy
        neural_policy = NeuralPolicy(belief_dim=10, n_actions=3)
        belief = normalize_belief(rand(10))
        action = neural_policy(belief)
        @assert 1 <= action <= 3 "Invalid action from neural policy"
        println("✓ Complete neural policy works")
        
        println("\n✓ All policy network tests passed!")
        return true
        
    catch e
        println("\n✗ Policy network test failed: $e")
        return false
    end
end

"""
    POMDPs.action(policy::NeuralPolicy, belief)

POMDPs.jl interface for NeuralPolicy.

Converts belief to vector format and selects action.
"""
function POMDPs.action(policy::NeuralPolicy, belief)
    try
        # Convert belief to vector
        if isa(belief, AbstractVector)
            belief_vec = Float64.(belief)  # Ensure Float64
        else
            # For DiscreteBelief, extract probability vector
            # Get states in order and compute pdf for each
            states_list = ordered_states(belief.pomdp)
            belief_vec = Float64[pdf(belief, s) for s in states_list]
        end
        
        # Convert to Float32 for Flux network
        belief_vec32 = Float32.(belief_vec)
        
        # Get action (returns index, not probs)
        action_idx = policy(belief_vec32, sample=policy.training_mode, return_probs=false)
        
        # Convert index to action symbol
        return actions(belief.pomdp)[action_idx]
        
    catch e
        @error "POMDPs.action failed for NeuralPolicy" exception=(e, catch_backtrace())
        # Return first action as fallback
        return actions(belief.pomdp)[1]
    end
end

# Export functions
export PolicyNetwork, NeuralPolicy
export sample_action, greedy_action
export test_policy_network
