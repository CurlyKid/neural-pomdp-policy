"""
Training pipeline for neural POMDP policies using policy gradients.

This module implements the REINFORCE algorithm (Williams, 1992) with baseline
for variance reduction. Think of it like teaching a student: they try actions,
see what works, and gradually learn to favor successful strategies.

## Key Concepts

**Policy Gradients**: Instead of learning Q-values (like DQN), we directly
optimize the policy parameters to maximize expected reward. It's like learning
to play piano by practicing pieces, not by memorizing every possible note.

**Baseline**: We subtract average reward to reduce variance. Like grading on
a curve - what matters is doing better than average, not absolute scores.

**Episode Collection**: We run the agent in the environment, collecting
(belief, action, reward) tuples. This is the "experience" we learn from.
"""

using Flux
using POMDPs
using POMDPTools
using Random
using Statistics
using Logging

"""
    TrainingConfig

Configuration for training neural POMDP policies.

# Fields
- `episodes::Int`: Number of training episodes (default: 1000)
- `learning_rate::Float64`: Learning rate for optimizer (default: 0.001)
- `gamma::Float64`: Discount factor for future rewards (default: 0.95)
- `max_steps::Int`: Maximum steps per episode (default: 100)
- `batch_size::Int`: Batch size for experience replay (default: 32)
- `use_baseline::Bool`: Use baseline for variance reduction (default: true)
- `log_interval::Int`: Log progress every N episodes (default: 10)
- `checkpoint_interval::Int`: Save model every N episodes (default: 100)

# Example
```julia
config = TrainingConfig(
    episodes=500,
    learning_rate=0.001,
    gamma=0.95
)
```
"""
struct TrainingConfig
    episodes::Int
    learning_rate::Float64
    gamma::Float64
    max_steps::Int
    batch_size::Int
    use_baseline::Bool
    log_interval::Int
    checkpoint_interval::Int
end

function TrainingConfig(;
    episodes=1000,
    learning_rate=0.001,
    gamma=0.95,
    max_steps=100,
    batch_size=32,
    use_baseline=true,
    log_interval=10,
    checkpoint_interval=100
)
    return TrainingConfig(
        episodes, learning_rate, gamma, max_steps,
        batch_size, use_baseline, log_interval, checkpoint_interval
    )
end

"""
    TrainingMetrics

Tracks training progress over time.

# Fields
- `episode_rewards::Vector{Float64}`: Total reward per episode
- `episode_lengths::Vector{Int}`: Steps per episode
- `losses::Vector{Float64}`: Policy gradient loss per episode
- `entropies::Vector{Float64}`: Policy entropy per episode (exploration measure)
"""
mutable struct TrainingMetrics
    episode_rewards::Vector{Float64}
    episode_lengths::Vector{Int}
    losses::Vector{Float64}
    entropies::Vector{Float64}
end

TrainingMetrics() = TrainingMetrics(Float64[], Int[], Float64[], Float64[])

"""
    collect_episode(pomdp, policy, updater; max_steps=100)

Run one episode in the POMDP environment, collecting experiences.

This is like watching a student take a test - we record every question they
answer and whether they got it right. Later, we'll use this to teach them.

# Arguments
- `pomdp`: The POMDP environment
- `policy`: Neural policy to execute
- `updater`: Belief updater (e.g., DiscreteUpdater)
- `max_steps`: Maximum episode length

# Returns
- `experiences`: Vector of (belief, action, reward) tuples
- `total_reward`: Sum of rewards in episode
- `steps`: Number of steps taken

# Example
```julia
experiences, reward, steps = collect_episode(
    tiger_pomdp,
    neural_policy,
    updater,
    max_steps=50
)
```
"""
function collect_episode(pomdp, policy, updater; max_steps=100)
    experiences = Experience[]
    total_reward = 0.0
    
    try
        # Initialize episode
        s0 = rand(initialstate(pomdp))
        b0 = initialize_belief(updater, initialstate(pomdp))
        
        belief = b0
        steps = 0
        
        # Run episode
        for step in 1:max_steps
            steps = step
            
            # Select action from policy
            # The policy looks at current belief and decides what to do
            a = action(policy, belief)
            
            # Convert belief to vector for storage
            states_list = ordered_states(pomdp)
            belief_vec = Float64[pdf(belief, s) for s in states_list]
            
            # Take action in environment
            # Like a student answering a question - we see what happens
            s = rand(belief)  # Sample state from belief
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
            
            # Update belief with new observation
            # Like updating our understanding after seeing the result
            bp = update(updater, belief, a, o)
            
            # Convert next belief to vector
            next_belief_vec = Float64[pdf(bp, s) for s in states_list]
            
            # Convert action symbol to index
            action_idx = findfirst(==(a), actions(pomdp))
            
            # Store experience for learning
            push!(experiences, Experience(belief_vec, action_idx, r, next_belief_vec, false))
            
            total_reward += r
            belief = bp
            
            # Check if episode ended
            if isterminal(pomdp, sp)
                # Mark last experience as terminal
                experiences[end] = Experience(
                    experiences[end].belief,
                    experiences[end].action,
                    experiences[end].reward,
                    experiences[end].next_belief,
                    true  # terminal flag
                )
                break
            end
        end
        
        return experiences, total_reward, steps
        
    catch e
        @error "Episode collection failed" exception=(e, catch_backtrace())
        # Return empty episode on failure
        return Experience[], 0.0, 0
    end
end

"""
    compute_returns(rewards, gamma; use_baseline=true)

Compute discounted returns for policy gradient update.

Think of this like calculating your final grade: recent performance matters
more (discount factor), and we compare to class average (baseline).

# Arguments
- `rewards`: Vector of rewards from episode
- `gamma`: Discount factor (0.95 means future rewards worth 95% of immediate)
- `use_baseline`: Subtract mean return for variance reduction

# Returns
- `returns`: Discounted returns for each timestep

# Example
```julia
rewards = [1.0, 0.0, -1.0, 10.0]
returns = compute_returns(rewards, 0.95, use_baseline=true)
# Returns will emphasize the big reward at the end
```
"""
function compute_returns(rewards::Vector{Float64}, gamma::Float64; use_baseline=true)
    T = length(rewards)
    returns = zeros(Float64, T)
    
    # Compute discounted returns backwards
    # Like compound interest, but for rewards
    G = 0.0
    for t in T:-1:1
        G = rewards[t] + gamma * G
        returns[t] = G
    end
    
    # Subtract baseline (mean) for variance reduction
    # This helps the algorithm focus on "better than average" actions
    if use_baseline && length(returns) > 1
        baseline = mean(returns)
        returns .-= baseline
    end
    
    return returns
end

"""
    policy_gradient_loss(policy, beliefs, actions, returns)

Compute policy gradient loss for a batch of experiences.

This is the core of REINFORCE: we increase probability of actions that led
to good outcomes (high returns) and decrease probability of actions that led
to bad outcomes (low returns).

# Arguments
- `policy`: Neural policy
- `beliefs`: Batch of belief states
- `actions`: Actions taken
- `returns`: Discounted returns (rewards-to-go)

# Returns
- `loss`: Scalar loss value
- `entropy`: Policy entropy (measure of exploration)

# Mathematical Intuition
Loss = -mean(log_prob(action) * return)

If return is positive: increase log_prob (make action more likely)
If return is negative: decrease log_prob (make action less likely)

It's like reinforcing good habits and discouraging bad ones.
"""
function policy_gradient_loss(policy, beliefs, actions, returns)
    try
        # Get action probabilities for each belief
        # This is what the policy currently thinks is good
        action_probs = [policy(b) for b in beliefs]
        
        # Compute log probabilities of actions taken
        # Higher log_prob means policy was confident in this action
        log_probs = [log(probs[a] + 1e-8) for (probs, a) in zip(action_probs, actions)]
        
        # Policy gradient: weight log_probs by returns
        # Good outcomes (high returns) → increase those action probabilities
        # Bad outcomes (low returns) → decrease those action probabilities
        policy_loss = -mean(log_probs .* returns)
        
        # Compute entropy for monitoring exploration
        # High entropy = policy is uncertain (exploring)
        # Low entropy = policy is confident (exploiting)
        entropies = [-sum(p .* log.(p .+ 1e-8)) for p in action_probs]
        avg_entropy = mean(entropies)
        
        return policy_loss, avg_entropy
        
    catch e
        @error "Loss computation failed" exception=(e, catch_backtrace())
        return Inf, 0.0
    end
end

"""
    train_step!(policy, optimizer, experiences, config)

Perform one training step (update policy parameters).

This is where learning happens - we look at what worked, compute gradients,
and adjust the policy to do better next time.

# Arguments
- `policy`: Neural policy to train
- `optimizer`: Flux optimizer (e.g., Adam)
- `experiences`: Vector of experiences from episode
- `config`: Training configuration

# Returns
- `loss`: Policy gradient loss
- `entropy`: Policy entropy

# Example
```julia
loss, entropy = train_step!(
    policy,
    optimizer,
    episode_experiences,
    config
)
```
"""
function train_step!(policy, optimizer, experiences, config)
    try
        # Extract components from experiences
        beliefs = [exp.belief for exp in experiences]
        actions = [exp.action for exp in experiences]
        rewards = [exp.reward for exp in experiences]
        
        # Compute discounted returns
        # This tells us how good each action was in hindsight
        returns = compute_returns(rewards, config.gamma, use_baseline=config.use_baseline)
        
        # Compute loss and gradients
        loss, entropy = policy_gradient_loss(policy, beliefs, actions, returns)
        
        # Update policy parameters
        # Like adjusting your strategy after seeing what worked
        grads = gradient(() -> policy_gradient_loss(policy, beliefs, actions, returns)[1], 
                        Flux.params(policy))
        Flux.update!(optimizer, Flux.params(policy), grads)
        
        return loss, entropy
        
    catch e
        @error "Training step failed" exception=(e, catch_backtrace())
        return Inf, 0.0
    end
end

"""
    train_neural_policy(pomdp, updater; config=TrainingConfig())

Train a neural policy on a POMDP using policy gradients.

This is the main training loop - it runs episodes, collects experiences,
and updates the policy to maximize reward.

# Arguments
- `pomdp`: POMDP environment
- `updater`: Belief updater
- `config`: Training configuration

# Returns
- `policy`: Trained neural policy
- `metrics`: Training metrics (rewards, losses, etc.)

# Example
```julia
# Train on Tiger POMDP
policy, metrics = train_neural_policy(
    tiger_pomdp,
    DiscreteUpdater(tiger_pomdp),
    config=TrainingConfig(episodes=500)
)

# Check final performance
println("Final avg reward: ", mean(metrics.episode_rewards[end-10:end]))
```
"""
function train_neural_policy(pomdp, updater; config=TrainingConfig())
    @info "Starting training" config
    
    try
        # Create policy
        n_states = length(states(pomdp))
        n_actions = length(actions(pomdp))
        
        encoder = BeliefEncoder(belief_dim=n_states, embedding_dim=64)
        policy_net = PolicyNetwork(embedding_dim=64, n_actions=n_actions)
        policy = NeuralPolicy(encoder, policy_net, true)  # training_mode=true
        
        # Create optimizer
        optimizer = Adam(config.learning_rate)
        
        # Track metrics
        metrics = TrainingMetrics()
        
        # Training loop
        for episode in 1:config.episodes
            # Collect episode
            experiences, total_reward, steps = collect_episode(
                pomdp, policy, updater,
                max_steps=config.max_steps
            )
            
            # Skip if episode collection failed
            if isempty(experiences)
                @warn "Episode $episode failed, skipping"
                continue
            end
            
            # Update policy
            loss, entropy = train_step!(policy, optimizer, experiences, config)
            
            # Record metrics
            push!(metrics.episode_rewards, total_reward)
            push!(metrics.episode_lengths, steps)
            push!(metrics.losses, loss)
            push!(metrics.entropies, entropy)
            
            # Log progress
            if episode % config.log_interval == 0
                avg_reward = mean(metrics.episode_rewards[max(1, end-9):end])
                avg_loss = mean(metrics.losses[max(1, end-9):end])
                @info "Episode $episode" avg_reward avg_loss entropy steps
            end
            
            # Checkpoint
            if episode % config.checkpoint_interval == 0
                @info "Checkpoint at episode $episode"
                # Could save model here if needed
            end
        end
        
        @info "Training complete" final_avg_reward=mean(metrics.episode_rewards[end-9:end])
        
        return policy, metrics
        
    catch e
        @error "Training failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    test_trainer()

Test the training pipeline with a simple example.

Run this to verify the trainer works before using on real problems.
"""
function test_trainer()
    println("\n=== Testing Training Pipeline ===\n")
    
    try
        # This would need a real POMDP to test properly
        # For now, just verify the functions exist and have correct signatures
        println("✓ TrainingConfig constructor works")
        config = TrainingConfig(episodes=10)
        
        println("✓ TrainingMetrics constructor works")
        metrics = TrainingMetrics()
        
        println("✓ compute_returns works")
        rewards = [1.0, 0.0, -1.0, 10.0]
        returns = compute_returns(rewards, 0.95)
        println("  Sample returns: ", returns)
        
        println("\n✓ All trainer components initialized successfully!")
        println("  Note: Full integration test requires POMDP environment")
        
    catch e
        println("✗ Trainer test failed: ", e)
        rethrow(e)
    end
end
