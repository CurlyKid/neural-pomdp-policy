"""
# Experience Replay

Stores and samples past experiences for training.

## Why Experience Replay?

Imagine learning to play chess:
- **Without replay**: You only learn from your current game, then forget it
- **With replay**: You remember past games and learn from them repeatedly

Experience replay does three important things:
1. **Breaks correlations**: Consecutive experiences are similar; random sampling breaks this
2. **Improves sample efficiency**: Learn from each experience multiple times
3. **Stabilizes learning**: Smooth out the noise in individual experiences

## Real-World Analogy

Think of it like studying for an exam:
- You don't just read your notes once in order
- You shuffle flashcards and review them multiple times
- This repetition with variation helps you learn better

## The Data

Each experience is a tuple: (belief, action, reward, next_belief, done)
- belief: "Where was I?"
- action: "What did I do?"
- reward: "How did it go?"
- next_belief: "Where am I now?"
- done: "Is the episode over?"
"""

using DataStructures

"""
    Experience

A single experience tuple from interacting with the environment.

Think of this as a memory of one decision:
- "I was in this situation (belief)"
- "I took this action"
- "I got this reward"
- "Now I'm in this new situation (next_belief)"
- "The episode ended (or didn't)"

# Fields
- `belief::Vector{Float64}`: Belief state before action
- `action::Int`: Action taken
- `reward::Float64`: Reward received
- `next_belief::Vector{Float64}`: Belief state after action
- `done::Bool`: Whether episode ended

# Example
```julia
exp = Experience(
    belief=[0.3, 0.7],
    action=1,
    reward=10.0,
    next_belief=[0.8, 0.2],
    done=false
)
```
"""
struct Experience
    belief::Vector{Float64}
    action::Int
    reward::Float64
    next_belief::Vector{Float64}
    done::Bool
end

"""
    ExperienceReplay

Circular buffer for storing and sampling experiences.

# How It Works

Imagine a circular bookshelf:
1. You add books (experiences) one by one
2. When full, new books replace the oldest ones
3. You can randomly pick books to read (sample)

This "circular" property means we keep recent experiences
while gradually forgetting very old ones.

# Fields
- `buffer::CircularBuffer{Experience}`: The storage
- `capacity::Int`: Maximum number of experiences to store
- `size::Int`: Current number of experiences stored

# Example
```julia
replay = ExperienceReplay(capacity=10000)

# Add experience
push!(replay, belief, action, reward, next_belief, done)

# Sample batch for training
batch = sample(replay, batch_size=32)
```
"""
mutable struct ExperienceReplay
    buffer::CircularBuffer{Experience}
    capacity::Int
    size::Int
    
    function ExperienceReplay(capacity::Int)
        try
            @assert capacity > 0 "Capacity must be positive, got $capacity"
            
            # Warn if capacity seems too small
            if capacity < 100
                @warn "Experience replay capacity ($capacity) is very small. " *
                      "Consider using at least 1000 for stable learning."
            end
            
            buffer = CircularBuffer{Experience}(capacity)
            return new(buffer, capacity, 0)
            
        catch e
            error("ExperienceReplay creation failed: $e")
        end
    end
end

"""
    Base.push!(replay::ExperienceReplay, belief, action, reward, next_belief, done)

Add a new experience to the replay buffer.

# How It Works

Like adding a photo to a photo album:
- If there's space, add it to the next empty slot
- If full, replace the oldest photo

# Arguments
- `replay`: The replay buffer
- `belief`: Current belief state
- `action`: Action taken
- `reward`: Reward received
- `next_belief`: Next belief state
- `done`: Whether episode ended

# Error Handling
- Validates belief dimensions match
- Checks action is valid integer
- Ensures reward is finite (not NaN or Inf)

# Example
```julia
replay = ExperienceReplay(capacity=1000)
push!(replay, [0.5, 0.5], 1, 10.0, [0.8, 0.2], false)
```
"""
function Base.push!(
    replay::ExperienceReplay,
    belief::AbstractVector,
    action::Int,
    reward::Real,
    next_belief::AbstractVector,
    done::Bool
)
    try
        # Validate inputs
        @assert length(belief) == length(next_belief) "Belief dimensions must match"
        @assert action > 0 "Action must be positive integer"
        @assert isfinite(reward) "Reward must be finite (got $reward)"
        
        # Create experience
        exp = Experience(
            Vector{Float64}(belief),
            action,
            Float64(reward),
            Vector{Float64}(next_belief),
            done
        )
        
        # Add to buffer
        push!(replay.buffer, exp)
        
        # Update size (capped at capacity)
        replay.size = min(replay.size + 1, replay.capacity)
        
    catch e
        error("Failed to add experience to replay buffer: $e")
    end
end

"""
    Base.length(replay::ExperienceReplay)

Get the current number of experiences in the buffer.

# Example
```julia
replay = ExperienceReplay(capacity=1000)
push!(replay, belief, action, reward, next_belief, done)
println(length(replay))  # Prints: 1
```
"""
Base.length(replay::ExperienceReplay) = replay.size

"""
    sample(replay::ExperienceReplay, batch_size::Int; rng=Random.GLOBAL_RNG)

Sample a random batch of experiences for training.

# Why Random Sampling?

Imagine studying flashcards:
- **In order**: You might memorize the sequence, not the content
- **Random**: You actually learn the material

Random sampling breaks correlations between consecutive experiences,
leading to more stable and effective learning.

# Arguments
- `replay`: The replay buffer
- `batch_size`: Number of experiences to sample
- `rng`: Random number generator (for reproducibility)

# Returns
- `Vector{Experience}`: Batch of randomly sampled experiences

# Error Handling
- Checks buffer has enough experiences
- Validates batch_size is reasonable
- Provides helpful error messages

# Example
```julia
replay = ExperienceReplay(capacity=1000)
# ... add many experiences ...

# Sample batch for training
batch = sample(replay, batch_size=32)
```
"""
function sample(replay::ExperienceReplay, batch_size::Int; rng=Random.GLOBAL_RNG)
    try
        # Validate we have enough experiences
        if replay.size < batch_size
            error("Not enough experiences to sample. " *
                  "Have $(replay.size), need $batch_size. " *
                  "Collect more experiences before training.")
        end
        
        # Warn if batch size is very large relative to buffer
        if batch_size > replay.size / 2
            @warn "Batch size ($batch_size) is more than half the buffer size ($(replay.size)). " *
                  "This might lead to overfitting. Consider using a smaller batch size."
        end
        
        # Sample random indices
        indices = StatsBase.sample(rng, 1:replay.size, batch_size, replace=false)
        
        # Get experiences at those indices
        batch = [replay.buffer[i] for i in indices]
        
        return batch
        
    catch e
        error("Experience sampling failed: $e")
    end
end

"""
    unpack_batch(batch::Vector{Experience})

Unpack a batch of experiences into separate arrays.

# Why Unpack?

Neural networks work with matrices, not individual experiences.
This function converts:
- List of experiences → Matrices for batch processing

Think of it like organizing groceries:
- Before: Mixed bag of items
- After: All apples together, all oranges together, etc.

# Arguments
- `batch`: Vector of experiences

# Returns
- `beliefs`: Matrix of belief states (belief_dim × batch_size)
- `actions`: Vector of actions
- `rewards`: Vector of rewards
- `next_beliefs`: Matrix of next belief states
- `dones`: Vector of done flags

# Example
```julia
batch = sample(replay, batch_size=32)
beliefs, actions, rewards, next_beliefs, dones = unpack_batch(batch)

# Now you can pass beliefs to neural network as a batch
embeddings = encoder.network(beliefs)
```
"""
function unpack_batch(batch::Vector{Experience})
    try
        @assert !isempty(batch) "Cannot unpack empty batch"
        
        # Extract components
        beliefs = hcat([exp.belief for exp in batch]...)
        actions = [exp.action for exp in batch]
        rewards = [exp.reward for exp in batch]
        next_beliefs = hcat([exp.next_belief for exp in batch]...)
        dones = [exp.done for exp in batch]
        
        return beliefs, actions, rewards, next_beliefs, dones
        
    catch e
        error("Batch unpacking failed: $e")
    end
end

"""
    clear!(replay::ExperienceReplay)

Clear all experiences from the replay buffer.

Useful when starting a new training run or resetting the environment.

# Example
```julia
replay = ExperienceReplay(capacity=1000)
# ... add experiences ...
clear!(replay)
println(length(replay))  # Prints: 0
```
"""
function clear!(replay::ExperienceReplay)
    empty!(replay.buffer)
    replay.size = 0
end

"""
    test_experience_replay()

Test experience replay functionality.

Run this to verify the replay buffer works correctly.
"""
function test_experience_replay()
    println("Testing ExperienceReplay...")
    
    try
        # Test 1: Creation
        replay = ExperienceReplay(capacity=100)
        @assert length(replay) == 0 "Initial size should be 0"
        println("✓ Replay buffer created")
        
        # Test 2: Adding experiences
        for i in 1:50
            push!(replay, rand(5), 1, 10.0, rand(5), false)
        end
        @assert length(replay) == 50 "Should have 50 experiences"
        println("✓ Adding experiences works")
        
        # Test 3: Sampling
        batch = sample(replay, 10)
        @assert length(batch) == 10 "Batch size should be 10"
        println("✓ Sampling works")
        
        # Test 4: Unpacking
        beliefs, actions, rewards, next_beliefs, dones = unpack_batch(batch)
        @assert size(beliefs, 2) == 10 "Should have 10 beliefs"
        @assert length(actions) == 10 "Should have 10 actions"
        println("✓ Unpacking works")
        
        # Test 5: Circular buffer (overfill)
        for i in 1:100
            push!(replay, rand(5), 1, 10.0, rand(5), false)
        end
        @assert length(replay) == 100 "Should cap at capacity"
        println("✓ Circular buffer works")
        
        # Test 6: Clear
        clear!(replay)
        @assert length(replay) == 0 "Should be empty after clear"
        println("✓ Clear works")
        
        println("\n✓ All experience replay tests passed!")
        return true
        
    catch e
        println("\n✗ Experience replay test failed: $e")
        return false
    end
end

# Export types and functions
export Experience, ExperienceReplay
export sample, unpack_batch, clear!
export test_experience_replay
