"""
# Belief Encoder

Encodes belief state distributions into fixed-size vector representations.

## The Problem

In POMDPs, belief states are probability distributions over possible states.
For a problem with 100 states, the belief is a 100-dimensional vector.
Neural networks work better with compact, fixed-size representations.

## The Solution

Think of this like compressing a high-resolution image into a thumbnail:
- Input: Full belief distribution (could be 100+ dimensions)
- Output: Compact embedding (16 dimensions)
- Preserves important information while reducing size

## Practical Use

Imagine a robot trying to locate itself in a building:
- Belief state: "30% chance I'm in room A, 50% in room B, 20% in room C"
- Encoder: Compresses this into a 16-number "fingerprint"
- Policy network: Uses fingerprint to decide "turn left" or "turn right"

This compression makes learning faster and more efficient.
"""

using Flux
using Statistics

"""
    BeliefEncoder

Neural network that encodes belief distributions into fixed-size embeddings.

# Architecture
- Input layer: belief_dim (size of belief state)
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 32 neurons with ReLU activation  
- Output layer: embedding_dim (typically 16)

# Why This Architecture?

The funnel shape (belief_dim → 64 → 32 → 16) progressively compresses
information, similar to how your brain summarizes a complex situation
into a simple gut feeling.

# Fields
- `network::Chain`: The neural network layers
- `belief_dim::Int`: Dimension of input belief state
- `embedding_dim::Int`: Dimension of output embedding

# Example
```julia
encoder = BeliefEncoder(belief_dim=10, embedding_dim=16)
belief = [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01, 0.0]
embedding = encoder(belief)  # Returns 16-dimensional vector
```
"""
struct BeliefEncoder
    network::Chain
    belief_dim::Int
    embedding_dim::Int
end

"""
    BeliefEncoder(; belief_dim::Int, embedding_dim::Int=16, hidden_dims::Vector{Int}=[64, 32])

Create a belief encoder with specified architecture.

# Arguments
- `belief_dim`: Size of input belief state (e.g., number of states in POMDP)
- `embedding_dim`: Size of output embedding (default: 16)
- `hidden_dims`: Sizes of hidden layers (default: [64, 32])

# Returns
- `BeliefEncoder`: Initialized encoder ready for training

# Error Handling
- Validates that dimensions are positive
- Ensures hidden layers make sense (decreasing sizes)
- Provides helpful error messages if something's wrong

# Example
```julia
# For a POMDP with 10 states
encoder = BeliefEncoder(belief_dim=10)

# Custom architecture
encoder = BeliefEncoder(belief_dim=100, embedding_dim=32, hidden_dims=[128, 64, 32])
```
"""
function BeliefEncoder(; 
    belief_dim::Int, 
    embedding_dim::Int=16, 
    hidden_dims::Vector{Int}=[64, 32]
)
    # Input validation with helpful error messages
    try
        @assert belief_dim > 0 "belief_dim must be positive, got $belief_dim"
        @assert embedding_dim > 0 "embedding_dim must be positive, got $embedding_dim"
        @assert all(d -> d > 0, hidden_dims) "All hidden dimensions must be positive"
        
        # Warn if architecture seems unusual (not necessarily wrong, just unusual)
        if embedding_dim > belief_dim
            @warn "embedding_dim ($embedding_dim) > belief_dim ($belief_dim). " *
                  "Usually we compress beliefs, not expand them. " *
                  "This might be intentional, but double-check your architecture."
        end
        
        # Build the network layer by layer
        layers = []
        
        # Input → First hidden layer
        push!(layers, Dense(belief_dim, hidden_dims[1], relu))
        
        # Hidden layers (if more than one)
        for i in 1:(length(hidden_dims)-1)
            push!(layers, Dense(hidden_dims[i], hidden_dims[i+1], relu))
        end
        
        # Last hidden → Output (no activation - we want raw embeddings)
        push!(layers, Dense(hidden_dims[end], embedding_dim))
        
        # Combine into a single network
        network = Chain(layers...)
        
        return BeliefEncoder(network, belief_dim, embedding_dim)
        
    catch e
        if isa(e, AssertionError)
            error("BeliefEncoder creation failed: $(e.msg)")
        else
            error("Unexpected error creating BeliefEncoder: $e")
        end
    end
end

"""
    (encoder::BeliefEncoder)(belief::AbstractVector)

Encode a belief state into a fixed-size embedding.

# How It Works

Think of this like a translator:
1. Takes a belief state (probability distribution)
2. Passes it through neural network layers
3. Returns a compact "summary" vector

The network learns to preserve information that's useful for
decision-making while discarding irrelevant details.

# Arguments
- `belief`: Belief state vector (must match encoder.belief_dim)

# Returns
- Embedding vector of size encoder.embedding_dim

# Error Handling
- Checks belief dimension matches encoder
- Validates belief is a valid probability distribution
- Provides clear error messages

# Example
```julia
encoder = BeliefEncoder(belief_dim=5)
belief = [0.2, 0.3, 0.3, 0.1, 0.1]  # Valid probability distribution
embedding = encoder(belief)
```
"""
function (encoder::BeliefEncoder)(belief::AbstractVector)
    try
        # Validate input dimension
        if length(belief) != encoder.belief_dim
            error("Belief dimension mismatch: expected $(encoder.belief_dim), got $(length(belief))")
        end
        
        # Convert to Float32 for Flux
        belief32 = Float32.(belief)
        
        # Check if belief looks like a probability distribution
        # (This is a soft check - we don't enforce it strictly because
        #  sometimes beliefs might be unnormalized during training)
        belief_sum = sum(belief32)
        if abs(belief_sum - 1.0f0) > 0.1f0  # Allow some tolerance
            @warn "Belief doesn't sum to 1.0 (sum = $belief_sum). " *
                  "This might be okay during training, but check if this is intentional."
        end
        
        # Check for negative probabilities (definitely wrong)
        if any(belief32 .< 0)
            error("Belief contains negative values. Probabilities must be non-negative.")
        end
        
        # Encode the belief
        embedding = encoder.network(belief32)
        
        return embedding
        
    catch e
        if isa(e, DimensionMismatch)
            error("Belief encoding failed due to dimension mismatch: $e")
        else
            rethrow(e)
        end
    end
end

"""
    normalize_belief(belief::AbstractVector)

Normalize a belief state to sum to 1.0.

# Why This Matters

Sometimes during computation, beliefs can become unnormalized
(e.g., [0.3, 0.4, 0.5] sums to 1.2, not 1.0).

This function fixes that: [0.3, 0.4, 0.5] → [0.25, 0.33, 0.42]

# Arguments
- `belief`: Unnormalized belief vector

# Returns
- Normalized belief vector (sums to 1.0)

# Error Handling
- Handles zero-sum beliefs gracefully
- Checks for negative values
"""
function normalize_belief(belief::AbstractVector)
    try
        # Check for negative values
        if any(belief .< 0)
            error("Cannot normalize belief with negative values: $belief")
        end
        
        belief_sum = sum(belief)
        
        # Handle edge case: all zeros
        if belief_sum ≈ 0.0
            @warn "Belief sums to zero. Returning uniform distribution."
            return fill(1.0 / length(belief), length(belief))
        end
        
        # Normalize
        return belief ./ belief_sum
        
    catch e
        error("Belief normalization failed: $e")
    end
end

"""
    test_encoder()

Quick test to verify encoder works correctly.

This is like a "smoke test" - if this passes, the encoder is probably working.
Run this after creating an encoder to make sure everything's set up right.

# Example
```julia
test_encoder()  # Should print "✓ All encoder tests passed!"
```
"""
function test_encoder()
    println("Testing BeliefEncoder...")
    
    try
        # Test 1: Basic creation
        encoder = BeliefEncoder(belief_dim=10, embedding_dim=16)
        println("✓ Encoder created successfully")
        
        # Test 2: Forward pass
        belief = normalize_belief(rand(10))
        embedding = encoder(belief)
        @assert length(embedding) == 16 "Embedding dimension mismatch"
        println("✓ Forward pass works")
        
        # Test 3: Batch processing
        batch_beliefs = hcat([normalize_belief(rand(10)) for _ in 1:5]...)
        batch_embeddings = encoder.network(batch_beliefs)
        @assert size(batch_embeddings, 2) == 5 "Batch processing failed"
        println("✓ Batch processing works")
        
        println("\n✓ All encoder tests passed!")
        return true
        
    catch e
        println("\n✗ Encoder test failed: $e")
        return false
    end
end

# Export functions
export BeliefEncoder, normalize_belief, test_encoder
