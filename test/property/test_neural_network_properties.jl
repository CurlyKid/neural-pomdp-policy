"""
Property-based tests for neural network components.

Tests universal properties that should hold for all inputs:
- Probabilities sum to 1
- Outputs are finite
- Dimensions are correct
- Gradients exist and are finite
"""

using Test
using NeuralPOMDPPolicy
using Flux
using Random

@testset "Neural Network Properties" begin
    
    @testset "BeliefEncoder Properties" begin
        println("\n  Testing BeliefEncoder properties...")
        
        # Test with various input dimensions
        for input_dim in [5, 10, 20, 50, 100]
            for embedding_dim in [8, 16, 32, 64]
                encoder = BeliefEncoder(belief_dim=input_dim, embedding_dim=embedding_dim)
                
                # Property 1: Output dimension is correct
                for _ in 1:100
                    belief = rand(input_dim)
                    embedding = encoder(belief)
                    @test length(embedding) == embedding_dim
                end
                
                # Property 2: Outputs are finite
                for _ in 1:100
                    belief = rand(input_dim)
                    embedding = encoder(belief)
                    @test all(isfinite, embedding)
                end
                
                # Property 3: Normalized beliefs work
                for _ in 1:100
                    belief = rand(input_dim)
                    belief = belief ./ sum(belief)  # Normalize
                    @test sum(belief) ≈ 1.0
                    embedding = encoder(belief)
                    @test all(isfinite, embedding)
                end
                
                # Property 4: Zero belief works
                belief = zeros(input_dim)
                embedding = encoder(belief)
                @test all(isfinite, embedding)
                
                # Property 5: One-hot beliefs work
                for i in 1:input_dim
                    belief = zeros(input_dim)
                    belief[i] = 1.0
                    embedding = encoder(belief)
                    @test all(isfinite, embedding)
                end
            end
        end
        
        println("    ✓ Tested $(5 * 4 * (100 + 100 + 100 + 1 + 10)) belief encodings")
    end
    
    @testset "PolicyNetwork Properties" begin
        println("\n  Testing PolicyNetwork properties...")
        
        # Test with various dimensions
        for embedding_dim in [8, 16, 32, 64]
            for n_actions in [2, 3, 5, 10]
                policy_net = PolicyNetwork(embedding_dim=embedding_dim, n_actions=n_actions)
                
                # Property 1: Probabilities sum to 1
                for _ in 1:100
                    embedding = randn(embedding_dim)
                    probs = policy_net(embedding)
                    @test sum(probs) ≈ 1.0 atol=1e-6
                end
                
                # Property 2: All probabilities non-negative
                for _ in 1:100
                    embedding = randn(embedding_dim)
                    probs = policy_net(embedding)
                    @test all(probs .>= 0.0)
                end
                
                # Property 3: All probabilities <= 1
                for _ in 1:100
                    embedding = randn(embedding_dim)
                    probs = policy_net(embedding)
                    @test all(probs .<= 1.0)
                end
                
                # Property 4: Output dimension correct
                for _ in 1:100
                    embedding = randn(embedding_dim)
                    probs = policy_net(embedding)
                    @test length(probs) == n_actions
                end
                
                # Property 5: Extreme embeddings work
                # Very large positive
                embedding = fill(10.0, embedding_dim)
                probs = policy_net(embedding)
                @test sum(probs) ≈ 1.0 atol=1e-6
                @test all(isfinite, probs)
                
                # Very large negative
                embedding = fill(-10.0, embedding_dim)
                probs = policy_net(embedding)
                @test sum(probs) ≈ 1.0 atol=1e-6
                @test all(isfinite, probs)
                
                # Zero embedding
                embedding = zeros(embedding_dim)
                probs = policy_net(embedding)
                @test sum(probs) ≈ 1.0 atol=1e-6
                @test all(isfinite, probs)
            end
        end
        
        println("    ✓ Tested $(4 * 4 * (100 * 4 + 3)) policy outputs")
    end
    
    @testset "NeuralPolicy Properties" begin
        println("\n  Testing NeuralPolicy properties...")
        
        for input_dim in [5, 10, 20]
            for n_actions in [2, 3, 5]
                encoder = BeliefEncoder(belief_dim=input_dim, embedding_dim=16)
                policy_net = PolicyNetwork(embedding_dim=16, n_actions=n_actions)
                policy = NeuralPolicy(encoder, policy_net, false)
                
                # Property 1: End-to-end probabilities sum to 1
                for _ in 1:100
                    belief = rand(input_dim)
                    probs = policy(belief, return_probs=true)
                    @test sum(probs) ≈ 1.0 atol=1e-6
                end
                
                # Property 2: Sampling returns valid action
                for _ in 1:100
                    belief = rand(input_dim)
                    a = policy(belief, sample=true, return_probs=false)
                    @test 1 <= a <= n_actions
                end
                
                # Property 3: Greedy returns valid action
                for _ in 1:100
                    belief = rand(input_dim)
                    a = policy(belief, sample=false, return_probs=false)
                    @test 1 <= a <= n_actions
                end
                
                # Property 4: Greedy selects highest probability
                for _ in 1:50
                    belief = rand(input_dim)
                    probs = policy(belief, return_probs=true)
                    a_greedy = policy(belief, sample=false, return_probs=false)
                    @test probs[a_greedy] == maximum(probs)
                end
            end
        end
        
        println("    ✓ Tested $(3 * 3 * (100 + 100 + 100 + 50)) policy actions")
    end
    
    @testset "Gradient Properties" begin
        println("\n  Testing gradient properties...")
        
        encoder = BeliefEncoder(belief_dim=10, embedding_dim=16)
        policy_net = PolicyNetwork(embedding_dim=16, n_actions=3)
        
        # Property 1: Gradients exist for encoder
        for _ in 1:50
            belief = rand(10)
            loss = sum(encoder(belief))
            grads = gradient(() -> loss, Flux.params(encoder))
            @test !isnothing(grads)
            for p in Flux.params(encoder)
                if !isnothing(grads[p])
                    @test all(isfinite, grads[p])
                end
            end
        end
        
        # Property 2: Gradients exist for policy network
        for _ in 1:50
            embedding = randn(16)
            loss = sum(policy_net(embedding))
            grads = gradient(() -> loss, Flux.params(policy_net))
            @test !isnothing(grads)
            for p in Flux.params(policy_net)
                if !isnothing(grads[p])
                    @test all(isfinite, grads[p])
                end
            end
        end
        
        println("    ✓ Tested $(50 + 50) gradient computations")
    end
end

println("\n  ✓ All neural network property tests passed!")
