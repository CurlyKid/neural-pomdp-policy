"""
Property-based tests for training components.

Tests universal properties for:
- Experience replay
- Return computation
- Policy gradients
- Training stability
"""

using Test
using NeuralPOMDPPolicy
using Random
using Statistics

@testset "Training Properties" begin
    
    @testset "ExperienceReplay Properties" begin
        println("\n  Testing ExperienceReplay properties...")
        
        for capacity in [10, 50, 100, 500]
            buffer = ExperienceReplay(capacity)
            
            # Property 1: Length never exceeds capacity
            for i in 1:capacity * 2
                belief = rand(10)
                action = rand(1:3)
                reward = randn()
                next_belief = rand(10)
                terminal = rand() < 0.1
                
                add_experience!(buffer, belief, action, reward, next_belief, terminal)
                @test length(buffer) <= capacity
            end
            
            # Property 2: Can sample up to buffer length
            for batch_size in [1, 5, 10, min(20, capacity)]
                if batch_size <= length(buffer)
                    batch = sample_batch(buffer, batch_size)
                    @test length(batch) == batch_size
                end
            end
            
            # Property 3: Sampled experiences are valid
            if length(buffer) >= 10
                batch = sample_batch(buffer, 10)
                beliefs, actions, rewards, next_beliefs, terminals = unpack_batch(batch)
                
                @test length(beliefs) == 10
                @test length(actions) == 10
                @test length(rewards) == 10
                @test length(next_beliefs) == 10
                @test length(terminals) == 10
                
                @test all(length(b) == 10 for b in beliefs)
                @test all(1 <= a <= 3 for a in actions)
                @test all(isfinite, rewards)
                @test all(length(b) == 10 for b in next_beliefs)
            end
        end
        
        println("    ✓ Tested $(4 * (2 * 100 + 4 + 1)) experience replay operations")
    end
    
    @testset "Return Computation Properties" begin
        println("\n  Testing return computation properties...")
        
        for gamma in [0.9, 0.95, 0.99]
            # Property 1: Returns have same length as rewards
            for T in [1, 5, 10, 50, 100]
                rewards = randn(T)
                returns = compute_returns(rewards, gamma, use_baseline=false)
                @test length(returns) == T
            end
            
            # Property 2: All returns are finite
            for _ in 1:100
                T = rand(1:50)
                rewards = randn(T)
                returns = compute_returns(rewards, gamma, use_baseline=false)
                @test all(isfinite, returns)
            end
            
            # Property 3: Last return >= last reward (for positive gamma)
            for _ in 1:100
                T = rand(5:20)
                rewards = randn(T)
                returns = compute_returns(rewards, gamma, use_baseline=false)
                if rewards[end] >= 0
                    @test returns[end] >= rewards[end]
                end
            end
            
            # Property 4: Returns decrease with lower gamma
            rewards = ones(10)
            returns_high = compute_returns(rewards, 0.99, use_baseline=false)
            returns_low = compute_returns(rewards, 0.5, use_baseline=false)
            @test returns_high[1] > returns_low[1]
            
            # Property 5: Baseline reduces variance
            for _ in 1:50
                T = rand(10:50)
                rewards = randn(T)
                returns_no_baseline = compute_returns(rewards, gamma, use_baseline=false)
                returns_with_baseline = compute_returns(rewards, gamma, use_baseline=true)
                
                # With baseline should have lower variance
                if T > 1
                    @test std(returns_with_baseline) <= std(returns_no_baseline) + 1e-6
                end
            end
        end
        
        println("    ✓ Tested $(3 * (5 + 100 + 100 + 1 + 50)) return computations")
    end
    
    @testset "Policy Gradient Properties" begin
        println("\n  Testing policy gradient properties...")
        
        encoder = BeliefEncoder(belief_dim=10, embedding_dim=16)
        policy_net = PolicyNetwork(embedding_dim=16, n_actions=3)
        policy = NeuralPolicy(encoder, policy_net, false)
        
        # Property 1: Loss is finite for valid inputs
        for _ in 1:100
            beliefs = [rand(10) for _ in 1:10]
            actions = rand(1:3, 10)
            returns = randn(10)
            
            loss, entropy = policy_gradient_loss(policy, beliefs, actions, returns)
            @test isfinite(loss)
            @test isfinite(entropy)
        end
        
        # Property 2: Entropy is non-negative
        for _ in 1:100
            beliefs = [rand(10) for _ in 1:10]
            actions = rand(1:3, 10)
            returns = randn(10)
            
            loss, entropy = policy_gradient_loss(policy, beliefs, actions, returns)
            @test entropy >= 0.0
        end
        
        # Property 3: Higher returns should decrease loss
        # (for actions with high probability)
        for _ in 1:50
            belief = rand(10)
            probs = policy(belief)
            best_action = argmax(probs)
            
            # High return for best action
            beliefs_high = [belief]
            actions_high = [best_action]
            returns_high = [10.0]
            loss_high, _ = policy_gradient_loss(policy, beliefs_high, actions_high, returns_high)
            
            # Low return for best action
            beliefs_low = [belief]
            actions_low = [best_action]
            returns_low = [-10.0]
            loss_low, _ = policy_gradient_loss(policy, beliefs_low, actions_low, returns_low)
            
            # High return should give lower (more negative) loss
            @test loss_high < loss_low
        end
        
        println("    ✓ Tested $(100 + 100 + 50) policy gradient computations")
    end
    
    @testset "Training Stability Properties" begin
        println("\n  Testing training stability properties...")
        
        # Property 1: Training doesn't explode
        for _ in 1:10
            encoder = BeliefEncoder(belief_dim=10, embedding_dim=16)
            policy_net = PolicyNetwork(embedding_dim=16, n_actions=3)
            policy = NeuralPolicy(encoder, policy_net, true)
            optimizer = Adam(0.001)
            
            # Simulate training steps
            for step in 1:20
                beliefs = [rand(10) for _ in 1:10]
                actions = rand(1:3, 10)
                rewards = randn(10)
                returns = compute_returns(rewards, 0.95)
                
                loss, entropy = policy_gradient_loss(policy, beliefs, actions, returns)
                
                # Check parameters don't explode
                for p in Flux.params(policy)
                    @test all(isfinite, p)
                end
                
                # Update
                grads = gradient(() -> policy_gradient_loss(policy, beliefs, actions, returns)[1],
                                Flux.params(policy))
                Flux.update!(optimizer, Flux.params(policy), grads)
            end
            
            # After training, parameters should still be finite
            for p in Flux.params(policy)
                @test all(isfinite, p)
            end
        end
        
        println("    ✓ Tested $(10 * 20) training steps for stability")
    end
    
    @testset "Convergence Detection Properties" begin
        println("\n  Testing convergence detection properties...")
        
        # Property 1: Constant rewards should converge
        metrics = TrainingMetrics()
        metrics.episode_rewards = repeat([10.0], 100)
        @test is_converged(metrics, window=20, threshold=0.05)
        
        # Property 2: Improving rewards should not converge early
        metrics = TrainingMetrics()
        metrics.episode_rewards = collect(1.0:100.0)
        @test !is_converged(metrics, window=20, threshold=0.05)
        
        # Property 3: Small oscillations should converge
        metrics = TrainingMetrics()
        metrics.episode_rewards = [10.0 + 0.1 * sin(i) for i in 1:100]
        @test is_converged(metrics, window=20, threshold=0.05)
        
        # Property 4: Large oscillations should not converge
        metrics = TrainingMetrics()
        metrics.episode_rewards = [10.0 + 5.0 * sin(i) for i in 1:100]
        @test !is_converged(metrics, window=20, threshold=0.05)
        
        println("    ✓ Tested 4 convergence scenarios")
    end
end

println("\n  ✓ All training property tests passed!")
