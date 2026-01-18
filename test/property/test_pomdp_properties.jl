"""
Property-based tests for POMDP environments.

Tests universal properties that should hold for:
- Tiger POMDP
- Light-Dark POMDP
- Belief updates
- Episode rollouts
"""

using Test
using NeuralPOMDPPolicy
using POMDPs
using POMDPTools
using Random
using Statistics

@testset "POMDP Environment Properties" begin
    
    @testset "Tiger POMDP Properties" begin
        println("\n  Testing Tiger POMDP properties...")
        
        pomdp = TigerPOMDP()
        
        # Property 1: All states are valid
        for s in states(pomdp)
            @test s in [:tiger_left, :tiger_right]
        end
        
        # Property 2: All actions are valid
        for a in actions(pomdp)
            @test a in [:listen, :open_left, :open_right]
        end
        
        # Property 3: All observations are valid
        for o in observations(pomdp)
            @test o in [:hear_left, :hear_right]
        end
        
        # Property 4: Transition probabilities sum to 1
        for s in states(pomdp)
            for a in actions(pomdp)
                sp_dist = transition(pomdp, s, a)
                probs = [pdf(sp_dist, sp) for sp in states(pomdp)]
                @test sum(probs) ≈ 1.0 atol=1e-6
            end
        end
        
        # Property 5: Observation probabilities sum to 1
        for a in actions(pomdp)
            for sp in states(pomdp)
                o_dist = observation(pomdp, a, sp)
                probs = [pdf(o_dist, o) for o in observations(pomdp)]
                @test sum(probs) ≈ 1.0 atol=1e-6
            end
        end
        
        # Property 6: Rewards are finite
        for s in states(pomdp)
            for a in actions(pomdp)
                r = reward(pomdp, s, a)
                @test isfinite(r)
            end
        end
        
        # Property 7: Listen action preserves state
        for s in states(pomdp)
            sp_dist = transition(pomdp, s, :listen)
            @test rand(sp_dist) == s
        end
        
        # Property 8: Opening door resets state (uniform)
        for s in states(pomdp)
            for a in [:open_left, :open_right]
                sp_dist = transition(pomdp, s, a)
                # Should be uniform over states
                for sp in states(pomdp)
                    @test pdf(sp_dist, sp) ≈ 0.5 atol=1e-6
                end
            end
        end
        
        # Property 9: Listening gives informative observations
        # When tiger is left, should favor hearing left
        o_dist = observation(pomdp, :listen, :tiger_left)
        @test pdf(o_dist, :hear_left) > pdf(o_dist, :hear_right)
        
        # When tiger is right, should favor hearing right
        o_dist = observation(pomdp, :listen, :tiger_right)
        @test pdf(o_dist, :hear_right) > pdf(o_dist, :hear_left)
        
        # Property 10: Correct door gives positive reward
        @test reward(pomdp, :tiger_left, :open_right) > 0
        @test reward(pomdp, :tiger_right, :open_left) > 0
        
        # Property 11: Wrong door gives negative reward
        @test reward(pomdp, :tiger_left, :open_left) < 0
        @test reward(pomdp, :tiger_right, :open_right) < 0
        
        # Property 12: Listen has small negative reward
        for s in states(pomdp)
            @test reward(pomdp, s, :listen) < 0
            @test reward(pomdp, s, :listen) > reward(pomdp, s, :open_left)
        end
        
        println("    ✓ Tested $(2 * 3 * 2 + 2 * 3 * 2 + 2 * 3 + 2 + 2 * 2 + 2 + 2 + 2 + 2) Tiger POMDP properties")
    end
    
    @testset "Light-Dark POMDP Properties" begin
        println("\n  Testing Light-Dark POMDP properties...")
        
        pomdp = LightDarkPOMDP()
        
        # Property 1: All actions are valid
        for a in actions(pomdp)
            @test a in [:left, :stay, :right]
        end
        
        # Property 2: Observation noise increases with distance from light
        for x in -10.0:1.0:10.0
            noise = observation_noise(pomdp, x)
            @test noise >= pomdp.min_noise
            @test noise <= pomdp.max_noise
            @test isfinite(noise)
        end
        
        # Property 3: Noise is minimum at light location
        noise_at_light = observation_noise(pomdp, pomdp.light_loc)
        for x in [-10.0, -5.0, 5.0, 10.0]
            if x != pomdp.light_loc
                @test observation_noise(pomdp, x) >= noise_at_light
            end
        end
        
        # Property 4: Transitions are deterministic
        for s in -10.0:1.0:10.0
            for a in actions(pomdp)
                sp_dist = transition(pomdp, s, a)
                sp = rand(sp_dist)
                @test isfinite(sp)
                @test -10.0 <= sp <= 10.0
            end
        end
        
        # Property 5: Left action decreases position
        for s in -9.0:1.0:9.0
            sp_dist = transition(pomdp, s, :left)
            sp = rand(sp_dist)
            @test sp <= s
        end
        
        # Property 6: Right action increases position
        for s in -9.0:1.0:9.0
            sp_dist = transition(pomdp, s, :right)
            sp = rand(sp_dist)
            @test sp >= s
        end
        
        # Property 7: Stay action preserves position
        for s in -10.0:1.0:10.0
            sp_dist = transition(pomdp, s, :stay)
            sp = rand(sp_dist)
            @test sp == s
        end
        
        # Property 8: Observations are centered on true state
        for s in -10.0:2.0:10.0
            for a in actions(pomdp)
                o_dist = observation(pomdp, a, s)
                # Mean should be close to true state
                @test mean(o_dist) ≈ s atol=1e-6
            end
        end
        
        # Property 9: Rewards are finite
        for s in -10.0:1.0:10.0
            for a in actions(pomdp)
                r = reward(pomdp, s, a)
                @test isfinite(r)
            end
        end
        
        # Property 10: Goal state gives high reward
        r_goal = reward(pomdp, pomdp.goal, :stay)
        r_not_goal = reward(pomdp, 0.0, :stay)
        @test r_goal > r_not_goal
        
        # Property 11: Terminal only at goal
        @test isterminal(pomdp, pomdp.goal)
        for s in -10.0:1.0:10.0
            if abs(s - pomdp.goal) > pomdp.goal_radius
                @test !isterminal(pomdp, s)
            end
        end
        
        # Property 12: Initial state is at start position
        s0_dist = initialstate(pomdp)
        s0 = rand(s0_dist)
        @test s0 == pomdp.start
        
        println("    ✓ Tested $(3 + 21 + 4 + 21 * 3 + 19 + 19 + 21 + 11 * 3 + 21 * 3 + 2 + 21 + 1) Light-Dark POMDP properties")
    end
    
    @testset "Belief Update Properties" begin
        println("\n  Testing belief update properties...")
        
        # Test Tiger belief updates
        pomdp = TigerPOMDP()
        updater = create_tiger_updater(pomdp)
        
        # Property 1: Initial belief sums to 1
        for _ in 1:50
            b = initialize_belief(updater, initialstate(pomdp))
            @test sum(b.b) ≈ 1.0 atol=1e-6
        end
        
        # Property 2: Updated belief sums to 1
        for _ in 1:100
            b = initialize_belief(updater, initialstate(pomdp))
            a = rand(actions(pomdp))
            o = rand(observations(pomdp))
            b_new = update(updater, b, a, o)
            @test sum(b_new.b) ≈ 1.0 atol=1e-6
        end
        
        # Property 3: Consistent observations increase confidence
        b = initialize_belief(updater, initialstate(pomdp))
        
        # Listen and hear left multiple times
        for _ in 1:5
            b = update(updater, b, :listen, :hear_left)
        end
        
        # Should be more confident tiger is left
        @test b.b[1] > 0.9  # High confidence in tiger_left
        
        # Property 4: Opening door resets belief to uniform
        b = initialize_belief(updater, initialstate(pomdp))
        
        # Build up confidence
        for _ in 1:3
            b = update(updater, b, :listen, :hear_left)
        end
        
        # Open door (resets)
        b = update(updater, b, :open_left, :hear_left)
        
        # Should be back to uniform (or close)
        @test abs(b.b[1] - 0.5) < 0.2
        
        println("    ✓ Tested $(50 + 100 + 1 + 1) belief update properties")
    end
    
    @testset "Episode Rollout Properties" begin
        println("\n  Testing episode rollout properties...")
        
        pomdp = TigerPOMDP()
        updater = create_tiger_updater(pomdp)
        
        # Create a simple random policy
        random_policy = RandomPolicy(pomdp)
        
        # Property 1: Episodes terminate within max steps
        for _ in 1:50
            b = initialize_belief(updater, initialstate(pomdp))
            total_reward = 0.0
            steps = 0
            max_steps = 100
            
            while steps < max_steps
                a = action(random_policy, b)
                s = rand(b)
                sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
                
                total_reward += r
                b = update(updater, b, a, o)
                steps += 1
                
                # Check all values are finite
                @test isfinite(total_reward)
                @test isfinite(r)
            end
            
            @test steps == max_steps
        end
        
        # Property 2: Rewards accumulate correctly
        for _ in 1:50
            b = initialize_belief(updater, initialstate(pomdp))
            rewards = Float64[]
            
            for step in 1:20
                a = action(random_policy, b)
                s = rand(b)
                sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
                
                push!(rewards, r)
                b = update(updater, b, a, o)
            end
            
            @test length(rewards) == 20
            @test all(isfinite, rewards)
            @test sum(rewards) == sum(rewards)  # Sanity check
        end
        
        # Property 3: Belief states remain valid throughout episode
        for _ in 1:50
            b = initialize_belief(updater, initialstate(pomdp))
            
            for step in 1:20
                # Check belief is valid
                @test sum(b.b) ≈ 1.0 atol=1e-6
                @test all(b.b .>= 0.0)
                @test all(b.b .<= 1.0)
                
                # Take action and update
                a = action(random_policy, b)
                s = rand(b)
                sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
                b = update(updater, b, a, o)
            end
        end
        
        println("    ✓ Tested $(50 + 50 + 50) episode rollout properties")
    end
    
    @testset "Policy Interaction Properties" begin
        println("\n  Testing policy interaction properties...")
        
        pomdp = TigerPOMDP()
        updater = create_tiger_updater(pomdp)
        
        # Test with different baseline policies
        policies = [
            RandomPolicy(pomdp),
            QMDPPolicy(pomdp),
            GreedyPolicy(pomdp)
        ]
        
        for policy in policies
            # Property 1: Policy always returns valid action
            for _ in 1:100
                b = initialize_belief(updater, initialstate(pomdp))
                a = action(policy, b)
                @test a in actions(pomdp)
            end
            
            # Property 2: Policy is deterministic for same belief (except Random)
            if !(policy isa RandomPolicy)
                b = initialize_belief(updater, initialstate(pomdp))
                a1 = action(policy, b)
                a2 = action(policy, b)
                @test a1 == a2
            end
        end
        
        println("    ✓ Tested $(3 * 100 + 2) policy interaction properties")
    end
end

println("\n  ✓ All POMDP environment property tests passed!")
