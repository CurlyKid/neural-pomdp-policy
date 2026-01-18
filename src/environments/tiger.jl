"""
Tiger POMDP - Classic benchmark problem for POMDP algorithms.

## The Problem

You're standing in front of two doors. Behind one is a tiger (bad!), behind
the other is treasure (good!). You can't see which is which, but you can
listen to hear where the tiger is growling from.

**Actions**:
- Listen: Hear a noisy observation (85% accurate)
- Open Left: Open the left door
- Open Right: Open the right door

**Rewards**:
- Listen: -1 (costs time/effort)
- Open correct door: +10 (treasure!)
- Open wrong door: -100 (tiger eats you!)

**Key Challenge**: Balance exploration (listening to gather info) vs
exploitation (opening a door when confident).

This is a classic test because:
1. Simple enough to understand
2. Complex enough to require reasoning under uncertainty
3. Has an optimal solution we can compare against
4. Tests belief state reasoning

## Reference
Kaelbling, Littman, Cassandra (1998). "Planning and acting in partially
observable stochastic domains." Artificial Intelligence.
"""

using POMDPs
using POMDPTools
using POMDPTools: Uniform  # Explicit import to avoid ambiguity
using Random
using Distributions

"""
    TigerPOMDP

The Tiger POMDP problem.

# Fields
- `r_listen::Float64`: Reward for listening (default: -1.0)
- `r_findtiger::Float64`: Reward for opening tiger door (default: -100.0)
- `r_findtreasure::Float64`: Reward for opening treasure door (default: 10.0)
- `p_listen_correct::Float64`: Probability of correct observation (default: 0.85)
- `discount::Float64`: Discount factor (default: 0.95)

# Example
```julia
pomdp = TigerPOMDP()
# Or customize rewards:
pomdp = TigerPOMDP(r_listen=-2.0, r_findtreasure=20.0)
```
"""
struct TigerPOMDP <: POMDP{Symbol, Symbol, Symbol}
    r_listen::Float64
    r_findtiger::Float64
    r_findtreasure::Float64
    p_listen_correct::Float64
    discount::Float64
end

function TigerPOMDP(;
    r_listen=-1.0,
    r_findtiger=-100.0,
    r_findtreasure=10.0,
    p_listen_correct=0.85,
    discount=0.95
)
    return TigerPOMDP(r_listen, r_findtiger, r_findtreasure, p_listen_correct, discount)
end

# Define state space
# :tiger_left means tiger is behind left door (treasure behind right)
# :tiger_right means tiger is behind right door (treasure behind left)
POMDPs.states(pomdp::TigerPOMDP) = [:tiger_left, :tiger_right]
POMDPs.stateindex(pomdp::TigerPOMDP, s::Symbol) = s == :tiger_left ? 1 : 2

# Define action space
# :listen = gather information
# :open_left = open left door
# :open_right = open right door
POMDPs.actions(pomdp::TigerPOMDP) = [:listen, :open_left, :open_right]
POMDPs.actionindex(pomdp::TigerPOMDP, a::Symbol) = findfirst(==(a), actions(pomdp))

# Define observation space
# :hear_left = heard growling from left
# :hear_right = heard growling from right
POMDPs.observations(pomdp::TigerPOMDP) = [:hear_left, :hear_right]
POMDPs.obsindex(pomdp::TigerPOMDP, o::Symbol) = o == :hear_left ? 1 : 2

# Discount factor
POMDPs.discount(pomdp::TigerPOMDP) = pomdp.discount

"""
    transition(pomdp::TigerPOMDP, s::Symbol, a::Symbol)

State transition function.

**Key insight**: Opening a door resets the problem (tiger moves randomly).
Listening doesn't change the state (tiger stays put).

Think of it like: if you open a door, the game resets with tiger in a new
random position. If you just listen, tiger doesn't move.
"""
function POMDPs.transition(pomdp::TigerPOMDP, s::Symbol, a::Symbol)
    if a == :listen
        # Listening doesn't change state - tiger stays where it is
        return Deterministic(s)
    else
        # Opening a door resets the problem
        # Tiger randomly appears behind left or right door (50/50)
        return Uniform(states(pomdp))
    end
end

"""
    observation(pomdp::TigerPOMDP, a::Symbol, sp::Symbol)

Observation function.

**Key insight**: Listening gives noisy information (85% accurate).
Opening a door gives no information (you already committed!).

Think of it like: when you listen, you hear growling but it's muffled
(might be wrong). When you open a door, you don't hear anything because
you're already committed to that choice.
"""
function POMDPs.observation(pomdp::TigerPOMDP, a::Symbol, sp::Symbol)
    if a == :listen
        # Listening gives noisy observation
        # 85% chance of hearing correct side, 15% chance of hearing wrong side
        if sp == :tiger_left
            return SparseCat([:hear_left, :hear_right], 
                           [pomdp.p_listen_correct, 1.0 - pomdp.p_listen_correct])
        else  # sp == :tiger_right
            return SparseCat([:hear_left, :hear_right],
                           [1.0 - pomdp.p_listen_correct, pomdp.p_listen_correct])
        end
    else
        # Opening a door gives no observation (you already acted!)
        # Return uniform distribution (no information)
        return Uniform(observations(pomdp))
    end
end

"""
    reward(pomdp::TigerPOMDP, s::Symbol, a::Symbol)

Reward function.

**Key insight**: Listening costs a little, opening wrong door costs a lot,
opening right door gives big reward.

This creates the exploration-exploitation tradeoff: listen more to be sure
(but pay listening cost), or open early and risk the tiger?
"""
function POMDPs.reward(pomdp::TigerPOMDP, s::Symbol, a::Symbol)
    if a == :listen
        # Listening costs time/effort
        return pomdp.r_listen
    elseif a == :open_left
        # Opening left door
        if s == :tiger_left
            return pomdp.r_findtiger  # Oops, tiger!
        else
            return pomdp.r_findtreasure  # Treasure!
        end
    else  # a == :open_right
        # Opening right door
        if s == :tiger_right
            return pomdp.r_findtiger  # Oops, tiger!
        else
            return pomdp.r_findtreasure  # Treasure!
        end
    end
end

"""
    initialstate(pomdp::TigerPOMDP)

Initial state distribution.

Tiger starts behind left or right door with equal probability (50/50).
This represents complete uncertainty at the start.
"""
POMDPs.initialstate(pomdp::TigerPOMDP) = Uniform(states(pomdp))

"""
    isterminal(pomdp::TigerPOMDP, s::Symbol)

Check if state is terminal.

In standard Tiger POMDP, no states are terminal (game continues forever).
Each time you open a door, the problem resets.

For training purposes, we typically limit episode length externally.
"""
POMDPs.isterminal(pomdp::TigerPOMDP, s::Symbol) = false

"""
    create_tiger_updater(pomdp::TigerPOMDP)

Create a discrete belief updater for Tiger POMDP.

The updater maintains a probability distribution over states (belief state).
After each observation, it updates the belief using Bayes' rule.

Think of it like: you start 50/50 uncertain. Each time you listen, you
update your belief based on what you heard. If you hear left 3 times in a
row, you become very confident tiger is on the left.

# Example
```julia
pomdp = TigerPOMDP()
updater = create_tiger_updater(pomdp)

# Initialize belief (50/50 uncertain)
b = initialize_belief(updater, initialstate(pomdp))

# Listen and update belief
a = :listen
o = :hear_left  # Heard growling from left
b = update(updater, b, a, o)
# Now belief favors tiger_left (maybe 85/15)
```
"""
function create_tiger_updater(pomdp::TigerPOMDP)
    return DiscreteUpdater(pomdp)
end

"""
    optimal_tiger_value()

Return the optimal value for Tiger POMDP with default parameters.

This is computed using exact POMDP solvers (SARSOP, QMDP, etc.).
We use it as a baseline to compare neural policy performance.

For default parameters:
- Optimal value ≈ 19.4 (starting from uniform belief)
- Optimal policy: Listen 1-2 times, then open the more likely door

Neural policies typically achieve 70-85% of optimal (13-16 reward).
"""
function optimal_tiger_value()
    # This is the known optimal value for default Tiger POMDP
    # Computed using SARSOP solver
    return 19.4
end

"""
    test_tiger_pomdp()

Test the Tiger POMDP implementation.

Verifies that the POMDP is correctly defined and belief updates work.
"""
function test_tiger_pomdp()
    println("\n=== Testing Tiger POMDP ===\n")
    
    try
        # Create POMDP
        pomdp = TigerPOMDP()
        println("✓ TigerPOMDP created")
        
        # Check state space
        @assert length(states(pomdp)) == 2
        println("✓ State space: ", states(pomdp))
        
        # Check action space
        @assert length(actions(pomdp)) == 3
        println("✓ Action space: ", actions(pomdp))
        
        # Check observation space
        @assert length(observations(pomdp)) == 2
        println("✓ Observation space: ", observations(pomdp))
        
        # Test transition
        s = :tiger_left
        a = :listen
        sp_dist = transition(pomdp, s, a)
        println("✓ Transition (listen): ", sp_dist)
        
        a = :open_left
        sp_dist = transition(pomdp, s, a)
        println("✓ Transition (open): ", sp_dist)
        
        # Test observation
        a = :listen
        sp = :tiger_left
        o_dist = observation(pomdp, a, sp)
        println("✓ Observation (listen, tiger_left): ", o_dist)
        
        # Test reward
        r = reward(pomdp, :tiger_left, :listen)
        @assert r == -1.0
        println("✓ Reward (listen): ", r)
        
        r = reward(pomdp, :tiger_left, :open_left)
        @assert r == -100.0
        println("✓ Reward (open wrong door): ", r)
        
        r = reward(pomdp, :tiger_left, :open_right)
        @assert r == 10.0
        println("✓ Reward (open correct door): ", r)
        
        # Test belief updater
        updater = create_tiger_updater(pomdp)
        b = initialize_belief(updater, initialstate(pomdp))
        println("✓ Initial belief: ", b)
        
        # Simulate belief update
        a = :listen
        o = :hear_left
        b = update(updater, b, a, o)
        println("✓ Updated belief (heard left): ", b)
        
        println("\n✓ All Tiger POMDP tests passed!")
        println("  Optimal value: ", optimal_tiger_value())
        println("  Neural policies typically achieve 70-85% of optimal")
        
    catch e
        println("✗ Tiger POMDP test failed: ", e)
        rethrow(e)
    end
end
