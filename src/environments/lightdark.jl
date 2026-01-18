"""
Light-Dark Navigation - Continuous observation POMDP.

## The Problem

Navigate from start position to goal position on a 1D line. The catch?
Observations are noisy, and the noise depends on your position:

- **Light region** (near x=0): Low noise, you know where you are
- **Dark region** (far from x=0): High noise, very uncertain

**Key Challenge**: Do you take the direct path (through darkness) or detour
through the light to localize yourself first?

Think of it like: walking home in the dark. Do you take the shortcut through
the unlit alley (risky, might get lost), or go the long way under streetlights
(safer, you know where you are)?

## Why This Problem?

1. **Continuous observations**: Unlike Tiger (discrete), this has real-valued obs
2. **Information gathering**: Tests active sensing (going to light to localize)
3. **Exploration-exploitation**: Balance speed vs certainty
4. **Realistic**: Many robotics problems have this structure

## Reference
Platt et al. (2010). "Belief space planning assuming maximum likelihood
observations." Robotics: Science and Systems.
"""

using POMDPs
using POMDPTools
using Random
using Distributions
using LinearAlgebra

"""
    LightDarkPOMDP

Light-Dark navigation problem.

# Fields
- `goal::Float64`: Goal position (default: 5.0)
- `start::Float64`: Start position (default: -5.0)
- `light_loc::Float64`: Center of light region (default: 0.0)
- `min_noise::Float64`: Minimum observation noise (default: 0.1)
- `max_noise::Float64`: Maximum observation noise (default: 5.0)
- `step_size::Float64`: Movement per action (default: 1.0)
- `goal_radius::Float64`: Radius around goal that counts as success (default: 0.5)
- `discount::Float64`: Discount factor (default: 0.95)

# Example
```julia
pomdp = LightDarkPOMDP()
# Or customize:
pomdp = LightDarkPOMDP(goal=10.0, start=-10.0, step_size=0.5)
```
"""
struct LightDarkPOMDP <: POMDP{Float64, Symbol, Float64}
    goal::Float64
    start::Float64
    light_loc::Float64
    min_noise::Float64
    max_noise::Float64
    step_size::Float64
    goal_radius::Float64
    discount::Float64
end

function LightDarkPOMDP(;
    goal=5.0,
    start=-5.0,
    light_loc=0.0,
    min_noise=0.1,
    max_noise=5.0,
    step_size=1.0,
    goal_radius=0.5,
    discount=0.95
)
    return LightDarkPOMDP(
        goal, start, light_loc, min_noise, max_noise,
        step_size, goal_radius, discount
    )
end

# State space is continuous (position on line)
# We'll discretize for neural network purposes
POMDPs.states(pomdp::LightDarkPOMDP) = -10.0:0.5:10.0
POMDPs.stateindex(pomdp::LightDarkPOMDP, s::Float64) = Int(round((s + 10.0) / 0.5)) + 1

# Action space: move left, stay, move right
POMDPs.actions(pomdp::LightDarkPOMDP) = [:left, :stay, :right]
POMDPs.actionindex(pomdp::LightDarkPOMDP, a::Symbol) = findfirst(==(a), actions(pomdp))

# Observation space is continuous (noisy position measurement)
POMDPs.observations(pomdp::LightDarkPOMDP) = -15.0:0.1:15.0
POMDPs.obsindex(pomdp::LightDarkPOMDP, o::Float64) = Int(round((o + 15.0) / 0.1)) + 1

# Discount factor
POMDPs.discount(pomdp::LightDarkPOMDP) = pomdp.discount

"""
    observation_noise(pomdp::LightDarkPOMDP, s::Float64)

Compute observation noise at position s.

**Key insight**: Noise increases with distance from light source.

Think of it like: standing under a streetlight, you can see clearly (low noise).
Walk into darkness, everything becomes blurry (high noise).

# Formula
noise = min_noise + (max_noise - min_noise) * |s - light_loc| / 10

Near light (s ≈ 0): noise ≈ 0.1 (very accurate)
Far from light (s ≈ ±10): noise ≈ 5.0 (very uncertain)
"""
function observation_noise(pomdp::LightDarkPOMDP, s::Float64)
    distance_from_light = abs(s - pomdp.light_loc)
    # Noise increases linearly with distance from light
    noise = pomdp.min_noise + (pomdp.max_noise - pomdp.min_noise) * min(distance_from_light / 10.0, 1.0)
    return noise
end

"""
    transition(pomdp::LightDarkPOMDP, s::Float64, a::Symbol)

State transition function.

**Key insight**: Actions move you deterministically (no process noise).
The uncertainty comes from observations, not from movement.

Think of it like: you can walk straight, but in the dark you can't tell
exactly where you are.
"""
function POMDPs.transition(pomdp::LightDarkPOMDP, s::Float64, a::Symbol)
    if a == :left
        sp = s - pomdp.step_size
    elseif a == :right
        sp = s + pomdp.step_size
    else  # :stay
        sp = s
    end
    
    # Clamp to reasonable bounds
    sp = clamp(sp, -10.0, 10.0)
    
    return Deterministic(sp)
end

"""
    observation(pomdp::LightDarkPOMDP, a::Symbol, sp::Float64)

Observation function.

**Key insight**: You observe your position with noise that depends on
how far you are from the light.

Think of it like: looking at a GPS signal. In open areas (light), it's
accurate. In tunnels (dark), it's noisy and unreliable.
"""
function POMDPs.observation(pomdp::LightDarkPOMDP, a::Symbol, sp::Float64)
    noise_std = observation_noise(pomdp, sp)
    # Observation is true position + Gaussian noise
    return Normal(sp, noise_std)
end

"""
    reward(pomdp::LightDarkPOMDP, s::Float64, a::Symbol)

Reward function.

**Key insight**: Small penalty for each step (encourages efficiency),
big reward for reaching goal.

This creates the tradeoff: take risky shortcut (fewer steps, but might
miss goal) or safe detour (more steps, but confident you'll reach goal)?
"""
function POMDPs.reward(pomdp::LightDarkPOMDP, s::Float64, a::Symbol)
    # Check if at goal
    if abs(s - pomdp.goal) <= pomdp.goal_radius
        return 100.0  # Big reward for reaching goal
    else
        return -1.0  # Small penalty for each step (encourages efficiency)
    end
end

"""
    initialstate(pomdp::LightDarkPOMDP)

Initial state distribution.

Agent starts at the start position (deterministic).
"""
POMDPs.initialstate(pomdp::LightDarkPOMDP) = Deterministic(pomdp.start)

"""
    isterminal(pomdp::LightDarkPOMDP, s::Float64)

Check if state is terminal.

Episode ends when agent reaches goal.
"""
function POMDPs.isterminal(pomdp::LightDarkPOMDP, s::Float64)
    return abs(s - pomdp.goal) <= pomdp.goal_radius
end

"""
    create_lightdark_updater(pomdp::LightDarkPOMDP)

Create a particle filter updater for Light-Dark POMDP.

Since state space is continuous, we use particle filtering instead of
discrete belief updates.

Think of it like: instead of tracking exact probabilities, we maintain
a cloud of "guesses" (particles) about where we might be. Each observation
updates the cloud.

# Example
```julia
pomdp = LightDarkPOMDP()
updater = create_lightdark_updater(pomdp)

# Initialize belief (particles around start position)
b = initialize_belief(updater, initialstate(pomdp))

# Move and observe
a = :right
o = -3.5  # Noisy observation
b = update(updater, b, a, o)
# Particles now shifted right and reweighted based on observation
```
"""
function create_lightdark_updater(pomdp::LightDarkPOMDP; n_particles=1000)
    return BootstrapFilter(pomdp, n_particles)
end

"""
    optimal_lightdark_strategy()

Describe the optimal strategy for Light-Dark POMDP.

There's no closed-form optimal policy, but the general strategy is:

1. **If uncertain**: Move toward light to localize
2. **If confident**: Move directly toward goal
3. **Near goal**: Be extra careful (high cost of overshooting)

Neural policies should learn this information-gathering behavior.
"""
function optimal_lightdark_strategy()
    return """
    Optimal Light-Dark Strategy:
    
    1. Start at x=-5 (dark, uncertain)
    2. Move toward light (x=0) to reduce uncertainty
    3. Once confident in position, move toward goal (x=5)
    4. Near goal, move carefully to avoid overshooting
    
    Key insight: Detour through light is worth it for localization!
    
    Expected reward: ~80-90 (depends on parameters)
    Neural policies typically achieve 60-75% of optimal.
    """
end

"""
    test_lightdark_pomdp()

Test the Light-Dark POMDP implementation.

Verifies that the POMDP is correctly defined and observations work.
"""
function test_lightdark_pomdp()
    println("\n=== Testing Light-Dark POMDP ===\n")
    
    try
        # Create POMDP
        pomdp = LightDarkPOMDP()
        println("✓ LightDarkPOMDP created")
        
        # Check action space
        @assert length(actions(pomdp)) == 3
        println("✓ Action space: ", actions(pomdp))
        
        # Test observation noise
        noise_light = observation_noise(pomdp, 0.0)  # At light
        noise_dark = observation_noise(pomdp, 10.0)  # In darkness
        @assert noise_light < noise_dark
        println("✓ Observation noise (at light): ", noise_light)
        println("✓ Observation noise (in dark): ", noise_dark)
        
        # Test transition
        s = 0.0
        a = :right
        sp_dist = transition(pomdp, s, a)
        sp = rand(sp_dist)
        @assert sp == 1.0
        println("✓ Transition (right from 0): ", sp)
        
        # Test observation
        sp = 0.0
        o_dist = observation(pomdp, a, sp)
        o = rand(o_dist)
        println("✓ Observation (at light): ", o, " (should be near 0)")
        
        sp = 10.0
        o_dist = observation(pomdp, a, sp)
        o = rand(o_dist)
        println("✓ Observation (in dark): ", o, " (should be noisy)")
        
        # Test reward
        r = reward(pomdp, 0.0, :right)
        @assert r == -1.0
        println("✓ Reward (not at goal): ", r)
        
        r = reward(pomdp, 5.0, :stay)
        @assert r == 100.0
        println("✓ Reward (at goal): ", r)
        
        # Test terminal
        @assert !isterminal(pomdp, 0.0)
        @assert isterminal(pomdp, 5.0)
        println("✓ Terminal state detection works")
        
        println("\n✓ All Light-Dark POMDP tests passed!")
        println("\n", optimal_lightdark_strategy())
        
    catch e
        println("✗ Light-Dark POMDP test failed: ", e)
        rethrow(e)
    end
end
