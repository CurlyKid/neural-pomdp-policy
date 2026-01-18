"""
Visualize training curves from saved metrics.

This script loads training metrics and creates publication-quality plots
showing learning progress over time.
"""

using Plots
using Statistics
using BSON

"""
    plot_training_curves(metrics_file::String; save_path::String="plots/training_curves.png")

Create comprehensive training visualization.

# Arguments
- `metrics_file`: Path to saved metrics (BSON format)
- `save_path`: Where to save the plot

# Plots Created
1. Episode rewards over time
2. Moving average (window=10)
3. Policy loss
4. Policy entropy (exploration measure)
"""
function plot_training_curves(metrics_file::String; save_path::String="plots/training_curves.png")
    # Load metrics
    data = BSON.load(metrics_file)
    rewards = data[:episode_rewards]
    losses = data[:losses]
    entropies = data[:entropies]
    
    # Compute moving average
    window = 10
    moving_avg = [mean(rewards[max(1, i-window+1):i]) for i in 1:length(rewards)]
    
    # Create subplots
    p1 = plot(
        rewards,
        label="Episode Reward",
        alpha=0.3,
        color=:blue,
        xlabel="Episode",
        ylabel="Reward",
        title="Training Progress",
        legend=:bottomright
    )
    plot!(p1, moving_avg, label="Moving Average (10)", color=:blue, linewidth=2)
    
    p2 = plot(
        losses,
        label="Policy Loss",
        color=:red,
        xlabel="Episode",
        ylabel="Loss",
        title="Policy Gradient Loss",
        legend=:topright
    )
    
    p3 = plot(
        entropies,
        label="Policy Entropy",
        color=:green,
        xlabel="Episode",
        ylabel="Entropy",
        title="Exploration (Entropy)",
        legend=:topright
    )
    
    # Combine into single figure
    plot(p1, p2, p3, layout=(3, 1), size=(800, 900))
    
    # Save
    mkpath(dirname(save_path))
    savefig(save_path)
    
    println("✓ Training curves saved to: $save_path")
    
    # Print summary statistics
    println("\n=== Training Summary ===")
    println("Final 10-episode average: $(mean(rewards[end-9:end]))")
    println("Best episode reward: $(maximum(rewards))")
    println("Worst episode reward: $(minimum(rewards))")
    println("Final policy loss: $(losses[end])")
    println("Final entropy: $(entropies[end])")
end

"""
    plot_comparison(results::Dict; save_path::String="plots/comparison.png")

Compare multiple policies on same plot.

# Arguments
- `results`: Dict mapping policy names to reward vectors
- `save_path`: Where to save the plot

# Example
```julia
results = Dict(
    "Neural" => neural_rewards,
    "QMDP" => qmdp_rewards,
    "Random" => random_rewards
)
plot_comparison(results)
```
"""
function plot_comparison(results::Dict; save_path::String="plots/comparison.png")
    p = plot(
        xlabel="Episode",
        ylabel="Average Reward",
        title="Policy Comparison",
        legend=:bottomright,
        size=(800, 600)
    )
    
    colors = [:blue, :red, :green, :purple, :orange]
    
    for (i, (name, rewards)) in enumerate(results)
        # Compute moving average
        window = 10
        moving_avg = [mean(rewards[max(1, j-window+1):j]) for j in 1:length(rewards)]
        
        plot!(p, moving_avg, label=name, color=colors[i], linewidth=2)
    end
    
    # Save
    mkpath(dirname(save_path))
    savefig(save_path)
    
    println("✓ Comparison plot saved to: $save_path")
end

"""
    plot_belief_space(pomdp, policy; save_path::String="plots/belief_space.png")

Visualize policy behavior across belief space (2D only).

For Tiger POMDP, shows action probabilities as function of belief.
"""
function plot_belief_space(pomdp, policy; save_path::String="plots/belief_space.png")
    # Only works for 2-state POMDPs
    if length(states(pomdp)) != 2
        @warn "Belief space visualization only supports 2-state POMDPs"
        return
    end
    
    # Sample belief space
    beliefs = 0.0:0.01:1.0
    n_actions = length(actions(pomdp))
    
    # Compute action probabilities for each belief
    action_probs = zeros(length(beliefs), n_actions)
    
    for (i, b1) in enumerate(beliefs)
        b2 = 1.0 - b1
        belief_vec = Float32[b1, b2]
        
        # Get action probabilities
        probs = policy(belief_vec, return_probs=true)
        action_probs[i, :] = probs
    end
    
    # Plot
    p = plot(
        xlabel="Belief (P(tiger-left))",
        ylabel="Action Probability",
        title="Policy Behavior Across Belief Space",
        legend=:right,
        size=(800, 600)
    )
    
    action_names = ["Open Left", "Open Right", "Listen"]
    colors = [:red, :blue, :green]
    
    for (i, name) in enumerate(action_names)
        plot!(p, beliefs, action_probs[:, i], label=name, color=colors[i], linewidth=2)
    end
    
    # Save
    mkpath(dirname(save_path))
    savefig(save_path)
    
    println("✓ Belief space plot saved to: $save_path")
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    println("=== Training Visualization Example ===\n")
    
    # Check if metrics file exists
    metrics_file = "results/tiger_training.bson"
    
    if isfile(metrics_file)
        println("Loading metrics from: $metrics_file")
        plot_training_curves(metrics_file)
    else
        println("⚠ No metrics file found at: $metrics_file")
        println("Run examples/train_tiger.jl first to generate training data")
    end
end
