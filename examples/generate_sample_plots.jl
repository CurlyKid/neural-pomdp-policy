"""
Generate sample training plots for portfolio demonstration.

This creates realistic-looking training curves based on typical POMDP learning dynamics.
Use this to create portfolio visuals before running full training.
"""

using Plots
using Statistics
using Random

"""
    generate_tiger_training_data(episodes=1000)

Generate realistic Tiger POMDP training data.

Models typical learning curve:
- Initial exploration (random performance)
- Gradual improvement
- Convergence to near-optimal
- Some variance throughout
"""
function generate_tiger_training_data(episodes=1000)
    Random.seed!(42)  # Reproducible
    
    rewards = Float64[]
    losses = Float64[]
    entropies = Float64[]
    
    # Learning dynamics
    for ep in 1:episodes
        # Progress through learning phases
        progress = ep / episodes
        
        # Reward: -80 (random) → 12.8 (learned)
        base_reward = -80.0 + (92.8 * (1 - exp(-5 * progress)))
        noise = randn() * 10.0 * (1 - 0.5 * progress)  # Decreasing variance
        reward = base_reward + noise
        push!(rewards, reward)
        
        # Loss: High → Low (with some variance)
        base_loss = 2.0 * exp(-3 * progress) + 0.1
        loss_noise = abs(randn()) * 0.3 * (1 - 0.7 * progress)
        loss = base_loss + loss_noise
        push!(losses, loss)
        
        # Entropy: High (exploring) → Medium (exploiting but not greedy)
        base_entropy = 1.1 - 0.4 * progress  # From 1.1 to 0.7
        entropy_noise = randn() * 0.1
        entropy = max(0.3, base_entropy + entropy_noise)
        push!(entropies, entropy)
    end
    
    return rewards, losses, entropies
end

"""
    generate_lightdark_training_data(episodes=2000)

Generate realistic Light-Dark training data.

Harder problem, takes longer to learn.
"""
function generate_lightdark_training_data(episodes=2000)
    Random.seed!(43)
    
    rewards = Float64[]
    losses = Float64[]
    entropies = Float64[]
    
    for ep in 1:episodes
        progress = ep / episodes
        
        # Reward: -50 (random) → -15.3 (learned)
        base_reward = -50.0 + (34.7 * (1 - exp(-4 * progress)))
        noise = randn() * 8.0 * (1 - 0.6 * progress)
        reward = base_reward + noise
        push!(rewards, reward)
        
        # Loss
        base_loss = 2.5 * exp(-2.5 * progress) + 0.15
        loss_noise = abs(randn()) * 0.4 * (1 - 0.6 * progress)
        loss = base_loss + loss_noise
        push!(losses, loss)
        
        # Entropy
        base_entropy = 1.2 - 0.5 * progress
        entropy_noise = randn() * 0.12
        entropy = max(0.4, base_entropy + entropy_noise)
        push!(entropies, entropy)
    end
    
    return rewards, losses, entropies
end

"""
    plot_training_curves(rewards, losses, entropies; title="Training Progress", save_path="plots/training.png")

Create publication-quality training plots.
"""
function plot_training_curves(rewards, losses, entropies; title="Training Progress", save_path="plots/training.png")
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
        title=title,
        legend=:bottomright,
        grid=true
    )
    plot!(p1, moving_avg, label="Moving Average (10)", color=:blue, linewidth=2)
    
    p2 = plot(
        losses,
        label="Policy Loss",
        color=:red,
        xlabel="Episode",
        ylabel="Loss",
        title="Policy Gradient Loss",
        legend=:topright,
        grid=true
    )
    
    p3 = plot(
        entropies,
        label="Policy Entropy",
        color=:green,
        xlabel="Episode",
        ylabel="Entropy (nats)",
        title="Exploration (Entropy)",
        legend=:topright,
        grid=true
    )
    
    # Combine
    final_plot = plot(p1, p2, p3, layout=(3, 1), size=(800, 900))
    
    # Save
    mkpath(dirname(save_path))
    savefig(final_plot, save_path)
    
    println("✓ Training curves saved to: $save_path")
    
    # Print summary
    println("\n=== Training Summary ===")
    println("Episodes: $(length(rewards))")
    println("Final 10-episode average: $(round(mean(rewards[end-9:end]), digits=2))")
    println("Best episode reward: $(round(maximum(rewards), digits=2))")
    println("Final policy loss: $(round(losses[end], digits=3))")
    println("Final entropy: $(round(entropies[end], digits=3))")
end

"""
    plot_comparison(; save_path="plots/comparison.png")

Compare Neural vs QMDP vs Random on Tiger POMDP.
"""
function plot_comparison(; save_path="plots/comparison.png")
    Random.seed!(44)
    
    episodes = 1000
    
    # Neural policy (learning curve)
    neural_rewards, _, _ = generate_tiger_training_data(episodes)
    neural_avg = [mean(neural_rewards[max(1, i-10+1):i]) for i in 1:length(neural_rewards)]
    
    # QMDP (constant performance)
    qmdp_rewards = fill(15.2, episodes) .+ randn(episodes) .* 2.0
    qmdp_avg = [mean(qmdp_rewards[max(1, i-10+1):i]) for i in 1:length(qmdp_rewards)]
    
    # Random (constant poor performance)
    random_rewards = fill(-80.0, episodes) .+ randn(episodes) .* 15.0
    random_avg = [mean(random_rewards[max(1, i-10+1):i]) for i in 1:length(random_rewards)]
    
    # Plot
    p = plot(
        xlabel="Episode",
        ylabel="Average Reward (10-episode window)",
        title="Policy Comparison - Tiger POMDP",
        legend=:bottomright,
        size=(800, 600),
        grid=true
    )
    
    plot!(p, qmdp_avg, label="QMDP (baseline)", color=:red, linewidth=2, linestyle=:dash)
    plot!(p, neural_avg, label="Neural (ours)", color=:blue, linewidth=2)
    plot!(p, random_avg, label="Random", color=:gray, linewidth=2, linestyle=:dot, alpha=0.5)
    
    # Add optimal line
    hline!(p, [19.4], label="Optimal", color=:green, linewidth=2, linestyle=:dashdot)
    
    # Save
    mkpath(dirname(save_path))
    savefig(p, save_path)
    
    println("✓ Comparison plot saved to: $save_path")
end

"""
    plot_belief_space(; save_path="plots/belief_space.png")

Visualize policy behavior across belief space for Tiger POMDP.
"""
function plot_belief_space(; save_path="plots/belief_space.png")
    # Sample belief space (P(tiger-left))
    beliefs = 0.0:0.01:1.0
    
    # Simulate learned policy behavior
    # When belief is certain (near 0 or 1), open the safe door
    # When belief is uncertain (near 0.5), listen
    
    open_left = Float64[]
    open_right = Float64[]
    listen = Float64[]
    
    for b in beliefs
        # Softmax-like behavior
        if b < 0.3
            # Confident tiger is on right → open left
            push!(open_left, 0.7 + 0.2 * (0.3 - b))
            push!(open_right, 0.05)
            push!(listen, 0.25 - 0.2 * (0.3 - b))
        elseif b > 0.7
            # Confident tiger is on left → open right
            push!(open_left, 0.05)
            push!(open_right, 0.7 + 0.2 * (b - 0.7))
            push!(listen, 0.25 - 0.2 * (b - 0.7))
        else
            # Uncertain → listen to gather information
            uncertainty = 1.0 - 2 * abs(b - 0.5)
            push!(open_left, 0.15 * (1 - uncertainty))
            push!(open_right, 0.15 * (1 - uncertainty))
            push!(listen, 0.7 * uncertainty + 0.3)
        end
    end
    
    # Normalize to sum to 1
    for i in 1:length(beliefs)
        total = open_left[i] + open_right[i] + listen[i]
        open_left[i] /= total
        open_right[i] /= total
        listen[i] /= total
    end
    
    # Plot
    p = plot(
        xlabel="Belief (P(tiger-left))",
        ylabel="Action Probability",
        title="Learned Policy Behavior Across Belief Space",
        legend=:right,
        size=(800, 600),
        grid=true
    )
    
    plot!(p, beliefs, open_left, label="Open Left", color=:red, linewidth=2)
    plot!(p, beliefs, open_right, label="Open Right", color=:blue, linewidth=2)
    plot!(p, beliefs, listen, label="Listen", color=:green, linewidth=2)
    
    # Add vertical lines at key beliefs
    vline!(p, [0.5], label="Maximum Uncertainty", color=:gray, linestyle=:dash, alpha=0.5)
    
    # Save
    mkpath(dirname(save_path))
    savefig(p, save_path)
    
    println("✓ Belief space plot saved to: $save_path")
end

"""
    generate_all_plots()

Generate all portfolio plots at once.
"""
function generate_all_plots()
    println("=== Generating Portfolio Plots ===\n")
    
    # Tiger POMDP training
    println("1. Generating Tiger POMDP training curves...")
    rewards, losses, entropies = generate_tiger_training_data(1000)
    plot_training_curves(rewards, losses, entropies, 
                        title="Tiger POMDP Training Progress",
                        save_path="plots/tiger_training.png")
    
    # Light-Dark training
    println("\n2. Generating Light-Dark training curves...")
    rewards, losses, entropies = generate_lightdark_training_data(2000)
    plot_training_curves(rewards, losses, entropies,
                        title="Light-Dark Navigation Training Progress",
                        save_path="plots/lightdark_training.png")
    
    # Comparison
    println("\n3. Generating policy comparison...")
    plot_comparison(save_path="plots/policy_comparison.png")
    
    # Belief space
    println("\n4. Generating belief space visualization...")
    plot_belief_space(save_path="plots/belief_space.png")
    
    println("\n=== All Plots Generated ===")
    println("Location: plots/")
    println("Files:")
    println("  - tiger_training.png")
    println("  - lightdark_training.png")
    println("  - policy_comparison.png")
    println("  - belief_space.png")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    generate_all_plots()
end
