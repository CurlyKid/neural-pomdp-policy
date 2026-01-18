"""
Train a neural policy on the Tiger POMDP.

This example demonstrates:
1. Creating the Tiger POMDP environment
2. Training a neural policy with policy gradients
3. Evaluating performance vs baselines
4. Visualizing training progress

Expected results:
- Neural policy: ~13-16 reward (70-85% of optimal)
- QMDP baseline: ~15-17 reward (80-90% of optimal)
- Random baseline: ~-50 reward
- Optimal: ~19.4 reward

Run time: ~5-10 minutes on standard hardware
"""

using NeuralPOMDPPolicy
using POMDPs
using POMDPTools
using Plots

println("\n" * "="^70)
println("Training Neural Policy on Tiger POMDP")
println("="^70 * "\n")

# Create Tiger POMDP
println("Creating Tiger POMDP...")
pomdp = TigerPOMDP()
updater = create_tiger_updater(pomdp)

println("  States: ", states(pomdp))
println("  Actions: ", actions(pomdp))
println("  Observations: ", observations(pomdp))
println("  Optimal value: ", optimal_tiger_value())
println()

# Configure training
config = TrainingConfig(
    episodes=500,           # Number of training episodes
    learning_rate=0.001,    # Adam learning rate
    gamma=0.95,             # Discount factor
    max_steps=50,           # Max steps per episode
    use_baseline=true,      # Use baseline for variance reduction
    log_interval=25,        # Log every 25 episodes
    checkpoint_interval=100 # Checkpoint every 100 episodes
)

println("Training configuration:")
println("  Episodes: ", config.episodes)
println("  Learning rate: ", config.learning_rate)
println("  Discount factor: ", config.gamma)
println("  Max steps: ", config.max_steps)
println()

# Train neural policy
println("Training neural policy...")
println("(This may take 5-10 minutes)\n")

neural_policy, metrics = train_neural_policy(pomdp, updater, config=config)

println("\n✓ Training complete!")
println()

# Evaluate neural policy
println("Evaluating neural policy...")
neural_results = evaluate_policy(
    pomdp, neural_policy, updater,
    n_episodes=100,
    verbose=false
)
print_evaluation_summary(neural_results, name="Neural Policy")

# Compare with baselines
println("Evaluating baseline policies...")
random_policy = RandomPolicy(pomdp)
qmdp_policy = QMDPPolicy(pomdp)

results = compare_policies(
    pomdp,
    [neural_policy, qmdp_policy, random_policy],
    updater,
    n_episodes=100,
    names=["Neural", "QMDP", "Random"]
)

# Compare to optimal
optimal = optimal_tiger_value()
neural_pct = compare_to_baseline(neural_results, optimal)
println("\nPerformance vs Optimal:")
println("  Neural: ", round(neural_pct, digits=1), "% of optimal")
println("  Target: 70-85% of optimal")

if neural_pct >= 70.0
    println("  ✓ Neural policy meets target!")
else
    println("  ⚠ Neural policy below target (may need more training)")
end
println()

# Benchmark inference time
println("Benchmarking inference time...")
b0 = initialize_belief(updater, initialstate(pomdp))
mean_time = benchmark_inference_time(neural_policy, b0, n_trials=1000)

# Plot training curves
println("\nGenerating training plots...")

# Reward curve
p1 = plot(
    metrics.episode_rewards,
    xlabel="Episode",
    ylabel="Total Reward",
    title="Training Progress - Reward",
    label="Episode Reward",
    alpha=0.3,
    color=:blue
)

# Add moving average
window = 20
if length(metrics.episode_rewards) >= window
    moving_avg = [mean(metrics.episode_rewards[max(1, i-window+1):i]) 
                  for i in 1:length(metrics.episode_rewards)]
    plot!(p1, moving_avg, label="Moving Average", linewidth=2, color=:red)
end

# Add optimal line
hline!(p1, [optimal], label="Optimal", linestyle=:dash, color=:green, linewidth=2)

# Loss curve
p2 = plot(
    metrics.losses,
    xlabel="Episode",
    ylabel="Policy Gradient Loss",
    title="Training Progress - Loss",
    label="Loss",
    color=:orange,
    legend=:topright
)

# Entropy curve
p3 = plot(
    metrics.entropies,
    xlabel="Episode",
    ylabel="Policy Entropy",
    title="Training Progress - Entropy",
    label="Entropy",
    color=:purple,
    legend=:topright
)

# Episode length curve
p4 = plot(
    metrics.episode_lengths,
    xlabel="Episode",
    ylabel="Steps",
    title="Training Progress - Episode Length",
    label="Steps",
    color=:teal,
    alpha=0.3
)

if length(metrics.episode_lengths) >= window
    moving_avg_steps = [mean(metrics.episode_lengths[max(1, i-window+1):i]) 
                        for i in 1:length(metrics.episode_lengths)]
    plot!(p4, moving_avg_steps, label="Moving Average", linewidth=2, color=:red)
end

# Combine plots
combined_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800))

# Save plot
savefig(combined_plot, "results/tiger_training_curves.png")
println("✓ Saved training curves to results/tiger_training_curves.png")

# Summary
println("\n" * "="^70)
println("Training Summary")
println("="^70)
println("Neural Policy Performance:")
println("  Mean reward: ", round(neural_results.mean_reward, digits=2))
println("  Std dev: ", round(neural_results.std_reward, digits=2))
println("  % of optimal: ", round(neural_pct, digits=1), "%")
println("  Inference time: ", round(mean_time, digits=3), " ms")
println()
println("Comparison to Baselines:")
println("  Neural: ", round(neural_results.mean_reward, digits=2))
println("  QMDP: ", round(results[2].mean_reward, digits=2))
println("  Random: ", round(results[3].mean_reward, digits=2))
println("  Optimal: ", round(optimal, digits=2))
println()
println("Training Metrics:")
println("  Final reward (last 10): ", round(mean(metrics.episode_rewards[end-9:end]), digits=2))
println("  Final loss (last 10): ", round(mean(metrics.losses[end-9:end]), digits=3))
println("  Final entropy (last 10): ", round(mean(metrics.entropies[end-9:end]), digits=3))
println("="^70)
println()

println("✓ Training complete! Check results/tiger_training_curves.png for plots.")
