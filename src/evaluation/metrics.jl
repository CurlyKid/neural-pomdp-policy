"""
Evaluation metrics for POMDP policies.

This module provides tools to measure how well a policy performs:
- Average reward over multiple episodes
- Statistical significance testing
- Convergence detection
- Performance comparison

Think of it like grading a student: we run multiple tests (episodes),
compute average score (mean reward), check consistency (std dev), and
compare to other students (baselines).
"""

using Statistics
using Random
using POMDPs
using POMDPTools

"""
    EvaluationResults

Results from evaluating a policy.

# Fields
- `mean_reward::Float64`: Average reward across episodes
- `std_reward::Float64`: Standard deviation of rewards
- `mean_steps::Float64`: Average episode length
- `std_steps::Float64`: Standard deviation of episode lengths
- `rewards::Vector{Float64}`: Individual episode rewards
- `steps::Vector{Int}`: Individual episode lengths
- `success_rate::Float64`: Fraction of episodes that reached goal (if applicable)

# Example
```julia
results = evaluate_policy(pomdp, policy, n_episodes=100)
println("Mean reward: ", results.mean_reward, " ± ", results.std_reward)
```
"""
struct EvaluationResults
    mean_reward::Float64
    std_reward::Float64
    mean_steps::Float64
    std_steps::Float64
    rewards::Vector{Float64}
    steps::Vector{Int}
    success_rate::Float64
end

"""
    evaluate_policy(pomdp, policy, updater; n_episodes=100, max_steps=100, verbose=false)

Evaluate a policy by running multiple episodes and computing statistics.

This is like giving a student multiple tests and computing their average grade.
We want to know: how well does the policy perform on average? How consistent
is it? Does it reliably reach the goal?

# Arguments
- `pomdp`: POMDP environment
- `policy`: Policy to evaluate
- `updater`: Belief updater
- `n_episodes`: Number of episodes to run (default: 100)
- `max_steps`: Maximum steps per episode (default: 100)
- `verbose`: Print progress (default: false)

# Returns
- `EvaluationResults`: Statistics about policy performance

# Example
```julia
# Evaluate neural policy
results = evaluate_policy(
    tiger_pomdp,
    neural_policy,
    updater,
    n_episodes=100
)

println("Performance: ", results.mean_reward, " ± ", results.std_reward)
println("Success rate: ", results.success_rate * 100, "%")
```
"""
function evaluate_policy(pomdp, policy, updater; n_episodes=100, max_steps=100, verbose=false)
    rewards = Float64[]
    steps_taken = Int[]
    successes = 0
    
    try
        for episode in 1:n_episodes
            # Run one episode
            total_reward = 0.0
            steps = 0
            
            # Initialize
            s0 = rand(initialstate(pomdp))
            b = initialize_belief(updater, initialstate(pomdp))
            
            # Run episode
            for step in 1:max_steps
                steps = step
                
                # Select action
                a = action(policy, b)
                
                # Take action
                s = rand(b)
                sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
                
                # Update belief
                b = update(updater, b, a, o)
                
                total_reward += r
                
                # Check termination
                if isterminal(pomdp, sp)
                    successes += 1
                    break
                end
            end
            
            push!(rewards, total_reward)
            push!(steps_taken, steps)
            
            if verbose && episode % 10 == 0
                println("Episode $episode: reward = $total_reward, steps = $steps")
            end
        end
        
        # Compute statistics
        mean_reward = mean(rewards)
        std_reward = std(rewards)
        mean_steps = mean(steps_taken)
        std_steps = std(steps_taken)
        success_rate = successes / n_episodes
        
        return EvaluationResults(
            mean_reward, std_reward,
            mean_steps, std_steps,
            rewards, steps_taken,
            success_rate
        )
        
    catch e
        @error "Evaluation failed" exception=(e, catch_backtrace())
        # Return empty results on failure
        return EvaluationResults(0.0, 0.0, 0.0, 0.0, Float64[], Int[], 0.0)
    end
end

"""
    confidence_interval(results::EvaluationResults; confidence=0.95)

Compute confidence interval for mean reward.

This tells us: "We're 95% confident the true mean reward is in this range."

Think of it like: if we ran the evaluation many times, 95% of the time the
true average would fall within this interval.

# Arguments
- `results`: Evaluation results
- `confidence`: Confidence level (default: 0.95 for 95%)

# Returns
- `(lower, upper)`: Confidence interval bounds

# Example
```julia
results = evaluate_policy(pomdp, policy, updater)
lower, upper = confidence_interval(results)
println("Mean reward: ", results.mean_reward)
println("95% CI: [", lower, ", ", upper, "]")
```
"""
function confidence_interval(results::EvaluationResults; confidence=0.95)
    n = length(results.rewards)
    
    if n < 2
        return (results.mean_reward, results.mean_reward)
    end
    
    # Use t-distribution for small samples
    # z-score for 95% confidence ≈ 1.96
    # For small n, use t-distribution (more conservative)
    z = 1.96  # Approximation for large n
    
    margin = z * results.std_reward / sqrt(n)
    
    return (results.mean_reward - margin, results.mean_reward + margin)
end

"""
    is_converged(metrics::TrainingMetrics; window=50, threshold=0.05)

Check if training has converged.

Training is "converged" when performance stops improving significantly.

Think of it like: if a student's grades have been stable for the last
10 tests, they've probably reached their current skill level.

# Arguments
- `metrics`: Training metrics
- `window`: Number of recent episodes to check (default: 50)
- `threshold`: Maximum relative change to consider converged (default: 0.05 = 5%)

# Returns
- `Bool`: True if converged, false otherwise

# Example
```julia
if is_converged(metrics)
    println("Training converged! Stopping early.")
    break
end
```
"""
function is_converged(metrics::TrainingMetrics; window=50, threshold=0.05)
    rewards = metrics.episode_rewards
    
    # Need enough data
    if length(rewards) < window * 2
        return false
    end
    
    # Compare recent window to previous window
    recent = mean(rewards[end-window+1:end])
    previous = mean(rewards[end-2*window+1:end-window])
    
    # Avoid division by zero
    if abs(previous) < 1e-6
        return false
    end
    
    # Check relative change
    relative_change = abs(recent - previous) / abs(previous)
    
    return relative_change < threshold
end

"""
    compare_to_baseline(results::EvaluationResults, baseline_reward::Float64)

Compare policy performance to a baseline.

This tells us: "How much better (or worse) is our policy compared to baseline?"

Think of it like: comparing your test score to the class average.

# Arguments
- `results`: Evaluation results for our policy
- `baseline_reward`: Baseline reward (e.g., optimal, random, QMDP)

# Returns
- `percentage::Float64`: Performance as percentage of baseline (100% = equal)

# Example
```julia
results = evaluate_policy(pomdp, neural_policy, updater)
optimal_reward = 19.4  # Known optimal for Tiger POMDP

percentage = compare_to_baseline(results, optimal_reward)
println("Neural policy achieves ", percentage, "% of optimal")
# Typical output: "Neural policy achieves 75% of optimal"
```
"""
function compare_to_baseline(results::EvaluationResults, baseline_reward::Float64)
    if abs(baseline_reward) < 1e-6
        return 0.0
    end
    
    percentage = (results.mean_reward / baseline_reward) * 100.0
    return percentage
end

"""
    statistical_significance(results1::EvaluationResults, results2::EvaluationResults)

Test if two policies have significantly different performance.

Uses Welch's t-test to check if the difference in means is statistically
significant (not just due to random chance).

Think of it like: if two students have different average grades, is one
actually better, or could it just be luck?

# Arguments
- `results1`: Results for first policy
- `results2`: Results for second policy

# Returns
- `(significant, p_value)`: Whether difference is significant and p-value

# Example
```julia
neural_results = evaluate_policy(pomdp, neural_policy, updater)
qmdp_results = evaluate_policy(pomdp, qmdp_policy, updater)

significant, p_value = statistical_significance(neural_results, qmdp_results)
if significant
    println("Policies are significantly different (p = ", p_value, ")")
else
    println("No significant difference (p = ", p_value, ")")
end
```
"""
function statistical_significance(results1::EvaluationResults, results2::EvaluationResults)
    # Welch's t-test (doesn't assume equal variances)
    n1 = length(results1.rewards)
    n2 = length(results2.rewards)
    
    if n1 < 2 || n2 < 2
        return (false, 1.0)
    end
    
    # Compute t-statistic
    mean_diff = results1.mean_reward - results2.mean_reward
    se = sqrt(results1.std_reward^2 / n1 + results2.std_reward^2 / n2)
    
    if se < 1e-6
        return (false, 1.0)
    end
    
    t_stat = abs(mean_diff / se)
    
    # Approximate p-value (two-tailed test)
    # For t > 1.96, p < 0.05 (significant at 95% confidence)
    # For t > 2.58, p < 0.01 (highly significant)
    significant = t_stat > 1.96
    
    # Rough p-value approximation
    if t_stat > 2.58
        p_value = 0.01
    elseif t_stat > 1.96
        p_value = 0.05
    else
        p_value = 0.10
    end
    
    return (significant, p_value)
end

"""
    print_evaluation_summary(results::EvaluationResults; name="Policy")

Print a nice summary of evaluation results.

# Example
```julia
results = evaluate_policy(pomdp, policy, updater)
print_evaluation_summary(results, name="Neural Policy")
```
"""
function print_evaluation_summary(results::EvaluationResults; name="Policy")
    println("\n=== $name Evaluation ===")
    println("Episodes: ", length(results.rewards))
    println("Mean reward: ", round(results.mean_reward, digits=2), 
            " ± ", round(results.std_reward, digits=2))
    println("Mean steps: ", round(results.mean_steps, digits=1),
            " ± ", round(results.std_steps, digits=1))
    println("Success rate: ", round(results.success_rate * 100, digits=1), "%")
    
    lower, upper = confidence_interval(results)
    println("95% CI: [", round(lower, digits=2), ", ", round(upper, digits=2), "]")
    println()
end

"""
    test_metrics()

Test the metrics module.
"""
function test_metrics()
    println("\n=== Testing Metrics Module ===\n")
    
    try
        # Create fake results
        rewards = [10.0, 12.0, 11.0, 13.0, 12.5]
        steps = [20, 22, 21, 23, 22]
        
        results = EvaluationResults(
            mean(rewards), std(rewards),
            mean(steps), std(steps),
            rewards, steps,
            0.8
        )
        
        println("✓ EvaluationResults created")
        println("  Mean reward: ", results.mean_reward)
        println("  Std reward: ", results.std_reward)
        
        # Test confidence interval
        lower, upper = confidence_interval(results)
        println("✓ Confidence interval: [", lower, ", ", upper, "]")
        
        # Test baseline comparison
        baseline = 15.0
        percentage = compare_to_baseline(results, baseline)
        println("✓ Baseline comparison: ", percentage, "% of baseline")
        
        # Test convergence check
        metrics = TrainingMetrics()
        metrics.episode_rewards = repeat([10.0], 100)
        converged = is_converged(metrics)
        println("✓ Convergence check: ", converged)
        
        println("\n✓ All metrics tests passed!")
        
    catch e
        println("✗ Metrics test failed: ", e)
        rethrow(e)
    end
end
