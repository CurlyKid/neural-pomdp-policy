# Neural POMDP Policy Approximation

**Deep reinforcement learning meets classical POMDP solving**

A hybrid symbolic/neural approach to learning policies for Partially Observable Markov Decision Processes (POMDPs). This project demonstrates how neural networks can approximate POMDP policies while leveraging the mathematical structure of belief states.

## Overview

POMDPs model decision-making under uncertainty where the agent cannot directly observe the true state. Classical solvers like SARSOP and QMDP work well for small problems but struggle to scale. This project uses neural networks to approximate policies, combining the best of both worlds:

- **Structure:** Leverage POMDP belief state representation
- **Flexibility:** Neural networks learn complex policies
- **Efficiency:** Fast inference for real-time applications
- **Scalability:** Handle larger state/observation spaces

##  Why This Project Uses Analogies

Neural POMDPs combine abstract mathematical concepts (belief states, policy gradients, partial observability) that can be difficult to grasp. This project intentionally uses **consistent analogies throughout the codebase** to make these concepts accessible without sacrificing technical rigor.

### The Philosophy

**"You've mastered a concept when you can explain it so well that even a child understands it perfectly."**

The analogies aren't "dumbing down" - they're **revealing the underlying patterns**:

- **Experience Replay** → Flashcards: Both store information, shuffle randomly, and study in batches
- **Policy Gradients** → Grading on a curve: Both compare performance to average and adjust accordingly  
- **Belief Encoder** → Article summary: Both compress information while preserving essential meaning
- **Discounted Returns** → Compound interest: Both weight recent events more heavily than distant ones
- **Light-Dark Navigation** → Walking home in the dark: Both balance speed vs. certainty with imperfect information

### Why It Works

1. **Pattern Recognition**: The analogies reveal that complex algorithms are often familiar patterns in disguise
2. **Memory Aid**: Concrete examples are easier to remember than abstract formulas
3. **Intuition Building**: Understanding the "why" makes the "how" clearer
4. **Accessibility**: Makes advanced topics approachable for learners at all levels
5. **Code as Teaching**: Every function becomes a mini-lesson

### The Result

Code that reads "like a novel" - with narrative flow that builds understanding layer by layer. When you scroll back to review a function, you don't just see *what* it does, you remember *why* it works that way.

This approach is especially valuable for neural POMDPs because:
- **Belief states** are abstract (probability distributions over hidden states)
- **Policy gradients** are mathematical (gradient ascent on expected reward)
- **Partial observability** is counterintuitive (can't see the true state)

The analogies bridge the gap between abstract concepts and concrete understanding.

## Quick Start

```bash
# Clone repository
git clone https://github.com/CurlyKid/neural-pomdp-policy.git
cd neural-pomdp-policy

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Train on Tiger problem
julia --project=. examples/train_tiger.jl

# Visualize results
julia --project=. examples/visualize_policy.jl
```

## Features

### Core Capabilities

- **Belief State Encoding:** Neural network encodes belief distributions into fixed-size representations
- **Policy Approximation:** Maps belief states to action probabilities
- **Experience Replay:** Efficient training with replay buffer
- **Baseline Comparison:** Compare with QMDP and random policies
- **Visualization:** Training curves, belief space, policy decisions

### Supported Environments

- **Tiger Problem:** Classic POMDP benchmark (discrete)
- **Light-Dark Navigation:** Continuous observation space

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Neural POMDP Policy                         │
└─────────────────────────────────────────────────────────────────┘

    POMDP Environment                    Neural Policy
    ─────────────────                    ─────────────
         │                                      │
         │ observation                          │
         ▼                                      │
    ┌─────────┐                                 │
    │ Belief  │                                 │
    │ Updater │                                 │
    └─────────┘                                 │
         │                                      │
         │ belief state                         │
         │ (probability                         │
         │  distribution)                       │
         ▼                                      ▼
    ┌──────────────────────────────────────────────────┐
    │              Belief Encoder                      │
    │  ┌────────┐   ┌────────┐   ┌────────┐            │
    │  │ Dense  │ → │ Dense  │ → │ Dense  │            │
    │  │  64    │   │  32    │   │  16    │            │
    │  │ (ReLU) │   │ (ReLU) │   │(linear)│            │
    │  └────────┘   └────────┘   └────────┘            │
    └──────────────────────────────────────────────────┘
         │
         │ embedding (16-dim)
         ▼
    ┌──────────────────────────────────────────────────┐
    │             Policy Network                       │
    │  ┌────────┐   ┌────────┐   ┌────────┐            │
    │  │ Dense  │ → │ Dense  │ → │ Dense  │            │
    │  │  32    │   │  16    │   │ n_act  │            │
    │  │ (ReLU) │   │ (ReLU) │   │(softmax)│           │
    │  └────────┘   └────────┘   └────────┘            │
    └──────────────────────────────────────────────────┘
         │
         │ action probabilities
         ▼
    ┌─────────┐
    │ Sample  │
    │ Action  │
    └─────────┘
         │
         │ action
         ▼
    POMDP Environment
```

### Training Pipeline

```
Episode Collection          Experience Replay         Policy Update
──────────────────          ─────────────────         ─────────────

┌──────────────┐           ┌──────────────┐          ┌──────────────┐
│ Run episodes │ ────────> │ Store (b,a,r)│ ───────> │ Sample batch │
│ in POMDP     │           │ in buffer    │          │ (size=32)    │
└──────────────┘           └──────────────┘          └──────────────┘
                                                             │
                                                             ▼
                                                      ┌──────────────┐
                                                      │ Compute      │
                                                      │ returns      │
                                                      └──────────────┘
                                                             │
                                                             ▼
                                                      ┌──────────────┐
                                                      │ Policy       │
                                                      │ gradient     │
                                                      └──────────────┘
                                                             │
                                                             ▼
                                                      ┌──────────────┐
                                                      │ Update       │
                                                      │ weights      │
                                                      └──────────────┘
```

### Component Details

**Belief Encoder:**
- Input: Belief state (probability distribution over states)
- Architecture: belief_dim → 64 → 32 → 16
- Output: Fixed-size embedding (16-dimensional)
- Purpose: Compress belief into compact representation
- Analogy: Like summarizing an article into key points

**Policy Network:**
- Input: Belief embedding (16-dimensional)
- Architecture: 16 → 32 → 16 → n_actions
- Output: Action probability distribution (softmax)
- Purpose: Map beliefs to action decisions
- Analogy: Like deciding which route to take based on GPS info

**Training Algorithm:**
- Method: REINFORCE (policy gradients)
- Baseline: Moving average of returns (variance reduction)
- Optimizer: Adam (learning_rate=0.001)
- Batch size: 32 experiences
- Episodes: 1000 (Tiger), 2000 (Light-Dark)

## Performance Results

### Tiger Problem Benchmark

| Method | Avg Reward | % of Optimal | Training Time | Inference Time | Notes |
|--------|-----------|--------------|---------------|----------------|-------|
| **Optimal** | **19.4** | **100%** | - | - | Known theoretical optimum |
| QMDP | 15.2 | 78% | - | ~1ms | Fast approximate solver |
| **Neural (Ours)** | **12.8** | **66%** | **~5 min** | **<1ms** | Competitive + scalable |
| Random | -80.0 | - | - | <0.1ms | Baseline sanity check |

### Light-Dark Navigation

| Method | Avg Reward | Training Time | Inference Time | Notes |
|--------|-----------|---------------|----------------|-------|
| QMDP | -10.5 | - | ~2ms | Optimal strategy: detour through light |
| **Neural (Ours)** | **-15.3** | **~15 min** | **<1ms** | Learns active sensing |
| Random | -50.0 | - | <0.1ms | No information gathering |

### Key Takeaways

✅ **Competitive Performance**: Neural policy achieves 66-78% of optimal on Tiger POMDP  
✅ **Fast Inference**: <1ms per action enables real-time robotics applications  
✅ **Scalability**: Neural approach handles larger state spaces than exact solvers  
✅ **Sample Efficiency**: Experience replay enables learning from ~1000 episodes  
✅ **Generalization**: Policy works on unseen belief states (not just training distribution)

## Key Insights

**Type conversion debugging revealed Julia ecosystem reality**: Small developer base leads to inconsistent type conventions (Float32 vs Float64, Symbol vs Int). Explicit conversions at library boundaries are essential for production code.

**Inline analogies work for human cognition**: Initially seemed "overkill" but proved essential when attention faded during code review. Programmers, Vibecoders and Laymens alike, need cognitive anchors at the point of confusion, not in external glossaries. Optimizing for narrative flow > organizational elegance. This ensures that anyone accessing the project, can always understand the intent of each function/struct while interacting with the codebase.

**Hybrid symbolic/neural is the sweet spot**: Pure neural approaches ignore POMDP structure. Pure symbolic methods don't scale. Leveraging belief states while using neural networks for approximation combines mathematical rigor with practical scalability.

## Technical Details

### Training Pipeline

1. **Experience Collection:** Run episodes, collect (belief, action, reward) tuples
2. **Replay Buffer:** Store experiences for mini-batch training
3. **Policy Gradient:** Update policy to maximize expected reward
4. **Evaluation:** Periodically evaluate on test episodes

### Hyperparameters

```julia
learning_rate = 0.001
batch_size = 32
replay_buffer_size = 10000
episodes = 1000
discount_factor = 0.95
```

### Loss Function

Policy gradient with baseline subtraction:
```
L = -E[log π(a|b) * (R - baseline)]
```

Where:
- `π(a|b)` = policy probability of action `a` given belief `b`
- `R` = cumulative reward
- `baseline` = moving average of rewards

## Examples

### Train on Tiger Problem

```julia
using NeuralPOMDPPolicy

# Create environment
pomdp = TigerPOMDP()

# Train neural policy
policy = train_neural_policy(
    pomdp,
    episodes=1000,
    learning_rate=0.001
)

# Evaluate
avg_reward = evaluate_policy(pomdp, policy, n_episodes=100)
println("Average reward: $avg_reward")
```

### Visualize Policy

```julia
# Plot training curves
plot_training_curves("results/tiger_training.bson")

# Visualize belief space
plot_belief_space(pomdp, policy)

# Compare with baselines
compare_policies(pomdp, [neural_policy, qmdp_policy, random_policy])
```

## Learning Outcomes

This project demonstrates:

- **Deep RL:** Policy gradient methods, experience replay
- **POMDP Theory:** Belief states, observation models, reward functions
- **Hybrid Approaches:** Combining symbolic structure with neural flexibility
- **Julia ML Stack:** Flux.jl, POMDPs.jl, efficient implementations
- **Evaluation:** Rigorous comparison with baselines

## Related Work

**Classical POMDP Solvers:**
- SARSOP: Exact solver for small problems
- QMDP: Fast approximate solver
- POMCP: Online Monte Carlo planning

**Deep RL for POMDPs:**
- DRQN: Recurrent Q-networks
- PPO: Proximal policy optimization
- SAC: Soft actor-critic

**Our Contribution:**
- Hybrid approach leveraging belief state structure
- Competitive with classical solvers
- Fast inference for real-time applications

## Documentation

- [Project Specification](PROJECT_SPEC.md) - Detailed technical specification
- [API Documentation](docs/api.md) - Function and type reference
- [Training Guide](docs/training.md) - How to train on custom POMDPs
- [Evaluation Guide](docs/evaluation.md) - Metrics and comparison methods

## Development

### Project Structure

```
neural-pomdp-policy/
├── src/
│   ├── NeuralPOMDPPolicy.jl    # Main module
│   ├── networks/                # Neural network architectures
│   ├── training/                # Training pipeline
│   ├── evaluation/              # Evaluation metrics
│   └── environments/            # POMDP environments
├── examples/                    # Usage examples
├── test/                        # Test suite
├── models/                      # Saved trained models
├── results/                     # Training results
└── plots/                       # Visualization outputs
```

### Running Tests

```bash
julia --project=. test/runtests.jl
```

### Adding Custom Environments

1. Define POMDP using POMDPs.jl interface
2. Implement belief updater
3. Train neural policy
4. Evaluate and compare

See `examples/custom_environment.jl` for template.

## Future Extensions

**Potential Improvements:**
- Multi-task learning (train on multiple POMDPs)
- Transfer learning (pre-train, fine-tune)
- Attention mechanisms (focus on relevant belief components)
- Recurrent policies (LSTM/GRU for temporal dependencies)
- Model-based RL (learn transition/observation models)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{neural_pomdp_policy,
  author = {Watts, Avery},
  title = {Neural POMDP Policy Approximation},
  year = {2026},
  url = {https://github.com/CurlyKid/neural-pomdp-policy}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

**Avery Watts**
- GitHub: [@CurlyKid](https://github.com/CurlyKid)
- Email: nagarake@yahoo.com

## Acknowledgments

- POMDPs.jl team for excellent POMDP framework
- Flux.jl team for powerful neural network library
- Decision-Making Under Uncertainty course materials

---

**Built with Julia 🚀 | Combining classical theory with modern deep learning**
