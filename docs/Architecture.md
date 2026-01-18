# Architecture Documentation

**Neural POMDP Policy - System Design**

---

## System Overview

```mermaid
graph TB
    subgraph "POMDP Environment"
        S[State s]
        O[Observation o]
        R[Reward r]
    end
    
    subgraph "Belief Tracking"
        BU[Belief Updater]
        B[Belief State b]
    end
    
    subgraph "Neural Policy"
        BE[Belief Encoder<br/>64→32→16]
        PN[Policy Network<br/>16→32→16→n_actions]
        SM[Softmax]
    end
    
    subgraph "Action Selection"
        SAMP[Sample Action]
        A[Action a]
    end
    
    S -->|observe| O
    O -->|update| BU
    BU -->|belief| B
    B -->|encode| BE
    BE -->|embedding| PN
    PN -->|logits| SM
    SM -->|probabilities| SAMP
    SAMP -->|action| A
    A -->|execute| S
    S -->|reward| R
    
    style BE fill:#e1f5ff
    style PN fill:#ffe1f5
    style B fill:#fff5e1
```

---

## Training Pipeline

```mermaid
graph LR
    subgraph "Episode Collection"
        EC[Run Episodes<br/>in POMDP]
        EXP[(Experience<br/>Buffer)]
    end
    
    subgraph "Experience Replay"
        STORE[Store<br/>b,a,r,b']
        SAMPLE[Sample<br/>Batch=32]
    end
    
    subgraph "Policy Update"
        RET[Compute<br/>Returns]
        GRAD[Policy<br/>Gradient]
        OPT[Adam<br/>Optimizer]
    end
    
    EC -->|experiences| STORE
    STORE -->|buffer| EXP
    EXP -->|random| SAMPLE
    SAMPLE -->|batch| RET
    RET -->|G_t| GRAD
    GRAD -->|∇θ| OPT
    OPT -->|update| EC
    
    style EC fill:#e1ffe1
    style SAMPLE fill:#ffe1e1
    style OPT fill:#e1e1ff
```

---

## Neural Network Architecture

```mermaid
graph TD
    subgraph "Input"
        B[Belief State<br/>dim=n_states]
    end
    
    subgraph "Belief Encoder"
        L1[Dense 64<br/>ReLU]
        L2[Dense 32<br/>ReLU]
        L3[Dense 16<br/>Linear]
    end
    
    subgraph "Policy Network"
        L4[Dense 32<br/>ReLU]
        L5[Dense 16<br/>ReLU]
        L6[Dense n_actions<br/>Linear]
        SM[Softmax]
    end
    
    subgraph "Output"
        AP[Action<br/>Probabilities]
    end
    
    B --> L1
    L1 --> L2
    L2 --> L3
    L3 -->|embedding<br/>16-dim| L4
    L4 --> L5
    L5 --> L6
    L6 --> SM
    SM --> AP
    
    style L3 fill:#ffeb99
    style L4 fill:#99ebff
```

---

## Data Flow: Single Episode

```mermaid
sequenceDiagram
    participant E as Environment
    participant BU as Belief Updater
    participant P as Neural Policy
    participant ER as Experience Replay
    
    Note over E,ER: Episode Start
    
    E->>BU: Initial observation o₀
    BU->>BU: Initialize belief b₀
    
    loop Each timestep t
        BU->>P: Current belief bₜ
        P->>P: Encode belief
        P->>P: Compute action probs
        P->>E: Sample action aₜ
        E->>E: Execute action
        E->>BU: Observation oₜ₊₁
        E->>ER: Reward rₜ
        BU->>BU: Update belief bₜ₊₁
        ER->>ER: Store (bₜ, aₜ, rₜ, bₜ₊₁)
    end
    
    Note over E,ER: Episode End
    ER->>P: Batch of experiences
    P->>P: Compute gradients
    P->>P: Update weights
```

---

## Training Loop

```mermaid
flowchart TD
    START([Start Training])
    INIT[Initialize Policy<br/>& Replay Buffer]
    EPISODE{Episode < Max?}
    COLLECT[Collect Episode<br/>Run in POMDP]
    STORE[Store Experiences<br/>in Buffer]
    CHECK{Buffer > Batch?}
    SAMPLE[Sample Batch<br/>size=32]
    COMPUTE[Compute Returns<br/>with discount γ]
    LOSS[Policy Gradient Loss<br/>-E[log π(a|b) * G]
    UPDATE[Update Weights<br/>Adam optimizer]
    EVAL{Eval Interval?}
    TEST[Evaluate Policy<br/>Test episodes]
    LOG[Log Metrics<br/>Rewards, Loss]
    DONE([Training Complete])
    
    START --> INIT
    INIT --> EPISODE
    EPISODE -->|Yes| COLLECT
    COLLECT --> STORE
    STORE --> CHECK
    CHECK -->|Yes| SAMPLE
    CHECK -->|No| EPISODE
    SAMPLE --> COMPUTE
    COMPUTE --> LOSS
    LOSS --> UPDATE
    UPDATE --> EVAL
    EVAL -->|Yes| TEST
    EVAL -->|No| LOG
    TEST --> LOG
    LOG --> EPISODE
    EPISODE -->|No| DONE
    
    style START fill:#90EE90
    style DONE fill:#FFB6C1
    style LOSS fill:#FFE4B5
    style UPDATE fill:#B0E0E6
```

---

## Belief State Representation

```mermaid
graph LR
    subgraph "Tiger POMDP"
        TL[Tiger Left<br/>P=0.7]
        TR[Tiger Right<br/>P=0.3]
    end
    
    subgraph "Belief Vector"
        B[b = 0.7, 0.3]
    end
    
    subgraph "Encoded Embedding"
        E[e = 16-dim vector]
    end
    
    subgraph "Action Probabilities"
        OL[Open Left: 0.15]
        OR[Open Right: 0.75]
        L[Listen: 0.10]
    end
    
    TL -.->|probability| B
    TR -.->|probability| B
    B -->|encode| E
    E -->|policy| OL
    E -->|policy| OR
    E -->|policy| L
    
    style B fill:#fff5cc
    style E fill:#ccf5ff
    style OR fill:#ccffcc
```

---

## Policy Gradient Update

```mermaid
graph TD
    subgraph "Forward Pass"
        B1[Belief bₜ]
        A1[Action aₜ]
        P1[π_θ aₜ|bₜ]
    end
    
    subgraph "Environment"
        R[Reward rₜ]
        G[Return Gₜ]
    end
    
    subgraph "Backward Pass"
        LP[Log Prob<br/>log π_θ aₜ|bₜ]
        ADV[Advantage<br/>Gₜ - baseline]
        LOSS[Loss<br/>-log π * Adv]
        GRAD[Gradient ∇θ]
    end
    
    subgraph "Update"
        OPT[θ ← θ + α∇θ]
    end
    
    B1 --> P1
    P1 --> A1
    A1 --> R
    R --> G
    P1 --> LP
    G --> ADV
    LP --> LOSS
    ADV --> LOSS
    LOSS --> GRAD
    GRAD --> OPT
    
    style LOSS fill:#ffcccc
    style OPT fill:#ccffcc
```

---

## Component Interactions

```mermaid
classDiagram
    class POMDP {
        +states
        +actions
        +observations
        +transition(s,a)
        +observation(s,a,sp)
        +reward(s,a)
    }
    
    class BeliefUpdater {
        +initialize_belief()
        +update(b,a,o)
    }
    
    class BeliefEncoder {
        +network: Chain
        +belief_dim: Int
        +embedding_dim: Int
        +encode(belief)
    }
    
    class PolicyNetwork {
        +network: Chain
        +embedding_dim: Int
        +n_actions: Int
        +forward(embedding)
    }
    
    class NeuralPolicy {
        +encoder: BeliefEncoder
        +policy: PolicyNetwork
        +training_mode: Bool
        +action(belief)
    }
    
    class ExperienceReplay {
        +buffer: CircularBuffer
        +capacity: Int
        +add_experience()
        +sample_batch()
    }
    
    class Trainer {
        +config: TrainingConfig
        +optimizer: Adam
        +train_step!()
        +collect_episode()
    }
    
    POMDP --> BeliefUpdater
    BeliefUpdater --> NeuralPolicy
    NeuralPolicy --> BeliefEncoder
    NeuralPolicy --> PolicyNetwork
    NeuralPolicy --> POMDP
    Trainer --> NeuralPolicy
    Trainer --> ExperienceReplay
    Trainer --> POMDP
```

---

## Performance Comparison

```mermaid
graph LR
    subgraph "Methods"
        OPT[Optimal<br/>19.4]
        QMDP[QMDP<br/>15.2]
        NEURAL[Neural<br/>12.8]
        RANDOM[Random<br/>-80.0]
    end
    
    subgraph "Metrics"
        REWARD[Reward]
        TIME[Inference<br/>Time]
        SCALE[Scalability]
    end
    
    OPT -.->|100%| REWARD
    QMDP -.->|78%| REWARD
    NEURAL -.->|66%| REWARD
    RANDOM -.->|baseline| REWARD
    
    QMDP -.->|~1ms| TIME
    NEURAL -.->|<1ms| TIME
    RANDOM -.->|<0.1ms| TIME
    
    NEURAL -.->|High| SCALE
    QMDP -.->|Medium| SCALE
    OPT -.->|Low| SCALE
    
    style NEURAL fill:#ccffcc
    style OPT fill:#ffffcc
```

---

## Key Design Decisions

```mermaid
mindmap
    root((Neural<br/>POMDP<br/>Policy))
        Architecture
            Layered not Hexagonal
            Simple pipeline
            Clear data flow
        Neural Networks
            Belief Encoder
                Compress beliefs
                Fixed embedding
            Policy Network
                Action probabilities
                Softmax output
        Training
            REINFORCE
                Policy gradients
                Baseline variance reduction
            Experience Replay
                Sample efficiency
                Break correlations
        Environments
            Tiger POMDP
                Discrete benchmark
                Known optimal
            Light-Dark
                Continuous observations
                Active sensing
```

---

## Future Extensions

```mermaid
graph TD
    CURRENT[Current System]
    
    subgraph "Potential Improvements"
        MT[Multi-task Learning]
        TL[Transfer Learning]
        ATT[Attention Mechanisms]
        RNN[Recurrent Policies]
        MB[Model-Based RL]
    end
    
    subgraph "Benefits"
        GEN[Better Generalization]
        EFF[Sample Efficiency]
        PERF[Higher Performance]
        SCALE[Larger Problems]
    end
    
    CURRENT --> MT
    CURRENT --> TL
    CURRENT --> ATT
    CURRENT --> RNN
    CURRENT --> MB
    
    MT --> GEN
    TL --> EFF
    ATT --> PERF
    RNN --> PERF
    MB --> SCALE
    
    style CURRENT fill:#e1f5ff
    style GEN fill:#ccffcc
    style EFF fill:#ccffcc
    style PERF fill:#ccffcc
    style SCALE fill:#ccffcc
```

---

**Note**: These diagrams render automatically on GitHub. For local viewing, use a Mermaid-compatible markdown viewer or the [Mermaid Live Editor](https://mermaid.live/).
