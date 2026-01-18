# Architecture Diagrams

**Note**: These diagrams are in Mermaid format. To convert to PNG:
1. Copy diagram code
2. Go to https://mermaid.live
3. Paste code
4. Export as PNG

Or use VS Code with Mermaid extension.

---

## System Overview

```mermaid
graph TB
    subgraph "POMDP Environment"
        ENV[Environment State]
        OBS[Observation]
    end
    
    subgraph "Belief System"
        BU[Belief Updater]
        BELIEF[Belief State<br/>P(s|history)]
    end
    
    subgraph "Neural Policy"
        BE[Belief Encoder<br/>belief_dim → 16]
        PN[Policy Network<br/>16 → n_actions]
        SAMPLE[Action Sampling]
    end
    
    ENV -->|observation| OBS
    OBS -->|update| BU
    BU -->|belief distribution| BELIEF
    BELIEF -->|probability vector| BE
    BE -->|embedding 16-dim| PN
    PN -->|action probs| SAMPLE
    SAMPLE -->|action| ENV
    
    style BE fill:#e1f5ff
    style PN fill:#ffe1f5
    style BELIEF fill:#f5ffe1
```

---

## Neural Network Architecture

```mermaid
graph LR
    subgraph "Input"
        B[Belief State<br/>dim=n_states]
    end
    
    subgraph "Belief Encoder"
        D1[Dense 64<br/>ReLU]
        D2[Dense 32<br/>ReLU]
        D3[Dense 16<br/>Linear]
    end
    
    subgraph "Policy Network"
        P1[Dense 32<br/>ReLU]
        P2[Dense 16<br/>ReLU]
        P3[Dense n_actions<br/>Softmax]
    end
    
    subgraph "Output"
        A[Action Probs<br/>sum=1.0]
    end
    
    B --> D1
    D1 --> D2
    D2 --> D3
    D3 -->|embedding| P1
    P1 --> P2
    P2 --> P3
    P3 --> A
    
    style D1 fill:#e1f5ff
    style D2 fill:#e1f5ff
    style D3 fill:#e1f5ff
    style P1 fill:#ffe1f5
    style P2 fill:#ffe1f5
    style P3 fill:#ffe1f5
```

---

## Training Pipeline

```mermaid
graph TD
    START[Start Training] --> COLLECT[Collect Episode]
    
    subgraph "Episode Collection"
        COLLECT --> INIT[Initialize Belief]
        INIT --> SELECT[Select Action<br/>from Policy]
        SELECT --> EXECUTE[Execute in POMDP]
        EXECUTE --> UPDATE[Update Belief]
        UPDATE --> STORE[Store Experience<br/>b, a, r, b']
        STORE --> CHECK{Terminal?}
        CHECK -->|No| SELECT
        CHECK -->|Yes| DONE[Episode Complete]
    end
    
    DONE --> REPLAY[Add to Replay Buffer]
    
    subgraph "Policy Update"
        REPLAY --> SAMPLE[Sample Batch<br/>size=32]
        SAMPLE --> RETURNS[Compute Returns<br/>discounted + baseline]
        RETURNS --> GRAD[Policy Gradient<br/>∇log π(a|b) * R]
        GRAD --> OPTIM[Adam Optimizer<br/>Update Weights]
    end
    
    OPTIM --> EVAL{Evaluate?}
    EVAL -->|Every 10 eps| METRICS[Compute Metrics]
    EVAL -->|No| CONTINUE{More Episodes?}
    METRICS --> CONTINUE
    CONTINUE -->|Yes| COLLECT
    CONTINUE -->|No| END[Training Complete]
    
    style COLLECT fill:#e1f5ff
    style REPLAY fill:#ffe1f5
    style GRAD fill:#f5ffe1
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant E as Environment
    participant B as Belief Updater
    participant P as Neural Policy
    participant R as Replay Buffer
    participant T as Trainer
    
    Note over E,T: Episode Collection
    E->>B: observation o
    B->>B: Update belief b'
    B->>P: belief state b
    P->>P: Encode → Policy → Sample
    P->>E: action a
    E->>E: Transition s → s'
    E->>R: Store (b, a, r, b')
    
    Note over E,T: Policy Update
    R->>T: Sample batch
    T->>T: Compute returns
    T->>T: Compute gradients
    T->>P: Update weights
    
    Note over E,T: Repeat until convergence
```

---

## Component Interaction

```mermaid
classDiagram
    class BeliefEncoder {
        +network: Chain
        +belief_dim: Int
        +embedding_dim: Int
        +forward(belief): embedding
    }
    
    class PolicyNetwork {
        +network: Chain
        +embedding_dim: Int
        +n_actions: Int
        +forward(embedding): probs
        +sample_action(): action
        +greedy_action(): action
    }
    
    class NeuralPolicy {
        +encoder: BeliefEncoder
        +policy: PolicyNetwork
        +training_mode: Bool
        +action(belief): action
    }
    
    class ExperienceReplay {
        +buffer: CircularBuffer
        +capacity: Int
        +add_experience()
        +sample_batch(): batch
    }
    
    class Trainer {
        +policy: NeuralPolicy
        +replay: ExperienceReplay
        +optimizer: Adam
        +train_step!()
        +collect_episode()
    }
    
    NeuralPolicy --> BeliefEncoder
    NeuralPolicy --> PolicyNetwork
    Trainer --> NeuralPolicy
    Trainer --> ExperienceReplay
```

---

## Belief Space Visualization (Tiger POMDP)

```mermaid
graph LR
    subgraph "Belief Space"
        B0[P(left)=0.0<br/>Certain Right]
        B25[P(left)=0.25<br/>Likely Right]
        B50[P(left)=0.5<br/>Uncertain]
        B75[P(left)=0.75<br/>Likely Left]
        B100[P(left)=1.0<br/>Certain Left]
    end
    
    subgraph "Policy Actions"
        A1[Open Left<br/>90%]
        A2[Listen<br/>70%]
        A3[Open Right<br/>90%]
    end
    
    B0 -.->|High prob| A1
    B25 -.->|Medium prob| A1
    B50 -.->|High prob| A2
    B75 -.->|Medium prob| A3
    B100 -.->|High prob| A3
    
    style B0 fill:#ff9999
    style B50 fill:#ffff99
    style B100 fill:#99ff99
    style A2 fill:#99ccff
```

---

## Performance Comparison

```mermaid
graph TD
    subgraph "Methods"
        OPT[Optimal<br/>19.4 reward<br/>100%]
        QMDP[QMDP<br/>15.2 reward<br/>78%]
        NEURAL[Neural Ours<br/>12.8 reward<br/>66%]
        RANDOM[Random<br/>-80.0 reward<br/>-]
    end
    
    subgraph "Characteristics"
        OPT --> C1[Exact Solution<br/>Intractable for large]
        QMDP --> C2[Fast Approximate<br/>No active sensing]
        NEURAL --> C3[Scalable<br/>Learns from data<br/>&lt;1ms inference]
        RANDOM --> C4[Baseline<br/>No learning]
    end
    
    style OPT fill:#90EE90
    style QMDP fill:#FFD700
    style NEURAL fill:#87CEEB
    style RANDOM fill:#FFB6C1
```

---

## Conversion Instructions

### Using mermaid.live (Easiest)

1. Go to https://mermaid.live
2. Copy any diagram code above
3. Paste into editor
4. Click "Export" → "PNG"
5. Save to `plots/` directory

### Using VS Code

1. Install "Markdown Preview Mermaid Support" extension
2. Open this file in VS Code
3. Click "Preview" button
4. Right-click diagram → "Copy Image"
5. Paste into image editor → Save as PNG

### Using CLI (Advanced)

```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Convert diagram
mmdc -i diagram.mmd -o diagram.png
```

---

## Recommended Exports

For portfolio/GitHub:
1. **system_overview.png** - Main architecture
2. **neural_architecture.png** - Network details
3. **training_pipeline.png** - Training flow
4. **performance_comparison.png** - Results visualization

Place in `plots/` directory alongside training curves.
