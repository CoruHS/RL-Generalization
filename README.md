# RL Generalization Research

**What Makes RL Generalize? A Systematic Study with Gradient Agreement Regularization**

This research project studies how reinforcement learning algorithms generalize under environmental distribution shift.

## Key Contributions

1. **GRS (Generalization Robustness Score)** - A metric measuring normalized area under performance curve during environmental shift
2. **GAR (Gradient Agreement Regularization)** - Novel training technique filtering gradient updates by agreement across environment variations
3. **GAS (Gradient Agreement Score)** - Diagnostic tool predicting generalization from training dynamics
4. **Systematic experimental analysis** comparing three algorithm families across controlled environmental variations

## Project Structure

```
├── agents/           # RL algorithm implementations
│   ├── dqn.py       # Deep Q-Network with GAR support
│   ├── ppo.py       # Proximal Policy Optimization with GAR
│   └── es.py        # Evolution Strategies
├── core/            # Neural networks and utilities
│   ├── networks.py  # MLP, CNN architectures
│   ├── gar.py       # Gradient Agreement Regularization
│   ├── replay_buffer.py
│   └── rollout_buffer.py
├── envs/            # Parametric environments
│   ├── cartpole_var.py   # CartPole with dynamics variation
│   ├── minigrid_var.py   # MiniGrid with layout variation
│   └── starpilot_var.py  # Procgen Starpilot
├── evaluation/      # Metrics and analysis
│   ├── grs.py       # GRS computation and plotting
│   └── metrics.py   # Training metrics tracking
├── techniques/      # Gradient analysis utilities
│   ├── gar.py       # GAR implementation
│   └── gas.py       # Gradient Agreement Score
├── paper/           # LaTeX paper and figures
├── run_experiment.py    # Main experiment runner
├── play.py              # Watch trained agents
└── analyze_results.py   # Results analysis
```

## Experimental Setup

**108 experiments:** 3 algorithms × 4 techniques × 3 environments × 3 seeds

| Dimension | Options |
|-----------|---------|
| **Algorithms** | DQN, PPO, ES |
| **Techniques** | baseline, reg (L2), gar, gar+reg |
| **Environments** | CartPole, MiniGrid, Starpilot |

## Usage

```bash
# Install dependencies
pip install torch numpy gymnasium minigrid

# Run single environment experiments
python run_experiment.py --env-only cartpole

# Run specific experiment
python run_experiment.py --algo ppo --technique gar --env cartpole --seeds 0 1 2

# Watch trained agent
python play.py --env cartpole --algo dqn

# Analyze results
python run_experiment.py --analyze
```

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- Gymnasium
- MiniGrid
- Matplotlib
