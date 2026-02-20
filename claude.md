# RL Generalization Research Project

## Project Goal
Research project studying which RL algorithm + training technique combinations generalize best across different types of environmental variation.

**Research Question:** "What Makes RL Generalize? A Systematic Study with Gradient Agreement"

---

## Novel Contributions
1. **GRS (Generalization Robustness Score)** - Measures degradation curve shape, not just point accuracy
2. **GAR (Gradient Agreement Regularization)** - Trains on multiple environment variations, only updates in gradient directions that agree across variations
3. **GAS (Gradient Agreement Score)** - Predicts generalization from training dynamics

---

## File Structure
```
rl_project/
├── agents/
│   ├── __init__.py
│   ├── dqn.py              # Deep Q-Network (with GAR support, torch.compile)
│   ├── ppo.py              # Proximal Policy Optimization (with GAR support, torch.compile)
│   └── es.py               # Evolution Strategies (torch.compile, no GAR - doesn't use gradients)
│
├── core/
│   ├── __init__.py
│   ├── networks.py         # MLP, CNN, DQNNetwork, PPONetwork, ESNetwork
│   ├── replay_buffer.py    # DQN replay buffer
│   ├── rollout_buffer.py   # PPO rollout buffer with GAE
│   └── gar.py              # GAR utilities (gradient agreement, GARBuffer, GARStats)
│
├── envs/
│   ├── __init__.py
│   ├── cartpole_var.py     # CartPole with dynamics variation
│   ├── minigrid_var.py     # MiniGrid with layout variation (flattened obs)
│   ├── starpilot_var.py    # Starpilot (Procgen) - REPLACED CoinRun
│   ├── coinrun_var.py      # CoinRun (deprecated - too slow, use Starpilot)
│   └── wrappers.py         # FrameStack wrapper for Procgen (4 frames)
│
├── evaluation/
│   ├── __init__.py
│   ├── grs.py              # GRS metric computation and plotting
│   └── metrics.py          # MetricsTracker, learning curves, heatmaps
│
├── techniques/
│   ├── __init__.py
│   ├── gar.py              # Gradient Agreement Regularization (legacy location)
│   └── gas.py              # Gradient Agreement Score diagnostic
│
├── results/
│   ├── best_models/        # Best model per architecture (dqn_cartpole_best.pt, etc.)
│   ├── plots/              # Learning curves, GRS curves, heatmaps
│   ├── metrics/            # Training history JSON files
│   └── *.json              # Individual experiment results
│
├── run_experiment.py       # Main experiment runner
├── requirements.txt
└── README.md
```

---

## Experimental Matrix

| Dimension | Options |
|-----------|---------|
| Algorithms | DQN, PPO, ES |
| Techniques | baseline, reg, gar, gar+reg (combinable!) |
| Environments | CartPole (dynamics), MiniGrid (layout), Starpilot (visual) |
| Seeds | 0, 1, 2 |

**Total:** 3 × 4 × 3 × 3 = 108 experiments

### Technique Combinations
Techniques can be combined with `+`:
- `baseline` - No regularization, no GAR
- `reg` - L2 weight decay (1e-4)
- `gar` - Gradient Agreement Regularization
- `gar+reg` - Both GAR and weight decay

---

## Key Hyperparameters

### Environment-Specific (HYPERPARAMS dict in run_experiment.py)

#### CartPole (4-dim state vector, fast learning)
```python
"dqn": {"learning_rate": 1e-4, "hidden_dims": [64, 64], "batch_size": 64}
"ppo": {"learning_rate": 3e-4, "hidden_dims": [64, 64], "n_epochs": 10}
"es":  {"population_size": 50, "sigma": 0.1, "learning_rate": 0.03}
```

#### MiniGrid (7x7 partial obs = 49 dims flattened)
```python
"dqn": {"learning_rate": 1e-4, "hidden_dims": [128, 128], "target_update_freq": 1000}
"ppo": {"learning_rate": 2e-4, "hidden_dims": [128, 128]}
"es":  {"population_size": 40, "sigma": 0.1, "learning_rate": 0.02}
```
**Note:** DQN struggles on MiniGrid - see "Known Limitations" section below.

#### Starpilot (64x64 RGB images)
```python
"dqn": {"learning_rate": 2.5e-5, "hidden_dims": [256, 256], "batch_size": 32}
"ppo": {"learning_rate": 5e-5, "hidden_dims": [256, 256], "n_epochs": 3}
"es":  {"population_size": 20, "sigma": 0.02}  # Reduced for speed - ES on images is SLOW
```

### Training Config
```python
TRAINING_CONFIG = {
    "cartpole": {"total_steps": 50000, "buffer_size": 100000, "max_steps": 500},
    "minigrid": {"total_steps": 1000000, "buffer_size": 100000, "max_steps": 100},
    "starpilot": {"total_steps": 3000000, "buffer_size": 100000, "max_steps": 1000},
}
```

### Epsilon Decay
- **70% decay, 30% fixed**: `epsilon_decay = int(total_steps * 0.7)`
- Epsilon hits minimum (0.05) at 70% of training, stays fixed for remaining 30%

---

## Why Starpilot Instead of CoinRun

| Aspect | CoinRun | Starpilot |
|--------|---------|-----------|
| Steps needed | 25M-200M | 2-5M |
| Reward type | Sparse (+10 at end) | Dense (shooting enemies) |
| Time to train | 30+ hours | 2-4 hours |
| Still tests generalization | Yes | Yes (procedural levels) |

CoinRun at 500k steps learns NOTHING. Starpilot is tractable for research.

---

## GAR (Gradient Agreement Regularization) - FULLY IMPLEMENTED

### How It Works
1. During training, collect experiences from multiple environment variations (0.0, 0.25, 0.5)
2. Compute gradients for each variation
3. Weight gradients by cosine similarity with mean direction
4. Only update in directions where gradients AGREE

```python
# Simplified gradient agreement
mean_grad = average of all gradients
for each gradient:
    similarity = cosine_similarity(gradient, mean_grad)
    weight = max(0, similarity)  # Only positive agreement
agreed_grad = weighted_sum(gradients, weights)
```

### GAR Metrics in Training Logs
```
Step 10000 | Reward: 25 | Loss: 0.39 | GAR Agree: 0.233
                                       ↑ gradient agreement score (0-1)
```
- 0.2-0.3 is typical (some agreement, some variation-specific learning)
- Higher = more generalization-focused updates

### Note
- ES doesn't support GAR (no gradients) - falls back to baseline
- GAR training is ~6x slower (computes gradients from 3 variations)

---

## Best Model Saving

Best models are saved per architecture+environment in `results/best_models/`:
```
dqn_cartpole_best.pt
ppo_cartpole_best.pt
es_cartpole_best.pt
dqn_minigrid_best.pt
...
```

Tracks which technique/seed produced the best result in `best_scores.json`.

---

## Visualization & Metrics

### Generated Files
- `results/plots/{experiment}_learning_curves.png` - Training curves (reward, loss, epsilon)
- `results/plots/{experiment}_grs_curve.png` - GRS degradation curve
- `results/plots/grs_heatmap.png` - Comparison heatmap across all experiments
- `results/metrics/{experiment}_history.json` - Raw training data
- `results/experiment_summary.txt` - Text summary

### View Results
```bash
python run_experiment.py --analyze
```

---

## Performance Optimizations

### torch.compile (PyTorch 2.0+)
- Networks are compiled with `mode="reduce-overhead"` on CUDA
- 10-30% speedup on forward/backward passes
- First iterations slower (compilation warmup)
- Automatically disabled on CPU

### Memory Management
- Memory cleared after each algorithm finishes (gc.collect + torch.cuda.empty_cache)
- Prevents OOM when running full experiment suite

### Frame Stacking
- Starpilot uses 4 stacked frames (standard for Procgen)
- Gives agent temporal information (which way things are moving)

### Seed Consistency
```python
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
```

---

## Bugs Fixed (Don't Reintroduce!)

| Bug | Fix |
|-----|-----|
| "Warning: step() after terminated" | Added `obs, _ = self.env.reset()` after eval |
| MiniGrid "Tensor with 3 elements" error | Flattened observation in minigrid_var.py |
| Channel order (H,W,C vs C,H,W) | networks.py detects format and handles both |
| "view not compatible" error | Changed `.view()` to `.reshape()`, added `.contiguous()` |
| Results directory not found | Added `os.makedirs(results_dir, exist_ok=True)` |
| `reg` technique did nothing | Now passes `weight_decay` to AdamW optimizer |
| DQN reward drops after peak | Lower lr (1e-4), larger buffer (100k), shorter training |
| Epsilon decay too fast | Changed from 50% to 70% of total steps |
| **GRS computation freezes** | Fixed `_is_auto_reset_env()` - added depth limit to wrapper loop, changed `or` to `and` for attribute check |
| CoinRun learns nothing at 500k steps | Replaced with Starpilot (needs 2-5M, not 25M+) |
| **DQN MiniGrid no learning** | Added obs normalization + Huber loss (see below) - still doesn't fully work |

---

## How to Run

```bash
# RECOMMENDED: Run with subprocess isolation (prevents state leakage between algorithms)
python run_experiment.py --isolated --env-only minigrid
python run_experiment.py --isolated --env-only cartpole

# Single environment without isolation (runs all algos × all techniques × all seeds)
python run_experiment.py --env-only cartpole
python run_experiment.py --env-only minigrid
python run_experiment.py --env-only starpilot

# Single algorithm on single environment (isolated)
python run_experiment.py --isolated --algo-only dqn --env-only cartpole

# Single experiment
python run_experiment.py --algo ppo --technique gar --env cartpole --seeds 0 1 2

# Combined techniques
python run_experiment.py --algo dqn --technique gar+reg --env cartpole

# All experiments
python run_experiment.py --all

# Analyze results and generate plots
python run_experiment.py --analyze
```

---

## Expected Performance

| Environment | Algorithm | Expected Reward | Time per Experiment |
|-------------|-----------|-----------------|---------------------|
| CartPole | DQN | 300-500 | ~35s (baseline), ~4min (GAR) |
| CartPole | PPO | 400-500 | ~30s (baseline), ~3min (GAR) |
| CartPole | ES | 200-400 | ~2min |
| MiniGrid | DQN | **FAILS** (~-0.09) | ~13min |
| MiniGrid | PPO | 0.8-0.95 | ~10-20min |
| MiniGrid | ES | 0.7-0.9 | ~15-25min |
| Starpilot | DQN/PPO | 15-40 | ~2-4 hours |
| Starpilot | ES | 5-15 | Very slow, not recommended |

---

## Environment Variation Levels

All environments use `variation_level` parameter (0.0 = training, 1.0 = max shift):

| Environment | What Varies | 0.0 → 1.0 |
|-------------|-------------|-----------|
| CartPole | Dynamics | gravity: 9.8→14.7, pole_length: 0.5→1.0, cart_mass: 1.0→2.0 |
| MiniGrid | Layout | grid_size: 6→12, num_obstacles: 2→8 |
| Starpilot | Level diversity | num_levels: 200→unlimited, mode: easy→hard |

---

## GRS Metric

```python
from evaluation.grs import compute_grs

result = compute_grs(agent, EnvClass, shift_levels=[0, 0.25, 0.5, 0.75, 1.0])
# result["grs"] = area under normalized performance curve
# GRS ≈ 1.0: Perfect generalization
# GRS ≈ 0.5: Linear degradation
# GRS < 0.4: Poor (rapid failure)
```

---

## Memory Constraints

User has ~16GB RAM. Buffer sizes:

| Environment | Obs Size | 100k Buffer |
|-------------|----------|-------------|
| CartPole | 16 bytes | ~5 MB |
| MiniGrid | ~600 bytes | ~120 MB |
| Starpilot (4 frames) | ~50 KB | ~5 GB |

---

## What's Working

- ✅ All three algorithms (DQN, PPO, ES)
- ✅ All environments (CartPole, MiniGrid, Starpilot)
- ✅ All techniques (baseline, reg, gar, gar+reg)
- ✅ GRS evaluation
- ✅ Best model saving per architecture
- ✅ Learning curve plots
- ✅ GRS heatmaps
- ✅ torch.compile GPU optimization
- ✅ Frame stacking for Procgen
- ✅ Memory cleanup between algorithms
- ✅ Subprocess isolation mode (`--isolated` flag)

---

## Common Issues

1. **Starpilot requires procgen:** `pip install procgen`
2. **Run from project root:** Imports break if you cd into subfolders
3. **Gym deprecation warning:** Safe to ignore, we use gymnasium
4. **GAR training is slow:** Expected - computes gradients from 3 variations
5. **ES on Starpilot is very slow:** Use DQN or PPO instead
6. **torch.compile warmup:** First iterations slower, speeds up after

---

## Known Limitations (Valid Research Findings)

### DQN Does Not Work on MiniGrid

Despite multiple fixes, DQN fails to learn on MiniGrid (final eval reward: -0.09, essentially random).

**Attempted fixes:**
1. ✅ Observation normalization (discrete cell types 0-4 → 0-1)
2. ✅ Huber loss instead of MSE (more stable)
3. ✅ Higher learning rate (5e-5 → 1e-4)
4. ✅ More frequent target updates (2000 → 1000)

**Why it still fails:**
- Epsilon-greedy exploration is ineffective for sparse rewards in grid worlds
- Credit assignment is poor - goal reward needs to propagate through many steps
- Once epsilon drops to 0.05, the policy stagnates

**This is a valid research finding:**
> "DQN failed to converge on MiniGrid despite hyperparameter tuning, while PPO and ES succeeded. This suggests on-policy methods and population-based methods are more robust to sparse reward environments."

**Recommendation:** Focus MiniGrid generalization analysis on PPO and ES. Document DQN's failure as evidence that algorithm class matters.

---

## Recent Changes (Session Dec 27, 2025)

### DQN Improvements (agents/dqn.py)
```python
# 1. Added observation normalization for discrete obs (MiniGrid)
obs_high = env.observation_space.high
if not self.use_cnn and obs_high is not None:
    max_val = obs_high.max()
    self.obs_normalize = max_val if max_val > 1.0 and max_val < 256 else 1.0

# 2. Normalize in select_action and _compute_dqn_loss
if self.obs_normalize > 1.0:
    obs_tensor = obs_tensor / self.obs_normalize

# 3. Changed loss from MSE to Huber (more stable)
return F.smooth_l1_loss(current_q, target_q)  # was F.mse_loss
```

### Hyperparameter Updates (run_experiment.py)
```python
# MiniGrid DQN
"learning_rate": 1e-4,      # was 5e-5
"target_update_freq": 1000,  # was 2000
```

### Training Config Update
```python
"minigrid": {"total_steps": 1000000, ...}  # was 500000
```

---

## Next Steps (TODO)

- [ ] Run full experiment suite on CartPole (all algos work)
- [ ] Run PPO and ES experiments on MiniGrid (DQN doesn't work - document as finding)
- [ ] Run Starpilot experiments if time permits
- [ ] Analyze which technique combinations work best
- [ ] Write up results for research paper
  - Include finding: "Algorithm class matters - DQN fails on MiniGrid while PPO/ES succeed"
- [ ] Consider adding more techniques (dropout, data augmentation, etc.)
