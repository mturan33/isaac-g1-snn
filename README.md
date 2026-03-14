# isaac-g1-snn

Spiking Neural Network (LIF) policy for Unitree G1 humanoid locomotion, trained with PPO in NVIDIA Isaac Lab.

Biologically-inspired locomotion control using Spiking Neural Networks (SNN) trained with PPO in NVIDIA Isaac Lab.

## Why SNN for Robot Locomotion?

Standard MLP policies treat each timestep independently. SNN neurons carry **membrane potential** across timesteps — natural temporal memory without explicit recurrence. Locomotion is inherently rhythmic; SNN's temporal dynamics are a natural fit.

| Feature | MLP Policy | SNN Policy |
|---------|-----------|------------|
| Communication | Continuous values | Binary spikes |
| Temporal memory | None (stateless) | Membrane potential (stateful) |
| Computation | Dense (every neuron active) | Sparse (only spiking neurons) |
| Biological plausibility | Low | High |
| Neuromorphic hardware | Not compatible | Deployable on Loihi/SpiNNaker |

## Architecture

```
Observations (46D)
      ↓
[Linear Encoder] → Current injection
      ↓
[LIF Layer 1] (256 neurons, learnable β)
      ↓  spike trains
[LIF Layer 2] (128 neurons, learnable β)
      ↓  spike trains
[Linear Decoder] → Action mean (15D)
```

Each "forward pass" runs the SNN for `num_steps` internal timesteps (default=8), accumulating spikes via rate coding to produce continuous actions.

## Setup

```bash
# In your Isaac Lab conda environment:
pip install snntorch --break-system-packages

# Copy files to your Isaac Lab project
cp snn_actor_critic.py /path/to/your/isaaclab/project/
cp snn_ppo.py /path/to/your/isaaclab/project/
cp train_snn_g1.py /path/to/your/isaaclab/project/
```

## Training

```powershell
# From Isaac Lab root:
./isaaclab.bat -p train_snn_g1.py --num_envs 2048 --max_iterations 3000

# Compare with MLP baseline:
./isaaclab.bat -p train_snn_g1.py --num_envs 2048 --max_iterations 3000 --use_mlp
```

## Files

| File | Description |
|------|-------------|
| `snn_actor_critic.py` | SNN actor + critic networks with LIF neurons |
| `snn_ppo.py` | PPO algorithm adapted for SNN (handles membrane state) |
| `train_snn_g1.py` | Isaac Lab training entry point for G1 |
| `play_snn_g1.py` | Visualization / demo script |

## Key Design Decisions

1. **Surrogate Gradient**: ATan surrogate for backward pass (spike is non-differentiable)
2. **Rate Coding Output**: Spike count over num_steps → continuous action (smooth enough for motor control)
3. **Learnable Beta**: Each neuron learns its own decay rate (τ), allowing mixed fast/slow dynamics
4. **Separate Critic**: Critic uses standard MLP (value estimation doesn't need temporal dynamics)
5. **Membrane State Reset**: Reset per episode, carry within episode (temporal continuity)

## References

- Eon Systems: Embodied Fly Brain Emulation (2026) — Inspiration
- Shiu et al. (2024) "A Drosophila computational brain model" — Nature
- snnTorch: https://snntorch.readthedocs.io/
- Surrogate Gradient Learning in SNN: Neftci et al. (2019)
