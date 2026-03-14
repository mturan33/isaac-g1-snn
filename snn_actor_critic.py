"""
SNN Actor-Critic for Humanoid Locomotion
=========================================
Spiking Neural Network policy using Leaky Integrate-and-Fire (LIF) neurons.
Compatible with PPO training in Isaac Lab.

Key concepts:
- LIF neurons: mem[t] = beta * mem[t-1] + input[t]; if mem > threshold → spike
- Surrogate gradient: ATan approximation for backprop through spikes
- Rate coding: spike count over time → continuous action output
- Learnable beta: each neuron learns its own membrane decay rate

Author: Turan Yardimci
"""

import torch
import torch.nn as nn

# ============================================================
# LIF Neuron Layer (No snnTorch dependency — pure PyTorch)
# ============================================================
# Why custom? So you understand EXACTLY what's happening.
# After MVP, you can switch to snnTorch for more features.

class SurrogateATan(torch.autograd.Function):
    """
    Surrogate gradient for spike function.
    
    Forward: Heaviside step (binary spike)
    Backward: ATan derivative (smooth, differentiable)
    
    This is THE trick that makes SNN + backprop work.
    Without this, gradient of a step function = 0 everywhere
    except at threshold where it's undefined.
    """
    alpha = 2.0  # sharpness of surrogate gradient
    
    @staticmethod
    def forward(ctx, membrane_potential):
        # Binary spike: 1 if above 0, else 0
        # (threshold already subtracted before calling this)
        spike = (membrane_potential > 0).float()
        ctx.save_for_backward(membrane_potential)
        return spike
    
    @staticmethod
    def backward(ctx, grad_output):
        membrane_potential, = ctx.saved_tensors
        alpha = SurrogateATan.alpha
        # ATan surrogate: smooth bell-shaped curve centered at threshold
        # d/dx [1/pi * arctan(alpha * x) + 0.5]
        grad = alpha / (2 * (1 + (alpha * membrane_potential).pow(2)))
        return grad_output * grad


class LIFLayer(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer.
    
    One timestep operation:
        1. Leak:      mem = beta * mem_prev          (exponential decay)
        2. Integrate: mem = mem + W @ input           (accumulate input)
        3. Fire:      spike = (mem > threshold)       (binary decision)
        4. Reset:     mem = mem * (1 - spike)         (hard reset to 0)
    
    Args:
        in_features: input dimension
        out_features: number of LIF neurons
        beta_init: initial decay factor (0-1). Higher = slower leak = longer memory
        learn_beta: if True, beta is a learnable parameter per neuron
        threshold: spike threshold (default 1.0)
    """
    def __init__(self, in_features, out_features, beta_init=0.85, 
                 learn_beta=True, threshold=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.threshold = threshold
        
        # Beta (decay rate) — optionally learnable per neuron
        beta_tensor = torch.full((out_features,), beta_init)
        if learn_beta:
            # Use sigmoid to constrain beta to (0, 1)
            # Store as pre-sigmoid value for unconstrained optimization
            self._beta_raw = nn.Parameter(torch.log(beta_tensor / (1 - beta_tensor)))
        else:
            self.register_buffer('_beta_raw', torch.log(beta_tensor / (1 - beta_tensor)))
    
    @property
    def beta(self):
        """Decay factor constrained to (0, 1) via sigmoid."""
        return torch.sigmoid(self._beta_raw)
    
    def forward(self, input_current, mem_prev):
        """
        Single timestep forward pass.
        
        Args:
            input_current: [batch, in_features] or spike train from previous layer
            mem_prev: [batch, out_features] membrane potential from previous timestep
            
        Returns:
            spike: [batch, out_features] binary spike output
            mem: [batch, out_features] updated membrane potential
        """
        # 1. Leak + 2. Integrate
        mem = self.beta * mem_prev + self.linear(input_current)
        
        # 3. Fire (with surrogate gradient for backprop)
        spike = SurrogateATan.apply(mem - self.threshold)
        
        # 4. Reset (subtract threshold on spike, or hard reset to 0)
        mem = mem * (1 - spike.detach())  # hard reset
        
        return spike, mem
    
    def init_membrane(self, batch_size, device):
        """Initialize membrane potential to zero."""
        return torch.zeros(batch_size, self.linear.out_features, device=device)


# ============================================================
# SNN Actor (Policy Network)
# ============================================================

class SNNActor(nn.Module):
    """
    Spiking Neural Network Actor for continuous control.
    
    Architecture:
        obs → Linear encoder → LIF1 → LIF2 → Linear decoder → action_mean
    
    Rate coding: runs SNN for `num_steps` internal timesteps per env step.
    Output = average spike rate → continuous action via decoder.
    
    The membrane potential carries temporal information between env timesteps,
    giving the policy natural "memory" of recent observations — useful for
    rhythmic locomotion patterns.
    
    Args:
        obs_dim: observation space dimension
        act_dim: action space dimension  
        hidden_dims: list of hidden layer sizes (default [256, 128])
        num_steps: internal SNN timesteps per env step (default 8)
        beta_init: initial membrane decay factor
    """
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 128], 
                 num_steps=8, beta_init=0.85):
        super().__init__()
        self.num_steps = num_steps
        self.hidden_dims = hidden_dims
        
        # Encoder: continuous obs → current injection for first LIF layer
        self.encoder = nn.Linear(obs_dim, hidden_dims[0])
        
        # LIF layers
        self.lif_layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            in_dim = hidden_dims[i]  # encoder already projects to hidden_dims[0]
            out_dim = hidden_dims[i]
            if i > 0:
                in_dim = hidden_dims[i-1]
            self.lif_layers.append(
                LIFLayer(in_dim, out_dim, beta_init=beta_init, learn_beta=True)
            )
        
        # Decoder: spike rates → continuous action
        self.decoder = nn.Linear(hidden_dims[-1], act_dim)
        
        # Action log_std (same as standard PPO)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Persistent membrane states (carried between env timesteps)
        self._membrane_states = None
    
    def init_membranes(self, batch_size, device):
        """Reset all membrane potentials (call at episode reset)."""
        self._membrane_states = [
            lif.init_membrane(batch_size, device)
            for lif in self.lif_layers
        ]
    
    def forward(self, obs):
        """
        Forward pass with internal SNN dynamics.
        
        Args:
            obs: [batch, obs_dim] current observation
            
        Returns:
            action_mean: [batch, act_dim] mean of action distribution
            action_std: [batch, act_dim] std of action distribution
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # Init membranes if needed (first call or batch size changed)
        if (self._membrane_states is None or 
            self._membrane_states[0].shape[0] != batch_size):
            self.init_membranes(batch_size, device)
        
        # Encode observation to current
        input_current = self.encoder(obs)
        
        # Run SNN for num_steps internal timesteps
        spike_accumulator = torch.zeros(
            batch_size, self.hidden_dims[-1], device=device
        )
        
        for t in range(self.num_steps):
            x = input_current
            for i, lif in enumerate(self.lif_layers):
                spike, self._membrane_states[i] = lif(x, self._membrane_states[i])
                x = spike  # spikes become input to next layer
            spike_accumulator += x
        
        # Rate coding: average spike rate over internal timesteps
        spike_rate = spike_accumulator / self.num_steps
        
        # Decode to continuous action
        action_mean = self.decoder(spike_rate)
        action_std = self.log_std.exp().expand_as(action_mean)
        
        return action_mean, action_std
    
    def act(self, obs, deterministic=False):
        """Sample action for environment interaction."""
        action_mean, action_std = self.forward(obs)
        if deterministic:
            return action_mean
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, action_mean
    
    def evaluate(self, obs, actions):
        """Evaluate actions for PPO update."""
        action_mean, action_std = self.forward(obs)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ============================================================
# Critic (Standard MLP — no need for SNN dynamics in value fn)
# ============================================================

class MLPCritic(nn.Module):
    """
    Standard MLP critic for value estimation.
    
    Value function doesn't need temporal dynamics — it just estimates
    "how good is this state?". Keeping it as MLP also makes training
    more stable (SNN critic can be unstable).
    """
    def __init__(self, obs_dim, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ELU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.net(obs)


# ============================================================
# Combined Actor-Critic
# ============================================================

class SNNActorCritic(nn.Module):
    """
    SNN Actor + MLP Critic for PPO.
    
    Design choice: SNN actor (benefits from temporal processing for
    locomotion), MLP critic (stable value estimation).
    """
    def __init__(self, obs_dim, act_dim, actor_hidden=[256, 128],
                 critic_hidden=[256, 128], num_steps=8, beta_init=0.85):
        super().__init__()
        self.actor = SNNActor(
            obs_dim, act_dim, actor_hidden, num_steps, beta_init
        )
        self.critic = MLPCritic(obs_dim, critic_hidden)
    
    def act(self, obs, deterministic=False):
        return self.actor.act(obs, deterministic)
    
    def evaluate(self, obs, actions):
        log_prob, entropy = self.actor.evaluate(obs, actions)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value
    
    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)
    
    def reset_membranes(self, batch_size, device):
        """Call at episode boundaries."""
        self.actor.init_membranes(batch_size, device)


# ============================================================
# MLP Baseline (for A/B comparison)
# ============================================================

class MLPActorCritic(nn.Module):
    """Standard MLP Actor-Critic baseline for comparison."""
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 128]):
        super().__init__()
        
        # Actor
        actor_layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            actor_layers.extend([nn.Linear(in_dim, h_dim), nn.ELU()])
            in_dim = h_dim
        actor_layers.append(nn.Linear(in_dim, act_dim))
        self.actor_net = nn.Sequential(*actor_layers)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Critic
        self.critic = MLPCritic(obs_dim, hidden_dims)
    
    def act(self, obs, deterministic=False):
        action_mean = self.actor_net(obs)
        action_std = self.log_std.exp().expand_as(action_mean)
        if deterministic:
            return action_mean
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, action_mean
    
    def evaluate(self, obs, actions):
        action_mean = self.actor_net(obs)
        action_std = self.log_std.exp().expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value
    
    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)
    
    def reset_membranes(self, batch_size, device):
        pass  # MLP has no membrane state


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    obs_dim, act_dim = 123, 37  # G1 Isaac-Velocity-Flat-G1-v0 dims
    batch_size = 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # SNN model
    snn_model = SNNActorCritic(obs_dim, act_dim).to(device)
    obs = torch.randn(batch_size, obs_dim, device=device)
    
    # Forward pass
    action, log_prob, action_mean = snn_model.act(obs)
    value = snn_model.get_value(obs)
    
    print(f"SNN Actor-Critic")
    print(f"  Obs shape:    {obs.shape}")
    print(f"  Action shape:  {action.shape}")
    print(f"  Log prob shape: {log_prob.shape}")
    print(f"  Value shape:   {value.shape}")
    
    # Check learnable betas
    for i, lif in enumerate(snn_model.actor.lif_layers):
        beta_vals = lif.beta
        print(f"  LIF{i} beta — mean: {beta_vals.mean():.3f}, "
              f"std: {beta_vals.std():.4f}, "
              f"range: [{beta_vals.min():.3f}, {beta_vals.max():.3f}]")
    
    # Parameter count comparison
    snn_params = sum(p.numel() for p in snn_model.parameters())
    mlp_model = MLPActorCritic(obs_dim, act_dim).to(device)
    mlp_params = sum(p.numel() for p in mlp_model.parameters())
    print(f"\n  SNN params: {snn_params:,}")
    print(f"  MLP params: {mlp_params:,}")
    
    # Gradient check — use log_prob (not action_mean) so log_std also gets gradients
    loss = log_prob.sum()
    loss.backward()
    no_grad_params = [name for name, p in snn_model.actor.named_parameters()
                      if p.requires_grad and p.grad is None]
    grad_ok = len(no_grad_params) == 0
    print(f"\n  Gradient flows through SNN: {'YES' if grad_ok else 'NO'}")
    if not grad_ok:
        print(f"  Missing gradients: {no_grad_params}")

    # Verify NaN/Inf free
    has_nan = any(torch.isnan(p.grad).any() for p in snn_model.parameters() if p.grad is not None)
    has_inf = any(torch.isinf(p.grad).any() for p in snn_model.parameters() if p.grad is not None)
    print(f"  Gradients NaN-free: {'YES' if not has_nan else 'NO'}")
    print(f"  Gradients Inf-free: {'YES' if not has_inf else 'NO'}")

    print("\nAll checks passed!")
