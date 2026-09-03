"""
PPO Agent — Proximal Policy Optimization
==========================================
Implements PPO with:
    - Shared MLP encoder → separate actor + critic heads
    - Continuous action space (portfolio weights via Dirichlet distribution)
    - GAE (Generalized Advantage Estimation)
    - Clipped surrogate objective
    - Entropy bonus for exploration
    - Value function loss with clipping

Architecture:
    State → [Linear → LayerNorm → ReLU] × n_layers → shared_embedding
    shared_embedding → actor_head → mean (softmax) + log_std
    shared_embedding → critic_head → value scalar

Action distribution:
    Uses Dirichlet distribution (natural for portfolio weights ∈ simplex).
    Fallback: Gaussian with softmax transform.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Normal
from dataclasses import dataclass, field
from typing import Optional, List
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Network configuration
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Hyperparameters for the PPO agent."""
    # Network
    hidden_dims:    List[int] = field(default_factory=lambda: [256, 128, 64])
    activation:     str  = "relu"          # relu | tanh | elu
    use_layer_norm: bool = True

    # PPO
    clip_eps:       float = 0.2            # clipping epsilon ε
    gamma:          float = 0.99           # discount factor
    gae_lambda:     float = 0.95           # GAE λ
    entropy_coef:   float = 0.01           # entropy bonus weight
    value_coef:     float = 0.5            # value loss weight
    max_grad_norm:  float = 0.5            # gradient clipping

    # Training
    lr:             float = 3e-4           # learning rate
    n_epochs:       int   = 10             # PPO update epochs per rollout
    batch_size:     int   = 64
    rollout_len:    int   = 128            # steps per rollout collection

    # Distribution
    dist_type:      str   = "dirichlet"    # dirichlet | gaussian
    dirichlet_min:  float = 0.1           # min concentration parameter


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "relu",
    use_layer_norm: bool = True,
    output_activation: bool = False,
) -> nn.Sequential:
    """Build an MLP with optional LayerNorm after each hidden layer."""
    act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]

    layers = []
    in_dim = input_dim

    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h_dim))
        layers.append(act_fn())
        in_dim = h_dim

    layers.append(nn.Linear(in_dim, output_dim))
    if output_activation:
        layers.append(act_fn())

    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Actor network
# ---------------------------------------------------------------------------

class ActorNetwork(nn.Module):
    """
    Actor: maps state → action distribution parameters.

    For Dirichlet: outputs concentration parameters α > 0
        α = softplus(linear) + dirichlet_min  → guaranteed positive
        Portfolio weights ~ Dirichlet(α)

    For Gaussian: outputs mean + log_std
        weights = softmax(mean + std * noise)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: PPOConfig,
    ):
        super().__init__()
        self.action_dim  = action_dim
        self.dist_type   = config.dist_type
        self.dirichlet_min = config.dirichlet_min

        self.net = build_mlp(
            input_dim=state_dim,
            hidden_dims=config.hidden_dims,
            output_dim=action_dim,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
        )

        if config.dist_type == "gaussian":
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Orthogonal initialisation (recommended for RL)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        """
        Map state to action distribution.

        Parameters
        ----------
        state : (batch, state_dim)

        Returns
        -------
        distribution : Dirichlet or TransformedNormal
        """
        logits = self.net(state)

        if self.dist_type == "dirichlet":
            # Concentration parameters must be > 0
            concentration = F.softplus(logits) + self.dirichlet_min
            return Dirichlet(concentration)
        else:
            # Gaussian with softmax transform
            mean    = logits
            log_std = self.log_std.clamp(-4, 2)
            std     = log_std.exp()
            return Normal(mean, std)

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple:
        """
        Sample action from distribution.

        Returns (action, log_prob, entropy)
            action   : (batch, action_dim) portfolio weights
            log_prob : (batch,) log probability of action
            entropy  : (batch,) distribution entropy
        """
        dist = self.forward(state)

        if deterministic:
            if self.dist_type == "dirichlet":
                # Mode of Dirichlet: (α_i - 1) / (Σα - K)  [only valid if α > 1]
                concentration = dist.concentration
                mode = (concentration - 1).clamp(min=0)
                total = mode.sum(dim=-1, keepdim=True)
                action = mode / (total + 1e-8)
                # Fallback to mean if mode is degenerate
                action = torch.where(total < 1e-6, dist.mean, action)
            else:
                action = dist.mean
                action = F.softmax(action, dim=-1)
        else:
            raw = dist.rsample()
            if self.dist_type == "gaussian":
                action = F.softmax(raw, dim=-1)
            else:
                action = raw

        log_prob = dist.log_prob(
            raw if not deterministic and self.dist_type == "gaussian" else action
        )
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy()
        if len(entropy.shape) > 1:
            entropy = entropy.sum(dim=-1)

        return action, log_prob, entropy


# ---------------------------------------------------------------------------
# Critic network
# ---------------------------------------------------------------------------

class CriticNetwork(nn.Module):
    """
    Critic: maps state → scalar value estimate V(s).
    Separate from actor (no parameter sharing) for stability.
    """

    def __init__(self, state_dim: int, config: PPOConfig):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_dim,
            hidden_dims=config.hidden_dims,
            output_dim=1,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Final layer: small init for value head
        last_linear = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns value estimate V(s) of shape (batch,)."""
        return self.net(state).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores transitions collected during rollout for PPO update.

    Stores: states, actions, rewards, values, log_probs, dones
    Computes: returns, advantages (GAE)
    """

    def __init__(self, rollout_len: int, state_dim: int, action_dim: int):
        self.rollout_len = rollout_len
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.reset()

    def reset(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.values    = []
        self.log_probs = []
        self.dones     = []
        self.ptr       = 0

    def add(
        self,
        state:    np.ndarray,
        action:   np.ndarray,
        reward:   float,
        value:    float,
        log_prob: float,
        done:     bool,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.ptr += 1

    def compute_returns_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple:
        """
        Compute discounted returns and GAE advantages.

        GAE: A_t = Σ (γλ)^k · δ_{t+k}
             where δ_t = r_t + γ·V(s_{t+1}) - V(s_t)

        Returns (returns, advantages) as numpy arrays.
        """
        n       = len(self.rewards)
        returns = np.zeros(n, dtype=np.float32)
        advs    = np.zeros(n, dtype=np.float32)

        values   = np.array(self.values + [last_value], dtype=np.float32)
        rewards  = np.array(self.rewards, dtype=np.float32)
        dones    = np.array(self.dones, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            delta    = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae      = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advs[t]  = gae
            returns[t] = gae + values[t]

        return returns, advs

    def get_batches(
        self,
        batch_size: int,
        returns: np.ndarray,
        advantages: np.ndarray,
    ):
        """
        Yield random mini-batches for PPO update epochs.

        Yields dicts with tensors: states, actions, log_probs, returns, advantages.
        """
        n      = len(self.states)
        idx    = np.random.permutation(n)
        states = torch.FloatTensor(np.array(self.states))
        actions= torch.FloatTensor(np.array(self.actions))
        lps    = torch.FloatTensor(np.array(self.log_probs))
        rets   = torch.FloatTensor(returns)
        advs   = torch.FloatTensor(advantages)

        # Normalise advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            yield {
                "states":     states[batch_idx],
                "actions":    actions[batch_idx],
                "log_probs":  lps[batch_idx],
                "returns":    rets[batch_idx],
                "advantages": advs[batch_idx],
            }


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    Proximal Policy Optimization agent for portfolio allocation.

    Usage:
        agent = PPOAgent(state_dim=..., action_dim=..., config=PPOConfig())
        action, log_prob, value = agent.act(state)
        agent.update(buffer, last_value)
        agent.save("models/ppo_agent.pt")
        agent.load("models/ppo_agent.pt")
    """

    def __init__(
        self,
        state_dim:  int,
        action_dim: int,
        config:     PPOConfig = None,
        device:     str = "cpu",
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.config     = config or PPOConfig()
        self.device     = torch.device(device)

        # Networks
        self.actor  = ActorNetwork(state_dim, action_dim, self.config).to(self.device)
        self.critic = CriticNetwork(state_dim, self.config).to(self.device)

        # Optimisers
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=self.config.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr)

        # Tracking
        self.update_count = 0
        self.train_losses = []

    def act(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple:
        """
        Select action given state.

        Returns (action, log_prob, value) as numpy scalars/arrays.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(state_t, deterministic)
            value = self.critic(state_t)

        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.squeeze(0).cpu()),
            float(value.squeeze(0).cpu()),
        )

    def evaluate_actions(
        self,
        states:  torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple:
        """
        Evaluate actions under current policy (used in PPO update).

        Returns (log_probs, values, entropy).
        """
        dist      = self.actor.forward(states)
        log_probs = dist.log_prob(actions)
        if len(log_probs.shape) > 1:
            log_probs = log_probs.sum(dim=-1)
        entropy   = dist.entropy()
        if len(entropy.shape) > 1:
            entropy = entropy.sum(dim=-1)
        values    = self.critic(states)
        return log_probs, values, entropy

    def update(self, buffer: RolloutBuffer, last_value: float) -> dict:
        """
        Run PPO update using collected rollout.

        Parameters
        ----------
        buffer     : filled RolloutBuffer
        last_value : V(s_T) for bootstrapping

        Returns dict of training losses.
        """
        returns, advantages = buffer.compute_returns_advantages(
            last_value, self.config.gamma, self.config.gae_lambda
        )

        actor_losses, critic_losses, entropy_vals = [], [], []

        for epoch in range(self.config.n_epochs):
            for batch in buffer.get_batches(self.config.batch_size, returns, advantages):
                states     = batch["states"].to(self.device)
                actions    = batch["actions"].to(self.device)
                old_lp     = batch["log_probs"].to(self.device)
                rets       = batch["returns"].to(self.device)
                advs       = batch["advantages"].to(self.device)

                # Evaluate under current policy
                new_lp, values, entropy = self.evaluate_actions(states, actions)

                # PPO clipped surrogate loss
                ratio      = (new_lp - old_lp).exp()
                surr1      = ratio * advs
                surr2      = ratio.clamp(1 - self.config.clip_eps,
                                         1 + self.config.clip_eps) * advs
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = F.mse_loss(values, rets)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = (actor_loss
                              + self.config.value_coef * critic_loss
                              + self.config.entropy_coef * entropy_loss)

                # Actor update
                self.actor_opt.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_opt.step()

                # Critic update
                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_opt.step()

                actor_losses.append(float(actor_loss))
                critic_losses.append(float(critic_loss))
                entropy_vals.append(float(-entropy_loss))

        self.update_count += 1

        metrics = {
            "actor_loss":  round(float(np.mean(actor_losses)), 6),
            "critic_loss": round(float(np.mean(critic_losses)), 6),
            "entropy":     round(float(np.mean(entropy_vals)), 6),
            "update":      self.update_count,
        }
        self.train_losses.append(metrics)
        return metrics

    def save(self, path: str) -> None:
        """Save agent networks and config to disk."""
        torch.save({
            "actor":       self.actor.state_dict(),
            "critic":      self.critic.state_dict(),
            "actor_opt":   self.actor_opt.state_dict(),
            "critic_opt":  self.critic_opt.state_dict(),
            "config":      self.config,
            "state_dim":   self.state_dim,
            "action_dim":  self.action_dim,
            "update_count":self.update_count,
        }, path)

    def load(self, path: str) -> None:
        """Load agent from disk."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.update_count = ckpt.get("update_count", 0)

    def param_count(self) -> dict:
        """Return parameter counts for actor and critic."""
        actor_p  = sum(p.numel() for p in self.actor.parameters())
        critic_p = sum(p.numel() for p in self.critic.parameters())
        return {
            "actor_params":  actor_p,
            "critic_params": critic_p,
            "total_params":  actor_p + critic_p,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from env.trading_env import TradingEnv, EnvConfig, generate_market_data

    print("=" * 60)
    print("PPO Agent Demo")
    print("=" * 60)

    n_assets = 5
    data     = generate_market_data(n_assets=n_assets, n_days=504, seed=42)
    cfg      = EnvConfig(n_assets=n_assets)
    env      = TradingEnv(prices=data, config=cfg)
    obs      = env.reset()

    state_dim  = obs.shape[0]
    action_dim = n_assets + 1   # assets + cash

    ppo_cfg = PPOConfig(
        hidden_dims=[128, 64],
        dist_type="dirichlet",
        lr=3e-4,
    )
    agent = PPOAgent(state_dim, action_dim, ppo_cfg)

    print(f"\n  State dim  : {state_dim}")
    print(f"  Action dim : {action_dim}")
    print(f"  Params     : {agent.param_count()}")

    # Collect one rollout and do one update
    buffer = RolloutBuffer(ppo_cfg.rollout_len, state_dim, action_dim)
    obs    = env.reset()
    done   = False
    steps  = 0

    while steps < ppo_cfg.rollout_len and not done:
        action, log_prob, value = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, value, log_prob, done)
        obs   = next_obs
        steps += 1

    _, _, last_val = agent.act(obs)
    metrics = agent.update(buffer, last_val)

    print(f"\n  Rollout steps : {steps}")
    print(f"  Update metrics:")
    for k, v in metrics.items():
        print(f"    {k:<14}: {v}")

    # Test action distribution
    test_state = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        dist = agent.actor.forward(test_state)
    print(f"\n  Action dist type : {type(dist).__name__}")
    print(f"  Sample action    : {dist.sample().squeeze(0).numpy().round(4)}")
    print(f"  Mean action      : {dist.mean.squeeze(0).numpy().round(4)}")
    print(f"  Sum of weights   : {dist.mean.squeeze(0).sum().item():.6f}")
