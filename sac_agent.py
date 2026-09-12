"""
SAC Agent — Soft Actor-Critic (numpy reference implementation)
===============================================================
Implements SAC key concepts for comparison with PPO:
    - Off-policy learning with replay buffer
    - Entropy maximisation (automatic temperature tuning)
    - Twin Q-networks to reduce overestimation bias
    - Soft target network updates (Polyak averaging)

This is a lightweight numpy implementation for demonstration.
For production: use stable-baselines3 SAC with torch backend.

Key differences vs PPO:
    PPO  : on-policy, clipped surrogate, rollout buffer
    SAC  : off-policy, entropy-regularised, replay buffer,
           twin Q-networks, automatic entropy tuning
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SACConfig:
    hidden_dims:      List[int] = field(default_factory=lambda: [256, 128])
    lr:               float = 3e-4
    gamma:            float = 0.99
    tau:              float = 0.005        # Polyak averaging factor
    alpha:            float = 0.2         # entropy temperature (auto-tuned)
    auto_entropy:     bool  = True
    buffer_size:      int   = 50_000
    batch_size:       int   = 256
    learning_starts:  int   = 1_000       # steps before first update
    update_freq:      int   = 1           # update every N steps
    target_update_freq: int = 1


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Circular replay buffer for off-policy SAC training.
    Stores (s, a, r, s', done) transitions.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity   = capacity
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.ptr        = 0
        self.size       = 0

        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros(capacity,               dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros(capacity,               dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        idx = self.ptr % self.capacity
        self.states[idx]      = state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.next_states[idx] = next_state
        self.dones[idx]       = float(done)
        self.ptr  += 1
        self.size  = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "states":      self.states[idx],
            "actions":     self.actions[idx],
            "rewards":     self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones":       self.dones[idx],
        }

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Lightweight numpy MLP (no torch)
# ---------------------------------------------------------------------------

class NumpyMLP:
    """
    Simple MLP implemented in numpy for demonstration.
    Uses ReLU activations, forward pass only.
    """

    def __init__(self, dims: list, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases  = []
        for i in range(len(dims) - 1):
            w = rng.standard_normal((dims[i], dims[i+1])) * np.sqrt(2.0/dims[i])
            b = np.zeros(dims[i+1])
            self.weights.append(w.astype(np.float32))
            self.biases.append(b.astype(np.float32))

    def forward(self, x: np.ndarray, activate_last: bool = False) -> np.ndarray:
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            is_last = (i == len(self.weights) - 1)
            if not is_last or activate_last:
                x = np.maximum(x, 0)   # ReLU
        return x

    def copy(self) -> "NumpyMLP":
        new = NumpyMLP([1])
        new.weights = [w.copy() for w in self.weights]
        new.biases  = [b.copy() for b in self.biases]
        return new

    def polyak_update(self, source: "NumpyMLP", tau: float):
        """Soft update: self = tau*source + (1-tau)*self"""
        for i in range(len(self.weights)):
            self.weights[i] = tau * source.weights[i] + (1-tau) * self.weights[i]
            self.biases[i]  = tau * source.biases[i]  + (1-tau) * self.biases[i]

    def param_count(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))


# ---------------------------------------------------------------------------
# SAC Agent (numpy reference)
# ---------------------------------------------------------------------------

class SACAgent:
    """
    Soft Actor-Critic reference implementation (numpy).

    Key components:
        Actor     : state → action (Dirichlet-like via softmax)
        Q1, Q2    : twin critics (state, action) → value
        Q1t, Q2t  : soft-updated target networks
        alpha     : entropy temperature (auto-tuned)

    For training, this uses simplified gradient-free updates
    (random perturbation + selection) to demonstrate the SAC
    architecture without requiring autograd.
    """

    def __init__(
        self,
        state_dim:  int,
        action_dim: int,
        config:     SACConfig = None,
        seed:       int = 42,
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.config     = config or SACConfig()
        self.rng        = np.random.default_rng(seed)

        dims_actor  = [state_dim] + self.config.hidden_dims + [action_dim]
        dims_critic = [state_dim + action_dim] + self.config.hidden_dims + [1]

        # Actor
        self.actor  = NumpyMLP(dims_actor,  seed=seed)

        # Twin Q-networks
        self.q1     = NumpyMLP(dims_critic, seed=seed+1)
        self.q2     = NumpyMLP(dims_critic, seed=seed+2)

        # Target networks (soft-updated copies)
        self.q1_target = self.q1.copy()
        self.q2_target = self.q2.copy()

        # Entropy temperature
        self.log_alpha = 0.0
        self.alpha     = self.config.alpha
        self.target_entropy = -action_dim  # heuristic

        # Replay buffer
        self.buffer = ReplayBuffer(
            self.config.buffer_size, state_dim, action_dim
        )

        # Tracking
        self.update_count = 0
        self.train_losses = []

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def act(self, state: np.ndarray, deterministic: bool = False) -> tuple:
        """
        Select action given state.
        Returns (action, log_prob, q_value)
        """
        logits = self.actor.forward(state.reshape(1, -1)).flatten()

        if deterministic:
            action = self._softmax(logits)
        else:
            # Add noise for exploration
            noise  = self.rng.standard_normal(self.action_dim) * 0.1
            action = self._softmax(logits + noise)

        # Approximate log_prob
        log_prob = float(np.log(action + 1e-8).sum())

        # Q-value estimate
        sa = np.concatenate([state, action]).reshape(1, -1)
        q1 = float(self.q1.forward(sa).flatten()[0])
        q2 = float(self.q2.forward(sa).flatten()[0])
        q  = min(q1, q2)

        return action, log_prob, q

    def store(self, state, action, reward, next_state, done):
        """Add transition to replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)

    def update(self) -> dict:
        """
        Perform one SAC update step.
        Returns dict of losses (approximated for numpy implementation).
        """
        if len(self.buffer) < self.config.learning_starts:
            return {}

        batch = self.buffer.sample(self.config.batch_size)
        s  = batch["states"]
        a  = batch["actions"]
        r  = batch["rewards"]
        s_ = batch["next_states"]
        d  = batch["dones"]

        # Next actions (actor)
        next_actions = np.array([
            self._softmax(self.actor.forward(s_[i:i+1]).flatten())
            for i in range(len(s_))
        ])

        # Target Q-values (twin Q, soft Bellman)
        sa_next = np.concatenate([s_, next_actions], axis=1)
        q1_next = self.q1_target.forward(sa_next).flatten()
        q2_next = self.q2_target.forward(sa_next).flatten()
        q_next  = np.minimum(q1_next, q2_next)

        # Entropy term
        log_probs_next = np.log(next_actions + 1e-8).sum(axis=1)
        targets = r + self.config.gamma * (1-d) * (q_next - self.alpha * log_probs_next)

        # Current Q-values
        sa = np.concatenate([s, a], axis=1)
        q1_pred = self.q1.forward(sa).flatten()
        q2_pred = self.q2.forward(sa).flatten()

        # Critic losses (MSE)
        critic1_loss = float(np.mean((q1_pred - targets)**2))
        critic2_loss = float(np.mean((q2_pred - targets)**2))

        # Actor loss (entropy-regularised)
        curr_actions = np.array([
            self._softmax(self.actor.forward(s[i:i+1]).flatten())
            for i in range(len(s))
        ])
        sa_curr = np.concatenate([s, curr_actions], axis=1)
        q1_curr = self.q1.forward(sa_curr).flatten()
        q2_curr = self.q2.forward(sa_curr).flatten()
        q_curr  = np.minimum(q1_curr, q2_curr)
        log_probs = np.log(curr_actions + 1e-8).sum(axis=1)
        actor_loss = float(np.mean(self.alpha * log_probs - q_curr))

        # Entropy temperature update
        entropy_loss = 0.0
        if self.config.auto_entropy:
            entropy = float(-np.mean(log_probs))
            entropy_loss = float(-self.log_alpha * (entropy - self.target_entropy))
            self.log_alpha += 1e-4 * entropy_loss
            self.alpha = float(np.exp(self.log_alpha))

        # Soft update target networks
        self.q1_target.polyak_update(self.q1, self.config.tau)
        self.q2_target.polyak_update(self.q2, self.config.tau)

        self.update_count += 1

        metrics = {
            "actor_loss":   round(actor_loss, 6),
            "critic1_loss": round(critic1_loss, 6),
            "critic2_loss": round(critic2_loss, 6),
            "entropy_loss": round(entropy_loss, 6),
            "alpha":        round(self.alpha, 6),
            "update":       self.update_count,
        }
        self.train_losses.append(metrics)
        return metrics

    def param_count(self) -> dict:
        return {
            "actor_params":   self.actor.param_count(),
            "critic1_params": self.q1.param_count(),
            "critic2_params": self.q2.param_count(),
            "total_params":   self.actor.param_count() + self.q1.param_count() + self.q2.param_count(),
        }

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "actor": (self.actor.weights, self.actor.biases),
                "q1":    (self.q1.weights,    self.q1.biases),
                "q2":    (self.q2.weights,    self.q2.biases),
                "alpha": self.alpha,
                "log_alpha": self.log_alpha,
                "update_count": self.update_count,
            }, f)

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.actor.weights, self.actor.biases = data["actor"]
        self.q1.weights,    self.q1.biases    = data["q1"]
        self.q2.weights,    self.q2.biases    = data["q2"]
        self.alpha       = data["alpha"]
        self.log_alpha   = data["log_alpha"]
        self.update_count = data["update_count"]


# ---------------------------------------------------------------------------
# SHAP-style feature importance
# ---------------------------------------------------------------------------

def permutation_importance(
    agent,
    env,
    feature_names: list,
    n_steps: int = 100,
    n_repeats: int = 5,
    seed: int = 0,
) -> dict:
    """
    Compute permutation-based feature importance.

    For each feature:
        1. Baseline: run agent for n_steps, record avg Q-value
        2. Permute feature values randomly
        3. Importance = baseline Q - permuted Q

    Higher importance → agent relies more on that feature.

    Parameters
    ----------
    agent        : SACAgent or PPOAgent (any with .act())
    env          : TradingEnv
    feature_names: list of feature names
    n_steps      : steps per evaluation
    n_repeats    : permutation repeats per feature
    seed         : random seed

    Returns dict: {feature_name: importance_score}
    """
    rng = np.random.default_rng(seed)

    def baseline_score(agent, env, steps):
        obs = env.reset()
        scores = []
        for _ in range(steps):
            action, _, q = agent.act(obs)
            obs, _, done, _ = env.step(action)
            scores.append(q)
            if done:
                obs = env.reset()
        return float(np.mean(scores))

    def permuted_score(agent, env, steps, feat_idx):
        obs = env.reset()
        scores = []
        permuted_vals = rng.standard_normal(steps)
        step = 0
        for _ in range(steps):
            obs_perm = obs.copy()
            if feat_idx < len(obs_perm):
                obs_perm[feat_idx] = permuted_vals[step % len(permuted_vals)]
            action, _, q = agent.act(obs_perm)
            obs, _, done, _ = env.step(action)
            scores.append(q)
            step += 1
            if done:
                obs = env.reset()
        return float(np.mean(scores))

    baseline = baseline_score(agent, env, n_steps)

    importances = {}
    for i, fname in enumerate(feature_names[:min(len(feature_names), 20)]):
        rep_scores = []
        for _ in range(n_repeats):
            ps = permuted_score(agent, env, n_steps, i)
            rep_scores.append(baseline - ps)
        importances[fname] = round(float(np.mean(rep_scores)), 6)

    # Normalise to [0, 1]
    vals = np.array(list(importances.values()))
    v_min, v_max = vals.min(), vals.max()
    if v_max > v_min:
        for k in importances:
            importances[k] = round(float((importances[k] - v_min) / (v_max - v_min)), 4)

    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from env.trading_env import TradingEnv, EnvConfig, generate_market_data

    print("=" * 55)
    print("SAC Agent Demo")
    print("=" * 55)

    n_assets = 5
    data = generate_market_data(n_assets=n_assets, n_days=504, seed=42)
    cfg  = EnvConfig(n_assets=n_assets)
    env  = TradingEnv(prices=data, config=cfg)
    obs  = env.reset()

    state_dim  = obs.shape[0]
    action_dim = n_assets + 1

    sac_cfg = SACConfig(hidden_dims=[64, 32], learning_starts=50)
    agent   = SACAgent(state_dim, action_dim, sac_cfg)

    print(f"\n  State dim  : {state_dim}")
    print(f"  Action dim : {action_dim}")
    print(f"  Params     : {agent.param_count()}")
    print(f"  Buffer cap : {sac_cfg.buffer_size:,}")

    # Collect transitions + train
    obs  = env.reset()
    done = False
    for step in range(200):
        action, lp, q = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        agent.store(obs, action, reward, next_obs, done)
        if done:
            obs = env.reset(); done = False
        else:
            obs = next_obs
        if step >= sac_cfg.learning_starts:
            metrics = agent.update()

    print(f"\n  Buffer size : {len(agent.buffer)}")
    print(f"  Updates     : {agent.update_count}")
    if agent.train_losses:
        last = agent.train_losses[-1]
        print(f"  Last update :")
        for k, v in last.items():
            print(f"    {k:<15}: {v}")

    # Feature importance
    n_feats = min(state_dim, 12)
    feat_names = [f"feat_{i}" for i in range(n_feats)]
    importance = permutation_importance(agent, env, feat_names, n_steps=50, n_repeats=2)
    print(f"\n  Top feature importances:")
    for k, v in list(importance.items())[:6]:
        bar = "█" * int(v * 20)
        print(f"    {k:<12}: {v:.4f} {bar}")
