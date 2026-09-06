"""
PPO Training Loop
==================
Full training pipeline for the PPO portfolio agent.

Features:
    - Rollout collection + PPO update cycle
    - TensorBoard logging (losses, rewards, portfolio metrics)
    - Model checkpointing (best + periodic)
    - Early stopping on Sharpe ratio
    - Train / validation split (walk-forward)
    - Progress logging to console
    - Configurable via TrainConfig dataclass
"""

import os
import sys
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.trading_env import TradingEnv, EnvConfig, generate_market_data
from agents.ppo_agent import PPOAgent, PPOConfig, RolloutBuffer


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""
    # Data
    n_assets:          int   = 5
    n_days_train:      int   = 756       # 3 years train
    n_days_val:        int   = 252       # 1 year val
    seed:              int   = 42

    # Training
    total_timesteps:   int   = 50_000
    rollout_len:       int   = 128
    eval_freq:         int   = 2_000     # evaluate every N timesteps
    save_freq:         int   = 10_000    # checkpoint every N timesteps
    log_freq:          int   = 500

    # Early stopping
    patience:          int   = 10        # eval rounds without improvement
    min_delta:         float = 0.01      # min Sharpe improvement

    # Paths
    log_dir:           str   = "runs"
    model_dir:         str   = "models"
    exp_name:          str   = "ppo_portfolio"


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent: PPOAgent,
    env: TradingEnv,
    n_episodes: int = 3,
    deterministic: bool = True,
) -> dict:
    """
    Run n_episodes with the agent and return aggregated metrics.

    Returns dict with mean sharpe, total_return, max_drawdown, final_value.
    """
    all_sharpes, all_returns, all_dds, all_values = [], [], [], []

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False

        while not done:
            action, _, _ = agent.act(obs, deterministic=deterministic)
            obs, _, done, _ = env.step(action)

        summary = env.portfolio_metrics()
        all_sharpes.append(summary.get("sharpe",      0.0))
        all_returns.append(summary.get("total_return",  0.0))
        all_dds.append(    summary.get("max_drawdown",  0.0))
        all_values.append( summary.get("final_value", 100000.0))

    return {
        "sharpe":       round(float(np.mean(all_sharpes)), 4),
        "total_return": round(float(np.mean(all_returns)), 4),
        "max_drawdown": round(float(np.mean(all_dds)),    4),
        "final_value":  round(float(np.mean(all_values)), 2),
        "n_episodes":   n_episodes,
    }


# ---------------------------------------------------------------------------
# TensorBoard logger (with graceful fallback)
# ---------------------------------------------------------------------------

class Logger:
    """
    Logs scalars to TensorBoard and console.
    Falls back to console-only if tensorboard unavailable.
    """

    def __init__(self, log_dir: str, exp_name: str):
        self.log_dir  = Path(log_dir) / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer   = None
        self.history  = []   # (step, key, value) list for plotting

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.log_dir))
            print(f"  TensorBoard logging → {self.log_dir}")
        except Exception:
            print("  TensorBoard unavailable — console logging only.")

    def log(self, step: int, metrics: dict, prefix: str = ""):
        for key, val in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            if self.writer:
                self.writer.add_scalar(tag, val, step)
            self.history.append((step, tag, val))

    def close(self):
        if self.writer:
            self.writer.close()

    def get_history(self) -> list:
        return self.history


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Saves best and periodic model checkpoints."""

    def __init__(self, model_dir: str, exp_name: str):
        self.dir = Path(model_dir) / exp_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best_sharpe = -np.inf

    def save_periodic(self, agent: PPOAgent, step: int):
        path = str(self.dir / f"step_{step:07d}.pt")
        agent.save(path)

    def save_best(self, agent: PPOAgent, sharpe: float) -> bool:
        """Save if this is the best Sharpe seen. Returns True if saved."""
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            agent.save(str(self.dir / "best.pt"))
            return True
        return False

    def load_best(self, agent: PPOAgent) -> bool:
        """Load best checkpoint if it exists. Returns True if loaded."""
        path = self.dir / "best.pt"
        if path.exists():
            agent.load(str(path))
            return True
        return False


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

class Trainer:
    """
    Orchestrates the full PPO training pipeline.

    Usage:
        trainer = Trainer(train_cfg, ppo_cfg, env_cfg)
        result  = trainer.train()
    """

    def __init__(
        self,
        train_cfg:  TrainConfig = None,
        ppo_cfg:    PPOConfig   = None,
        env_cfg:    EnvConfig   = None,
    ):
        self.train_cfg = train_cfg or TrainConfig()
        self.ppo_cfg   = ppo_cfg   or PPOConfig()
        self.env_cfg   = env_cfg   or EnvConfig(n_assets=self.train_cfg.n_assets)

        # Sync rollout_len
        self.ppo_cfg.rollout_len = self.train_cfg.rollout_len

        self._setup()

    def _setup(self):
        """Build environments, agent, logger, checkpoint manager."""
        tc = self.train_cfg

        # Generate data
        n_total = tc.n_days_train + tc.n_days_val
        all_data = generate_market_data(
            n_assets=tc.n_assets,
            n_days=n_total,
            seed=tc.seed,
        )
        train_data = all_data.iloc[:tc.n_days_train]
        val_data   = all_data.iloc[tc.n_days_train:]

        # Environments
        self.train_env = TradingEnv(prices=train_data, config=self.env_cfg)
        self.val_env   = TradingEnv(prices=val_data,   config=self.env_cfg)

        # Agent
        obs        = self.train_env.reset()
        state_dim  = obs.shape[0]
        action_dim = tc.n_assets + 1
        self.agent = PPOAgent(state_dim, action_dim, self.ppo_cfg)

        # Logger + checkpointing
        self.logger  = Logger(tc.log_dir, tc.exp_name)
        self.ckpt    = CheckpointManager(tc.model_dir, tc.exp_name)

        print(f"  State dim  : {state_dim}")
        print(f"  Action dim : {action_dim}")
        print(f"  Train days : {len(train_data)}")
        print(f"  Val days   : {len(val_data)}")
        print(f"  Params     : {self.agent.param_count()}")

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns training history dict.
        """
        tc      = self.train_cfg
        agent   = self.agent
        env     = self.train_env

        obs      = env.reset()
        done     = False
        t_start  = time.time()

        total_steps    = 0
        ep_count       = 0
        no_improve     = 0
        best_sharpe    = -np.inf

        ep_rewards     = []
        all_ep_rewards = []
        train_metrics  = []
        eval_metrics   = []

        print(f"\n  Starting training — {tc.total_timesteps:,} timesteps")
        print("  " + "─" * 50)

        while total_steps < tc.total_timesteps:

            # ── Collect rollout ──────────────────────────────────────────
            buf = RolloutBuffer(tc.rollout_len, agent.state_dim, agent.action_dim)
            rollout_reward = 0.0

            for _ in range(tc.rollout_len):
                action, log_prob, value = agent.act(obs)
                next_obs, reward, done, info = env.step(action)

                buf.add(obs, action, reward, value, log_prob, done)
                ep_rewards.append(reward)
                rollout_reward += reward
                total_steps    += 1
                obs             = next_obs

                if done:
                    all_ep_rewards.append(sum(ep_rewards))
                    ep_rewards = []
                    ep_count  += 1
                    obs  = env.reset()
                    done = False

                if total_steps >= tc.total_timesteps:
                    break

            # ── PPO update ───────────────────────────────────────────────
            _, _, last_val = agent.act(obs)
            update_metrics = agent.update(buf, last_val)
            train_metrics.append({**update_metrics, "step": total_steps})

            # ── Logging ──────────────────────────────────────────────────
            if total_steps % tc.log_freq < tc.rollout_len:
                self.logger.log(total_steps, update_metrics, "train")
                self.logger.log(total_steps, {
                    "rollout_reward": rollout_reward,
                    "episodes":       ep_count,
                }, "rollout")

                elapsed = time.time() - t_start
                fps     = total_steps / max(elapsed, 1)
                print(f"  [{total_steps:>7,}/{tc.total_timesteps:,}] "
                      f"actor={update_metrics['actor_loss']:+.4f}  "
                      f"critic={update_metrics['critic_loss']:.4f}  "
                      f"entropy={update_metrics['entropy']:.3f}  "
                      f"fps={fps:.0f}")

            # ── Evaluation ───────────────────────────────────────────────
            if total_steps % tc.eval_freq < tc.rollout_len:
                eval_result = evaluate_agent(agent, self.val_env, n_episodes=2)
                eval_metrics.append({**eval_result, "step": total_steps})
                self.logger.log(total_steps, eval_result, "eval")

                sharpe = eval_result["sharpe"]
                improved = self.ckpt.save_best(agent, sharpe)

                if improved:
                    best_sharpe = sharpe
                    no_improve  = 0
                    tag = " ← best"
                else:
                    no_improve += 1
                    tag = ""

                print(f"  [EVAL step={total_steps:,}] "
                      f"Sharpe={sharpe:.3f}  "
                      f"Return={eval_result['total_return']:.2f}%  "
                      f"MDD={eval_result['max_drawdown']:.2f}%{tag}")

                # Early stopping
                if no_improve >= tc.patience:
                    print(f"\n  Early stopping: no improvement for {tc.patience} evals.")
                    break

            # ── Periodic checkpoint ──────────────────────────────────────
            if total_steps % tc.save_freq < tc.rollout_len:
                self.ckpt.save_periodic(agent, total_steps)

        elapsed = time.time() - t_start
        print(f"\n  Training complete — {total_steps:,} steps in {elapsed:.1f}s")
        print(f"  Best val Sharpe : {best_sharpe:.4f}")

        self.logger.close()

        return {
            "total_steps":    total_steps,
            "episodes":       ep_count,
            "best_sharpe":    best_sharpe,
            "train_metrics":  train_metrics,
            "eval_metrics":   eval_metrics,
            "elapsed_s":      round(elapsed, 2),
            "log_history":    self.logger.get_history(),
        }


# ---------------------------------------------------------------------------
# Quick training run (short, for demo/testing)
# ---------------------------------------------------------------------------

def quick_train(
    n_assets: int = 5,
    total_timesteps: int = 2_000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run a short training session for demo / testing purposes.
    Returns training result dict.
    """
    train_cfg = TrainConfig(
        n_assets=n_assets,
        n_days_train=504,
        n_days_val=126,
        total_timesteps=total_timesteps,
        rollout_len=64,
        eval_freq=500,
        save_freq=1_000,
        log_freq=500,
        patience=5,
        seed=seed,
        exp_name=f"quick_run_{seed}",
    )
    ppo_cfg = PPOConfig(
        hidden_dims=[64, 32],
        n_epochs=3,
        rollout_len=64,
        dist_type="dirichlet",
    )
    env_cfg = EnvConfig(n_assets=n_assets, reward_type="sharpe_dd")

    if not verbose:
        import io, contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            trainer = Trainer(train_cfg, ppo_cfg, env_cfg)
            result  = trainer.train()
    else:
        trainer = Trainer(train_cfg, ppo_cfg, env_cfg)
        result  = trainer.train()

    return result, trainer


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PPO Training Loop Demo")
    print("=" * 60)

    result, trainer = quick_train(
        n_assets=5,
        total_timesteps=3_000,
        verbose=True,
    )

    print(f"\n  Summary:")
    print(f"    Total steps  : {result['total_steps']:,}")
    print(f"    Episodes     : {result['episodes']}")
    print(f"    Best Sharpe  : {result['best_sharpe']:.4f}")
    print(f"    Elapsed      : {result['elapsed_s']:.1f}s")

    if result["eval_metrics"]:
        print(f"\n  Eval history:")
        for m in result["eval_metrics"]:
            print(f"    step={m['step']:,}  sharpe={m['sharpe']:.3f}  "
                  f"return={m['total_return']:.2f}%")
