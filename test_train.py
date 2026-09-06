"""
Unit Tests — Training Loop
============================
Run with: python -m pytest tests/test_train.py -v
"""

import pytest
import numpy as np
from train.train import (
    TrainConfig, Trainer, Logger, CheckpointManager,
    evaluate_agent, quick_train,
)
from agents.ppo_agent import PPOAgent, PPOConfig
from env.trading_env import TradingEnv, EnvConfig, generate_market_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def quick_result():
    result, trainer = quick_train(n_assets=3, total_timesteps=512, verbose=False)
    return result, trainer


@pytest.fixture(scope="module")
def small_env():
    data = generate_market_data(n_assets=3, n_days=252, seed=0)
    cfg  = EnvConfig(n_assets=3)
    return TradingEnv(prices=data, config=cfg)


@pytest.fixture(scope="module")
def small_agent(small_env):
    obs   = small_env.reset()
    cfg   = PPOConfig(hidden_dims=[32, 16], n_epochs=2)
    return PPOAgent(obs.shape[0], 4, cfg)


# ---------------------------------------------------------------------------
# 1. TrainConfig
# ---------------------------------------------------------------------------

class TestTrainConfig:

    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.n_assets > 0
        assert cfg.total_timesteps > 0

    def test_custom_values(self):
        cfg = TrainConfig(n_assets=3, total_timesteps=1000)
        assert cfg.n_assets == 3
        assert cfg.total_timesteps == 1000


# ---------------------------------------------------------------------------
# 2. Logger
# ---------------------------------------------------------------------------

class TestLogger:

    def test_creates_without_tensorboard(self, tmp_path):
        logger = Logger(str(tmp_path), "test_run")
        assert logger is not None

    def test_log_stores_history(self, tmp_path):
        logger = Logger(str(tmp_path), "test_run2")
        logger.log(0, {"loss": 0.5, "reward": 1.0})
        assert len(logger.get_history()) == 2

    def test_log_with_prefix(self, tmp_path):
        logger = Logger(str(tmp_path), "test_run3")
        logger.log(10, {"sharpe": 0.8}, prefix="eval")
        history = logger.get_history()
        assert any("eval" in tag for _, tag, _ in history)

    def test_close_no_error(self, tmp_path):
        logger = Logger(str(tmp_path), "test_close")
        logger.close()


# ---------------------------------------------------------------------------
# 3. CheckpointManager
# ---------------------------------------------------------------------------

class TestCheckpointManager:

    def test_save_best(self, tmp_path, small_agent):
        cm = CheckpointManager(str(tmp_path), "test_ckpt")
        saved = cm.save_best(small_agent, sharpe=0.5)
        assert saved is True

    def test_save_best_only_if_better(self, tmp_path, small_agent):
        cm = CheckpointManager(str(tmp_path), "test_ckpt2")
        cm.save_best(small_agent, sharpe=0.5)
        saved = cm.save_best(small_agent, sharpe=0.3)
        assert saved is False

    def test_load_best(self, tmp_path, small_agent, small_env):
        cm = CheckpointManager(str(tmp_path), "test_load")
        cm.save_best(small_agent, sharpe=1.0)
        agent2 = PPOAgent(small_agent.state_dim, small_agent.action_dim, small_agent.config)
        loaded = cm.load_best(agent2)
        assert loaded is True

    def test_load_best_no_file(self, tmp_path, small_agent):
        cm = CheckpointManager(str(tmp_path), "test_empty")
        agent2 = PPOAgent(small_agent.state_dim, small_agent.action_dim, small_agent.config)
        loaded = cm.load_best(agent2)
        assert loaded is False

    def test_save_periodic(self, tmp_path, small_agent):
        cm = CheckpointManager(str(tmp_path), "test_periodic")
        cm.save_periodic(small_agent, step=1000)
        assert (tmp_path / "test_periodic" / "step_0001000.pt").exists()


# ---------------------------------------------------------------------------
# 4. evaluate_agent
# ---------------------------------------------------------------------------

class TestEvaluateAgent:

    def test_returns_dict(self, small_agent, small_env):
        result = evaluate_agent(small_agent, small_env, n_episodes=1)
        assert isinstance(result, dict)

    def test_has_sharpe(self, small_agent, small_env):
        result = evaluate_agent(small_agent, small_env, n_episodes=1)
        assert "sharpe" in result

    def test_has_total_return(self, small_agent, small_env):
        result = evaluate_agent(small_agent, small_env, n_episodes=1)
        assert "total_return" in result

    def test_n_episodes(self, small_agent, small_env):
        result = evaluate_agent(small_agent, small_env, n_episodes=2)
        assert result["n_episodes"] == 2

    def test_final_value_positive(self, small_agent, small_env):
        result = evaluate_agent(small_agent, small_env, n_episodes=1)
        assert result["final_value"] > 0


# ---------------------------------------------------------------------------
# 5. Trainer
# ---------------------------------------------------------------------------

class TestTrainer:

    def test_quick_train_returns_dict(self, quick_result):
        result, _ = quick_result
        assert isinstance(result, dict)

    def test_total_steps(self, quick_result):
        result, _ = quick_result
        assert result["total_steps"] > 0

    def test_episodes_run(self, quick_result):
        result, _ = quick_result
        assert result["episodes"] >= 0

    def test_train_metrics_populated(self, quick_result):
        result, _ = quick_result
        assert len(result["train_metrics"]) > 0

    def test_train_metrics_keys(self, quick_result):
        result, _ = quick_result
        m = result["train_metrics"][0]
        assert "actor_loss" in m
        assert "critic_loss" in m

    def test_eval_metrics_populated(self, quick_result):
        result, _ = quick_result
        assert len(result["eval_metrics"]) >= 0

    def test_elapsed_positive(self, quick_result):
        result, _ = quick_result
        assert result["elapsed_s"] > 0

    def test_trainer_has_agent(self, quick_result):
        _, trainer = quick_result
        assert trainer.agent is not None

    def test_trainer_has_envs(self, quick_result):
        _, trainer = quick_result
        assert trainer.train_env is not None
        assert trainer.val_env   is not None

    def test_log_history_populated(self, quick_result):
        result, _ = quick_result
        assert isinstance(result["log_history"], list)
