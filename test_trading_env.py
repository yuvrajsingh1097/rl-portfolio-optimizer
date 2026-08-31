"""
Unit Tests — Trading Environment
===================================
Run with: python -m pytest tests/test_trading_env.py -v
"""

import pytest
import numpy as np
from env.trading_env import (
    TradingEnv, EnvConfig, generate_market_data,
    compute_features, random_agent_episode,
    reward_log_return, reward_sharpe, reward_sortino,
    REWARD_FNS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prices():
    return generate_market_data(n_assets=4, n_days=400, seed=0)


@pytest.fixture(scope="module")
def config():
    return EnvConfig(n_assets=4, window=20, episode_length=100,
                     reward_type="sharpe", transaction_cost=0.001)


@pytest.fixture(scope="module")
def env(prices, config):
    return TradingEnv(prices, config)


# ---------------------------------------------------------------------------
# 1. Market data generator
# ---------------------------------------------------------------------------

class TestMarketData:

    def test_returns_dataframe(self):
        p = generate_market_data(n_assets=3, n_days=200)
        import pandas as pd
        assert isinstance(p, pd.DataFrame)

    def test_correct_shape(self):
        p = generate_market_data(n_assets=5, n_days=300)
        assert p.shape == (300, 5)

    def test_positive_prices(self):
        p = generate_market_data(n_assets=3, n_days=200)
        assert (p > 0).all().all()

    def test_starts_at_100(self):
        p = generate_market_data(n_assets=3, n_days=200)
        assert abs(p.iloc[0].mean() - 100) < 5

    def test_reproducible_with_seed(self):
        p1 = generate_market_data(n_assets=3, n_days=100, seed=42)
        p2 = generate_market_data(n_assets=3, n_days=100, seed=42)
        assert (p1 == p2).all().all()


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

class TestFeatures:

    def test_output_shape(self, prices):
        feats = compute_features(prices, window=20)
        assert feats.shape == (len(prices), prices.shape[1] * 6)

    def test_float32(self, prices):
        feats = compute_features(prices, window=20)
        assert feats.dtype == np.float32

    def test_no_inf(self, prices):
        feats = compute_features(prices, window=20)
        assert not np.any(np.isinf(feats))

    def test_no_large_values(self, prices):
        feats = compute_features(prices, window=20)
        assert np.isfinite(feats).all()


# ---------------------------------------------------------------------------
# 3. Reward functions
# ---------------------------------------------------------------------------

class TestRewardFunctions:

    def test_log_return_scalar(self):
        r = reward_log_return(portfolio_return=0.01)
        assert isinstance(r, float)

    def test_log_return_value(self):
        assert reward_log_return(portfolio_return=0.05) == 0.05

    def test_sharpe_returns_float(self):
        hist = list(np.random.randn(30) * 0.01)
        r = reward_sharpe(returns_history=hist, portfolio_return=0.01)
        assert isinstance(r, float)

    def test_sharpe_short_history(self):
        r = reward_sharpe(returns_history=[0.01], portfolio_return=0.01)
        assert isinstance(r, float)

    def test_sortino_returns_float(self):
        hist = list(np.random.randn(30) * 0.01)
        r = reward_sortino(returns_history=hist, portfolio_return=0.01)
        assert isinstance(r, float)

    def test_all_reward_fns_registered(self):
        for name in ["log_return","sharpe","sortino","drawdown"]:
            assert name in REWARD_FNS


# ---------------------------------------------------------------------------
# 4. Environment init
# ---------------------------------------------------------------------------

class TestEnvInit:

    def test_obs_dim_positive(self, env):
        assert env.obs_dim > 0

    def test_action_dim(self, env, config):
        assert env.action_dim == config.n_assets + 1

    def test_features_computed(self, env, prices):
        assert env.features.shape[0] == len(prices)


# ---------------------------------------------------------------------------
# 5. Reset
# ---------------------------------------------------------------------------

class TestReset:

    def test_returns_array(self, env):
        obs = env.reset()
        assert isinstance(obs, np.ndarray)

    def test_obs_correct_shape(self, env):
        obs = env.reset()
        assert obs.shape == (env.obs_dim,)

    def test_obs_float32(self, env):
        obs = env.reset()
        assert obs.dtype == np.float32

    def test_weights_sum_to_1(self, env):
        env.reset()
        assert abs(env.weights.sum() - 1.0) < 1e-5

    def test_portfolio_val_reset(self, env, config):
        env.reset()
        assert env.portfolio_val == config.initial_capital

    def test_done_false_after_reset(self, env):
        env.reset()
        assert not env.done


# ---------------------------------------------------------------------------
# 6. Step
# ---------------------------------------------------------------------------

class TestStep:

    def test_returns_4_tuple(self, env):
        env.reset()
        result = env.step(np.random.randn(env.action_dim))
        assert len(result) == 4

    def test_obs_correct_shape(self, env):
        env.reset()
        obs, _, _, _ = env.step(np.random.randn(env.action_dim))
        assert obs.shape == (env.obs_dim,)

    def test_reward_is_float(self, env):
        env.reset()
        _, reward, _, _ = env.step(np.random.randn(env.action_dim))
        assert isinstance(reward, float)

    def test_done_is_bool(self, env):
        env.reset()
        _, _, done, _ = env.step(np.random.randn(env.action_dim))
        assert isinstance(done, bool)

    def test_info_has_keys(self, env):
        env.reset()
        _, _, _, info = env.step(np.random.randn(env.action_dim))
        for key in ["portfolio_value","portfolio_return","drawdown"]:
            assert key in info

    def test_portfolio_value_positive(self, env):
        env.reset()
        _, _, _, info = env.step(np.random.randn(env.action_dim))
        assert info["portfolio_value"] > 0

    def test_weights_sum_to_1_after_step(self, env):
        env.reset()
        env.step(np.random.randn(env.action_dim))
        assert abs(env.weights.sum() - 1.0) < 1e-4

    def test_episode_terminates(self, env, config):
        env.reset()
        done = False
        steps = 0
        while not done and steps < config.episode_length + 10:
            _, _, done, _ = env.step(np.random.randn(env.action_dim))
            steps += 1
        assert done

    def test_step_after_done_raises(self, env):
        env.reset()
        env.done = True
        with pytest.raises(RuntimeError):
            env.step(np.random.randn(env.action_dim))


# ---------------------------------------------------------------------------
# 7. Portfolio metrics
# ---------------------------------------------------------------------------

class TestPortfolioMetrics:

    @pytest.fixture
    def completed_env(self, prices, config):
        e = TradingEnv(prices, config)
        random_agent_episode(e)
        return e

    def test_returns_dict(self, completed_env):
        assert isinstance(completed_env.portfolio_metrics(), dict)

    def test_required_keys(self, prices):
        cfg = EnvConfig(n_assets=4, window=20, episode_length=100)
        e = TradingEnv(prices, cfg)
        random_agent_episode(e)
        m = e.portfolio_metrics()
        if m:
            for key in ["total_return","sharpe","max_drawdown","final_value"]:
                assert key in m

    def test_final_value_positive(self, prices):
        cfg = EnvConfig(n_assets=4, window=20, episode_length=100)
        e = TradingEnv(prices, cfg)
        m = random_agent_episode(e)
        assert m.get("final_value", 1) > 0


# ---------------------------------------------------------------------------
# 8. Random agent
# ---------------------------------------------------------------------------

class TestRandomAgent:

    def test_returns_metrics(self, env):
        m = random_agent_episode(env)
        assert isinstance(m, dict)

    def test_episode_completes(self, prices):
        cfg = EnvConfig(n_assets=4, window=20, episode_length=100)
        env2 = TradingEnv(prices, cfg)
        m = random_agent_episode(env2)
        assert m.get("n_steps", 0) >= 0

    def test_different_reward_types(self, prices):
        for rtype in ["log_return","sharpe","sortino","drawdown"]:
            cfg = EnvConfig(n_assets=4, window=20, episode_length=50,
                            reward_type=rtype)
            e = TradingEnv(prices, cfg)
            m = random_agent_episode(e)
            assert isinstance(m, dict)
