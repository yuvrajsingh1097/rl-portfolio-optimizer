"""
Unit Tests — SAC Agent & Feature Importance
Run with: python -m pytest tests/test_sac_agent.py -v
"""
import pytest
import numpy as np
from agents.sac_agent import (
    SACConfig, SACAgent, NumpyMLP, ReplayBuffer,
    permutation_importance,
)
from env.trading_env import TradingEnv, EnvConfig, generate_market_data

STATE_DIM  = 22
ACTION_DIM = 6


@pytest.fixture(scope="module")
def cfg():
    return SACConfig(hidden_dims=[32, 16], learning_starts=20, batch_size=16)


@pytest.fixture(scope="module")
def agent(cfg):
    return SACAgent(STATE_DIM, ACTION_DIM, cfg, seed=0)


@pytest.fixture(scope="module")
def trained_agent(cfg):
    a   = SACAgent(STATE_DIM, ACTION_DIM, cfg, seed=1)
    rng = np.random.default_rng(42)
    for _ in range(100):
        s  = rng.standard_normal(STATE_DIM).astype(np.float32)
        ac = rng.dirichlet(np.ones(ACTION_DIM)).astype(np.float32)
        s_ = rng.standard_normal(STATE_DIM).astype(np.float32)
        a.store(s, ac, float(rng.standard_normal()), s_, False)
    for _ in range(50):
        a.update()
    return a


@pytest.fixture(scope="module")
def small_env():
    d = generate_market_data(n_assets=5, n_days=252, seed=0)
    return TradingEnv(prices=d, config=EnvConfig(n_assets=5))


# ---------------------------------------------------------------------------
# 1. NumpyMLP
# ---------------------------------------------------------------------------
class TestNumpyMLP:

    def test_forward_shape(self):
        net = NumpyMLP([10, 32, 16, 4])
        out = net.forward(np.ones((1, 10), dtype=np.float32))
        assert out.shape == (1, 4)

    def test_relu_no_negatives(self):
        net = NumpyMLP([4, 8, 4])
        # Intermediate layers use ReLU
        out = net.forward(np.ones((1, 4), dtype=np.float32), activate_last=True)
        assert (out >= 0).all()

    def test_copy_independence(self):
        net  = NumpyMLP([4, 8, 4])
        copy = net.copy()
        copy.weights[0] += 999
        assert not np.allclose(net.weights[0], copy.weights[0])

    def test_polyak_update(self):
        src = NumpyMLP([4, 8, 4], seed=1)
        tgt = NumpyMLP([4, 8, 4], seed=99)
        orig_w = tgt.weights[0].copy()
        tgt.polyak_update(src, tau=0.5)
        assert not np.allclose(tgt.weights[0], orig_w)

    def test_param_count_positive(self):
        net = NumpyMLP([10, 32, 4])
        assert net.param_count() > 0


# ---------------------------------------------------------------------------
# 2. ReplayBuffer
# ---------------------------------------------------------------------------
class TestReplayBuffer:

    def test_add_and_size(self):
        buf = ReplayBuffer(100, STATE_DIM, ACTION_DIM)
        buf.add(np.zeros(STATE_DIM), np.ones(ACTION_DIM)/ACTION_DIM, 1.0, np.zeros(STATE_DIM), False)
        assert len(buf) == 1

    def test_circular_overwrite(self):
        buf = ReplayBuffer(10, STATE_DIM, ACTION_DIM)
        for _ in range(15):
            buf.add(np.zeros(STATE_DIM), np.ones(ACTION_DIM)/ACTION_DIM, 0.0, np.zeros(STATE_DIM), False)
        assert len(buf) == 10

    def test_sample_shape(self):
        buf = ReplayBuffer(100, STATE_DIM, ACTION_DIM)
        for _ in range(50):
            buf.add(np.zeros(STATE_DIM), np.ones(ACTION_DIM)/ACTION_DIM, 0.0, np.zeros(STATE_DIM), False)
        batch = buf.sample(16)
        assert batch["states"].shape == (16, STATE_DIM)
        assert batch["actions"].shape == (16, ACTION_DIM)

    def test_sample_keys(self):
        buf = ReplayBuffer(50, STATE_DIM, ACTION_DIM)
        for _ in range(30):
            buf.add(np.zeros(STATE_DIM), np.ones(ACTION_DIM)/ACTION_DIM, 0.0, np.zeros(STATE_DIM), False)
        batch = buf.sample(8)
        for k in ["states","actions","rewards","next_states","dones"]:
            assert k in batch


# ---------------------------------------------------------------------------
# 3. SACAgent.act
# ---------------------------------------------------------------------------
class TestSACAgentAct:

    def test_returns_tuple(self, agent):
        s = np.zeros(STATE_DIM, dtype=np.float32)
        assert len(agent.act(s)) == 3

    def test_action_shape(self, agent):
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a, _, _ = agent.act(s)
        assert len(a) == ACTION_DIM

    def test_action_sums_to_one(self, agent):
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a, _, _ = agent.act(s)
        assert abs(a.sum() - 1.0) < 1e-5

    def test_action_nonnegative(self, agent):
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a, _, _ = agent.act(s)
        assert (a >= 0).all()

    def test_deterministic_reproducible(self, agent):
        s  = np.ones(STATE_DIM, dtype=np.float32)
        a1, _, _ = agent.act(s, deterministic=True)
        a2, _, _ = agent.act(s, deterministic=True)
        assert np.allclose(a1, a2)

    def test_log_prob_finite(self, agent):
        s = np.random.randn(STATE_DIM).astype(np.float32)
        _, lp, _ = agent.act(s)
        assert np.isfinite(lp)


# ---------------------------------------------------------------------------
# 4. SACAgent.update
# ---------------------------------------------------------------------------
class TestSACAgentUpdate:

    def test_update_returns_empty_before_starts(self, agent):
        m = agent.update()
        assert isinstance(m, dict)

    def test_update_returns_dict_after_filling(self, trained_agent):
        assert isinstance(trained_agent.train_losses, list)
        assert len(trained_agent.train_losses) > 0

    def test_update_keys(self, trained_agent):
        m = trained_agent.train_losses[-1]
        for k in ["actor_loss","critic1_loss","critic2_loss","alpha"]:
            assert k in m

    def test_update_count_increments(self, trained_agent):
        assert trained_agent.update_count > 0

    def test_alpha_positive(self, trained_agent):
        assert trained_agent.alpha > 0


# ---------------------------------------------------------------------------
# 5. Save / load
# ---------------------------------------------------------------------------
class TestSaveLoad:

    def test_save_and_load(self, trained_agent, tmp_path):
        path = str(tmp_path / "sac_test.pkl")
        trained_agent.save(path)
        a2 = SACAgent(STATE_DIM, ACTION_DIM, trained_agent.config)
        a2.load(path)
        s = np.ones(STATE_DIM, dtype=np.float32)
        act1, _, _ = trained_agent.act(s, deterministic=True)
        act2, _, _ = a2.act(s, deterministic=True)
        assert np.allclose(act1, act2, atol=1e-5)

    def test_param_count(self, agent):
        p = agent.param_count()
        assert p["total_params"] > 0
        assert p["actor_params"] > 0


# ---------------------------------------------------------------------------
# 6. Permutation importance
# ---------------------------------------------------------------------------
class TestPermutationImportance:

    def test_returns_dict(self, trained_agent, small_env):
        fnames = [f"feat_{i}" for i in range(10)]
        imp = permutation_importance(trained_agent, small_env, fnames, n_steps=20, n_repeats=2)
        assert isinstance(imp, dict)

    def test_keys_match_features(self, trained_agent, small_env):
        fnames = [f"feat_{i}" for i in range(5)]
        imp = permutation_importance(trained_agent, small_env, fnames, n_steps=20, n_repeats=1)
        for k in fnames:
            assert k in imp

    def test_values_in_range(self, trained_agent, small_env):
        fnames = [f"feat_{i}" for i in range(5)]
        imp = permutation_importance(trained_agent, small_env, fnames, n_steps=20, n_repeats=1)
        for v in imp.values():
            assert 0.0 <= v <= 1.0

    def test_sorted_descending(self, trained_agent, small_env):
        fnames = [f"feat_{i}" for i in range(8)]
        imp = permutation_importance(trained_agent, small_env, fnames, n_steps=20, n_repeats=1)
        vals = list(imp.values())
        assert vals == sorted(vals, reverse=True)
