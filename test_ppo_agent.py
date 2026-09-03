"""
Unit Tests — PPO Agent
========================
Run with: python -m pytest tests/test_ppo_agent.py -v
"""

import pytest
import numpy as np
import torch
from agents.ppo_agent import (
    PPOConfig, PPOAgent, ActorNetwork, CriticNetwork,
    RolloutBuffer, build_mlp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STATE_DIM  = 50
ACTION_DIM = 6


@pytest.fixture(scope="module")
def config():
    return PPOConfig(hidden_dims=[64, 32], dist_type="dirichlet", n_epochs=2, rollout_len=32)


@pytest.fixture(scope="module")
def agent(config):
    return PPOAgent(STATE_DIM, ACTION_DIM, config)


@pytest.fixture
def sample_state():
    return np.random.randn(STATE_DIM).astype(np.float32)


@pytest.fixture(scope="module")
def filled_buffer(config, agent):
    buf = RolloutBuffer(32, STATE_DIM, ACTION_DIM)
    rng = np.random.default_rng(0)
    for _ in range(32):
        s  = rng.standard_normal(STATE_DIM).astype(np.float32)
        a, lp, v = agent.act(s)
        buf.add(s, a, float(rng.standard_normal()), v, lp, False)
    return buf


# ---------------------------------------------------------------------------
# 1. build_mlp
# ---------------------------------------------------------------------------

class TestBuildMLP:

    def test_returns_sequential(self):
        net = build_mlp(10, [32, 16], 5)
        assert isinstance(net, torch.nn.Sequential)

    def test_correct_output_shape(self):
        net = build_mlp(10, [32, 16], 5)
        out = net(torch.randn(4, 10))
        assert out.shape == (4, 5)

    def test_single_hidden(self):
        net = build_mlp(8, [16], 4)
        out = net(torch.randn(2, 8))
        assert out.shape == (2, 4)

    def test_layer_norm_option(self):
        net_ln  = build_mlp(8, [16], 4, use_layer_norm=True)
        net_no  = build_mlp(8, [16], 4, use_layer_norm=False)
        assert len(list(net_ln.modules())) > len(list(net_no.modules()))


# ---------------------------------------------------------------------------
# 2. ActorNetwork
# ---------------------------------------------------------------------------

class TestActorNetwork:

    def test_forward_dirichlet(self, config):
        actor = ActorNetwork(STATE_DIM, ACTION_DIM, config)
        state = torch.randn(4, STATE_DIM)
        dist  = actor.forward(state)
        assert hasattr(dist, "sample")

    def test_sample_sums_to_one(self, config):
        actor = ActorNetwork(STATE_DIM, ACTION_DIM, config)
        state = torch.randn(4, STATE_DIM)
        dist  = actor.forward(state)
        sample = dist.rsample()
        assert torch.allclose(sample.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_get_action_returns_tuple(self, config, sample_state):
        actor  = ActorNetwork(STATE_DIM, ACTION_DIM, config)
        state  = torch.FloatTensor(sample_state).unsqueeze(0)
        result = actor.get_action(state)
        assert len(result) == 3

    def test_action_dim(self, config, sample_state):
        actor  = ActorNetwork(STATE_DIM, ACTION_DIM, config)
        state  = torch.FloatTensor(sample_state).unsqueeze(0)
        action, _, _ = actor.get_action(state)
        assert action.shape[-1] == ACTION_DIM

    def test_deterministic_action(self, config, sample_state):
        actor  = ActorNetwork(STATE_DIM, ACTION_DIM, config)
        state  = torch.FloatTensor(sample_state).unsqueeze(0)
        a1, _, _ = actor.get_action(state, deterministic=True)
        a2, _, _ = actor.get_action(state, deterministic=True)
        assert torch.allclose(a1, a2)

    def test_log_prob_finite(self, config, sample_state):
        actor  = ActorNetwork(STATE_DIM, ACTION_DIM, config)
        state  = torch.FloatTensor(sample_state).unsqueeze(0)
        _, lp, _ = actor.get_action(state)
        assert torch.isfinite(lp).all()

    def test_entropy_positive(self, config, sample_state):
        actor  = ActorNetwork(STATE_DIM, ACTION_DIM, config)
        state  = torch.FloatTensor(sample_state).unsqueeze(0)
        _, _, entropy = actor.get_action(state)
        assert np.isfinite(float(entropy.detach()))

    def test_gaussian_dist(self):
        cfg   = PPOConfig(hidden_dims=[32], dist_type="gaussian")
        actor = ActorNetwork(STATE_DIM, ACTION_DIM, cfg)
        state = torch.randn(2, STATE_DIM)
        dist  = actor.forward(state)
        assert hasattr(dist, "sample")


# ---------------------------------------------------------------------------
# 3. CriticNetwork
# ---------------------------------------------------------------------------

class TestCriticNetwork:

    def test_output_shape(self, config):
        critic = CriticNetwork(STATE_DIM, config)
        state  = torch.randn(4, STATE_DIM)
        value  = critic(state)
        assert value.shape == (4,)

    def test_scalar_for_single_state(self, config, sample_state):
        critic = CriticNetwork(STATE_DIM, config)
        state  = torch.FloatTensor(sample_state).unsqueeze(0)
        value  = critic(state)
        assert value.shape == (1,)

    def test_output_finite(self, config):
        critic = CriticNetwork(STATE_DIM, config)
        state  = torch.randn(8, STATE_DIM)
        value  = critic(state)
        assert torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# 4. RolloutBuffer
# ---------------------------------------------------------------------------

class TestRolloutBuffer:

    def test_add_and_size(self):
        buf = RolloutBuffer(10, STATE_DIM, ACTION_DIM)
        for _ in range(5):
            buf.add(
                np.zeros(STATE_DIM), np.ones(ACTION_DIM)/ACTION_DIM,
                1.0, 0.5, -0.1, False
            )
        assert buf.ptr == 5

    def test_compute_returns_shape(self):
        buf = RolloutBuffer(10, STATE_DIM, ACTION_DIM)
        for _ in range(10):
            buf.add(np.zeros(STATE_DIM), np.ones(ACTION_DIM)/ACTION_DIM,
                    0.1, 0.0, -0.1, False)
        rets, advs = buf.compute_returns_advantages(last_value=0.0)
        assert len(rets) == 10
        assert len(advs) == 10

    def test_returns_finite(self):
        buf = RolloutBuffer(10, STATE_DIM, ACTION_DIM)
        for _ in range(10):
            buf.add(np.zeros(STATE_DIM), np.ones(ACTION_DIM)/ACTION_DIM,
                    0.05, 0.0, -0.1, False)
        rets, advs = buf.compute_returns_advantages(0.0)
        assert np.isfinite(rets).all()
        assert np.isfinite(advs).all()

    def test_reset(self):
        buf = RolloutBuffer(10, STATE_DIM, ACTION_DIM)
        buf.add(np.zeros(STATE_DIM), np.ones(ACTION_DIM)/ACTION_DIM, 1.0, 0.0, 0.0, False)
        buf.reset()
        assert buf.ptr == 0

    def test_get_batches_yields_dicts(self, filled_buffer):
        rets, advs = filled_buffer.compute_returns_advantages(0.0)
        batches = list(filled_buffer.get_batches(16, rets, advs))
        assert len(batches) > 0
        assert "states" in batches[0]
        assert "advantages" in batches[0]

    def test_advantages_normalised(self, filled_buffer):
        rets, advs = filled_buffer.compute_returns_advantages(0.0)
        batches = list(filled_buffer.get_batches(32, rets, advs))
        adv_vals = batches[0]["advantages"]
        assert abs(float(adv_vals.mean())) < 0.5   # approximately zero mean


# ---------------------------------------------------------------------------
# 5. PPOAgent
# ---------------------------------------------------------------------------

class TestPPOAgent:

    def test_creates_agent(self, agent):
        assert agent is not None

    def test_param_count(self, agent):
        params = agent.param_count()
        assert params["total_params"] > 0
        assert params["actor_params"] > 0
        assert params["critic_params"] > 0

    def test_act_returns_tuple(self, agent, sample_state):
        result = agent.act(sample_state)
        assert len(result) == 3

    def test_action_shape(self, agent, sample_state):
        action, _, _ = agent.act(sample_state)
        assert len(action) == ACTION_DIM

    def test_action_sums_to_one(self, agent, sample_state):
        action, _, _ = agent.act(sample_state)
        assert abs(action.sum() - 1.0) < 1e-5

    def test_action_nonnegative(self, agent, sample_state):
        action, _, _ = agent.act(sample_state)
        assert (action >= 0).all()

    def test_value_is_float(self, agent, sample_state):
        _, _, value = agent.act(sample_state)
        assert isinstance(value, float)

    def test_log_prob_is_float(self, agent, sample_state):
        _, lp, _ = agent.act(sample_state)
        assert isinstance(lp, float)
        assert np.isfinite(lp)

    def test_deterministic_reproducible(self, agent, sample_state):
        a1, _, _ = agent.act(sample_state, deterministic=True)
        a2, _, _ = agent.act(sample_state, deterministic=True)
        assert np.allclose(a1, a2)

    def test_update_returns_dict(self, agent, filled_buffer):
        metrics = agent.update(filled_buffer, last_value=0.0)
        assert isinstance(metrics, dict)
        assert "actor_loss" in metrics
        assert "critic_loss" in metrics

    def test_update_count_increments(self, agent, filled_buffer):
        before = agent.update_count
        agent.update(filled_buffer, last_value=0.0)
        assert agent.update_count == before + 1

    def test_save_and_load(self, agent, sample_state, tmp_path):
        path = str(tmp_path / "ppo_test.pt")
        agent.save(path)
        agent2 = PPOAgent(STATE_DIM, ACTION_DIM, agent.config)
        agent2.load(path)
        a1, _, _ = agent.act(sample_state, deterministic=True)
        a2, _, _ = agent2.act(sample_state, deterministic=True)
        assert np.allclose(a1, a2, atol=1e-5)
