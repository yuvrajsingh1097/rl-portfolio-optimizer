"""
Unit Tests — Walk-Forward Evaluation
Run with: python -m pytest tests/test_evaluate.py -v
"""
import pytest
import numpy as np
import pandas as pd
from evaluate.evaluate import (
    compute_metrics, rolling_sharpe, monthly_returns_table,
    run_buy_and_hold, run_random_agent, run_momentum, run_equal_sharpe,
    full_comparison, WalkForwardConfig, WalkForwardEvaluator,
)
from env.trading_env import TradingEnv, EnvConfig, generate_market_data


@pytest.fixture(scope="module")
def prices():
    return generate_market_data(n_assets=3, n_days=252, seed=42)


@pytest.fixture(scope="module")
def cfg():
    return EnvConfig(n_assets=3)


@pytest.fixture(scope="module")
def big_data():
    return generate_market_data(n_assets=3, n_days=504, seed=7)


# ---------------------------------------------------------------------------
# 1. compute_metrics
# ---------------------------------------------------------------------------
class TestComputeMetrics:
    def test_returns_dict(self):
        r = np.random.randn(252) * 0.01
        m = compute_metrics(r)
        assert isinstance(m, dict)

    def test_required_keys(self):
        r = np.random.randn(100) * 0.01
        m = compute_metrics(r)
        for k in ["sharpe","sortino","calmar","max_drawdown_pct","win_rate_pct"]:
            assert k in m

    def test_max_dd_nonpositive(self):
        r = np.random.randn(200) * 0.01
        assert compute_metrics(r)["max_drawdown_pct"] <= 0

    def test_win_rate_in_range(self):
        r = np.random.randn(200) * 0.01
        assert 0 <= compute_metrics(r)["win_rate_pct"] <= 100

    def test_empty_returns(self):
        m = compute_metrics(np.array([]))
        assert isinstance(m, dict)

    def test_positive_returns_positive_sharpe(self):
        r = np.ones(252) * 0.002
        assert compute_metrics(r)["sharpe"] > 0

    def test_negative_returns_negative_sharpe(self):
        r = np.ones(252) * -0.002
        assert compute_metrics(r)["sharpe"] < 0


# ---------------------------------------------------------------------------
# 2. rolling_sharpe
# ---------------------------------------------------------------------------
class TestRollingSharpe:
    def test_returns_array(self):
        r = np.random.randn(200) * 0.01
        rs = rolling_sharpe(r, window=63)
        assert isinstance(rs, np.ndarray)
        assert len(rs) == 200

    def test_first_values_nan(self):
        r = np.random.randn(200) * 0.01
        rs = rolling_sharpe(r, window=63)
        assert np.isnan(rs[:63]).all()


# ---------------------------------------------------------------------------
# 3. monthly_returns_table
# ---------------------------------------------------------------------------
class TestMonthlyReturnsTable:
    def test_returns_dataframe(self, prices):
        r = np.random.randn(len(prices)) * 0.01
        t = monthly_returns_table(r, prices.index)
        assert isinstance(t, pd.DataFrame)

    def test_columns_are_months(self, prices):
        r = np.random.randn(len(prices)) * 0.01
        t = monthly_returns_table(r, prices.index)
        months = {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"}
        assert set(t.columns).issubset(months)


# ---------------------------------------------------------------------------
# 4. Benchmark strategies
# ---------------------------------------------------------------------------
class TestBenchmarks:
    def test_buy_and_hold_returns_array(self, prices, cfg):
        r = run_buy_and_hold(prices, cfg)
        assert isinstance(r, np.ndarray)
        assert len(r) > 0

    def test_random_returns_array(self, prices, cfg):
        r = run_random_agent(prices, cfg)
        assert isinstance(r, np.ndarray)

    def test_momentum_returns_array(self, prices, cfg):
        r = run_momentum(prices, cfg)
        assert isinstance(r, np.ndarray)

    def test_inv_vol_returns_array(self, prices, cfg):
        r = run_equal_sharpe(prices, cfg)
        assert isinstance(r, np.ndarray)

    def test_60_40_lower_vol_than_100pct(self, prices, cfg):
        r_full = run_buy_and_hold(prices, cfg, weight_cash=0.0)
        r_6040 = run_buy_and_hold(prices, cfg, weight_cash=0.4)
        assert r_6040.std() <= r_full.std() + 1e-4

    def test_random_different_seeds(self, prices, cfg):
        r1 = run_random_agent(prices, cfg, seed=0)
        r2 = run_random_agent(prices, cfg, seed=99)
        assert not np.allclose(r1, r2)

    def test_buy_hold_same_weights(self, prices, cfg):
        r1 = run_buy_and_hold(prices, cfg)
        r2 = run_buy_and_hold(prices, cfg)
        assert np.allclose(r1, r2)


# ---------------------------------------------------------------------------
# 5. full_comparison
# ---------------------------------------------------------------------------
class TestFullComparison:
    def test_returns_dict(self, prices, cfg):
        comp = full_comparison(prices, cfg)
        assert isinstance(comp, dict)

    def test_has_strategies(self, prices, cfg):
        comp = full_comparison(prices, cfg)
        assert "Buy & Hold" in comp
        assert "Momentum"   in comp

    def test_each_has_metrics(self, prices, cfg):
        comp = full_comparison(prices, cfg)
        for name, res in comp.items():
            assert "metrics" in res
            assert "returns" in res
            assert "equity"  in res

    def test_equity_starts_at_capital(self, prices, cfg):
        comp = full_comparison(prices, cfg)
        for res in comp.values():
            assert abs(res["equity"][0] - 100_000) < 1000

    def test_metrics_sharpe_finite(self, prices, cfg):
        comp = full_comparison(prices, cfg)
        for res in comp.values():
            assert np.isfinite(res["metrics"]["sharpe"])


# ---------------------------------------------------------------------------
# 6. WalkForwardEvaluator
# ---------------------------------------------------------------------------
class TestWalkForward:
    def test_run_returns_list(self, big_data, cfg):
        wf = WalkForwardEvaluator(big_data, cfg, WalkForwardConfig(n_folds=2))
        r  = wf.run(verbose=False)
        assert isinstance(r, list)
        assert len(r) == 2

    def test_fold_has_strategies(self, big_data, cfg):
        wf = WalkForwardEvaluator(big_data, cfg, WalkForwardConfig(n_folds=1))
        r  = wf.run(verbose=False)
        assert "buy_and_hold" in r[0]
        assert "random"       in r[0]

    def test_summary_table_returns_df(self, big_data, cfg):
        wf = WalkForwardEvaluator(big_data, cfg, WalkForwardConfig(n_folds=2))
        wf.run(verbose=False)
        t = wf.summary_table()
        assert isinstance(t, pd.DataFrame)

    def test_summary_table_index(self, big_data, cfg):
        wf = WalkForwardEvaluator(big_data, cfg, WalkForwardConfig(n_folds=1))
        wf.run(verbose=False)
        t = wf.summary_table()
        assert "buy_and_hold" in t.index

    def test_summary_columns(self, big_data, cfg):
        wf = WalkForwardEvaluator(big_data, cfg, WalkForwardConfig(n_folds=1))
        wf.run(verbose=False)
        t = wf.summary_table()
        assert "avg_sharpe" in t.columns

    def test_empty_results_empty_df(self, big_data, cfg):
        wf = WalkForwardEvaluator(big_data, cfg)
        t  = wf.summary_table()
        assert isinstance(t, pd.DataFrame)
