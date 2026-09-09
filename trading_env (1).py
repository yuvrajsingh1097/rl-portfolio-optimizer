"""Minimal trading environment — numpy only."""
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class EnvConfig:
    n_assets: int = 5
    window: int = 21
    transaction_cost: float = 0.001
    max_drawdown_pct: float = 0.30
    reward_type: str = "sharpe_dd"
    initial_capital: float = 100_000.0

def generate_market_data(n_assets=5, n_days=504, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    mu = 0.0003; sig = 0.012
    rets = mu + sig * rng.standard_normal((n_days, n_assets))
    # Add regime
    rets[100:200] *= 0.3; rets[100:200] -= 0.001
    cols = [f"Asset_{i+1}" for i in range(n_assets)]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=cols)
    return prices

class TradingEnv:
    def __init__(self, prices, config=None):
        self.prices = prices.values.astype(np.float32)
        self.dates  = prices.index
        self.n_assets = prices.shape[1]
        self.config = config or EnvConfig(n_assets=self.n_assets)
        self._reset_state()

    def _reset_state(self):
        self.step_idx = self.config.window
        self.weights  = np.ones(self.n_assets + 1) / (self.n_assets + 1)
        self.portfolio_value = self.config.initial_capital
        self.peak_value = self.config.initial_capital
        self.returns_history = []
        self.value_history = [self.config.initial_capital]
        self.done = False

    def reset(self):
        self._reset_state()
        return self._obs()

    def _obs(self):
        s = max(0, self.step_idx - self.config.window)
        hist = self.prices[s:self.step_idx]
        if len(hist) < 2:
            return np.zeros(self.n_assets * 3 + self.n_assets + 1, dtype=np.float32)
        feats = []
        for i in range(self.n_assets):
            r = np.log(hist[1:, i] / hist[:-1, i])
            feats.extend([r[-1], r.mean(), r.std()])
        feats.extend(self.weights.tolist())
        dd = (self.portfolio_value - self.peak_value) / (self.peak_value + 1e-8)
        feats.append(dd)
        return np.array(feats, dtype=np.float32)

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        exp_a = np.exp(action - action.max())
        w = exp_a / exp_a.sum()
        # Asset returns
        if self.step_idx >= len(self.prices):
            return self._obs(), 0.0, True, {"portfolio_value": self.portfolio_value, "portfolio_return": 0.0}
        price_today = self.prices[self.step_idx]
        price_prev  = self.prices[self.step_idx - 1]
        asset_rets  = np.log(price_today / price_prev)
        tc = self.config.transaction_cost * np.abs(w[:-1] - self.weights[:-1]).sum()
        port_ret = float(np.dot(asset_rets, w[:-1]) - tc)
        self.portfolio_value *= np.exp(port_ret)
        self.weights = w.copy()
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        dd = (self.portfolio_value - self.peak_value) / (self.peak_value + 1e-8)
        self.returns_history.append(port_ret)
        self.value_history.append(self.portfolio_value)
        self.step_idx += 1
        done = self.step_idx >= len(self.prices) or dd < -self.config.max_drawdown_pct
        self.done = done
        info = {"portfolio_value": self.portfolio_value, "portfolio_return": port_ret, "drawdown": dd}
        return self._obs(), port_ret, done, info

    def portfolio_metrics(self):
        rets = np.array(self.returns_history)
        vals = np.array(self.value_history)
        if len(rets) < 2:
            return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "final_value": self.portfolio_value, "n_steps": len(rets)}
        sharpe = float(rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)
        total  = (vals[-1] / vals[0] - 1) * 100
        peak   = np.maximum.accumulate(vals)
        mdd    = float(((vals - peak) / peak).min()) * 100
        return {"sharpe": round(sharpe, 4), "total_return": round(total, 3),
                "max_drawdown": round(mdd, 3), "final_value": round(float(vals[-1]), 2), "n_steps": len(rets)}
