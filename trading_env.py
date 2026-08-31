"""
Custom Trading Environment — OpenAI Gym
=========================================
A portfolio management environment for training RL agents.

State space:
    - Historical price features (returns, vol, momentum, RSI)
    - Current portfolio weights
    - Current portfolio value (normalised)
    - Days since last rebalance

Action space:
    - Continuous: target portfolio weights (n_assets + cash)
    - Softmax-normalised to sum to 1

Reward function (configurable):
    - v1: Log portfolio return
    - v2: Rolling Sharpe ratio
    - v3: Sharpe + drawdown penalty
    - v4: Sortino ratio

Features:
    - Realistic transaction costs (proportional + fixed)
    - Drawdown penalty
    - Short selling support (optional)
    - Multi-asset: up to 20 stocks
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """Configuration for the trading environment."""
    n_assets:         int   = 5        # number of risky assets
    window:           int   = 30       # lookback window for features
    initial_capital:  float = 100_000  # starting portfolio value
    transaction_cost: float = 0.001    # proportional cost per trade (0.1%)
    reward_type:      str   = "sharpe" # log_return | sharpe | drawdown | sortino
    allow_short:      bool  = False    # allow short positions
    drawdown_penalty: float = 0.1      # penalty weight for max drawdown
    episode_length:   int   = 252      # trading days per episode


# ---------------------------------------------------------------------------
# Synthetic market data generator
# ---------------------------------------------------------------------------

def generate_market_data(
    n_assets:  int   = 5,
    n_days:    int   = 1000,
    seed:      int   = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily return data for n_assets.

    Includes:
        - Regime shifts (bull / bear / sideways)
        - Sector correlation structure
        - Fat-tailed returns (t-distribution)
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")

    # Regime schedule
    regimes = np.zeros(n_days, dtype=int)
    regimes[200:350] = 1   # bear
    regimes[600:700] = 1   # bear
    regimes[400:600] = 2   # sideways

    regime_params = {
        0: dict(mu=0.10/252,  sigma=0.15/np.sqrt(252)),   # bull
        1: dict(mu=-0.20/252, sigma=0.30/np.sqrt(252)),   # bear
        2: dict(mu=0.02/252,  sigma=0.08/np.sqrt(252)),   # sideways
    }

    # Asset correlation matrix (block structure for 2 sectors)
    half = n_assets // 2
    corr = np.eye(n_assets) * 0.4
    for i in range(half):
        for j in range(half):
            if i != j: corr[i,j] = 0.6
    for i in range(half, n_assets):
        for j in range(half, n_assets):
            if i != j: corr[i,j] = 0.5
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    # Cholesky decomposition for correlated returns
    L = np.linalg.cholesky(corr)

    returns = np.zeros((n_days, n_assets))
    for t in range(n_days):
        p = regime_params[regimes[t]]
        # t-distributed returns (fat tails)
        z = rng.standard_t(df=5, size=n_assets)
        z = z / np.sqrt(5/3)   # normalise variance
        corr_z = L @ z
        returns[t] = p["mu"] + p["sigma"] * corr_z

    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=[f"Asset_{i+1}" for i in range(n_assets)],
    )
    return prices


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_features(prices: pd.DataFrame, window: int = 30) -> np.ndarray:
    """
    Compute feature matrix from price data.

    Features per asset (6 features × n_assets):
        - Log return (1-day)
        - Rolling volatility (window days, annualised)
        - Momentum (window-day return)
        - RSI (14-day)
        - Return relative to cross-sectional mean
        - Vol ratio (short vol / long vol)

    Returns array of shape (n_days, n_assets * 6).
    """
    rets = np.log(prices / prices.shift(1)).fillna(0).values
    n_days, n_assets = rets.shape
    feats = np.zeros((n_days, n_assets * 6))

    for a in range(n_assets):
        r = rets[:, a]

        # 1. Daily return
        feats[:, a*6 + 0] = r

        # 2. Rolling vol (annualised)
        for t in range(window, n_days):
            feats[t, a*6 + 1] = r[t-window:t].std() * np.sqrt(252)

        # 3. Momentum
        prices_a = prices.values[:, a]
        for t in range(window, n_days):
            feats[t, a*6 + 2] = (prices_a[t] - prices_a[t-window]) / prices_a[t-window]

        # 4. RSI (14-day)
        delta = np.diff(prices_a, prepend=prices_a[0])
        gain  = np.maximum(delta, 0)
        loss  = np.maximum(-delta, 0)
        for t in range(14, n_days):
            avg_gain = gain[t-14:t].mean()
            avg_loss = loss[t-14:t].mean()
            rs = avg_gain / (avg_loss + 1e-8)
            feats[t, a*6 + 3] = 100 - 100/(1+rs)

        # 5. Relative return (vs cross-section)
        for t in range(1, n_days):
            cross_mean = rets[t].mean()
            feats[t, a*6 + 4] = r[t] - cross_mean

        # 6. Vol ratio (short / long vol)
        for t in range(window, n_days):
            short_vol = r[max(0,t-10):t].std() + 1e-8
            long_vol  = r[t-window:t].std() + 1e-8
            feats[t, a*6 + 5] = short_vol / long_vol

    # Clip outliers before standardise
    feats = np.where(np.isfinite(feats), feats, 0.0)
    feats = np.clip(feats, -10, 10)
    mean  = feats[window:].mean(axis=0)
    std   = feats[window:].std(axis=0) + 1e-8
    feats = (feats - mean) / std
    return feats.astype(np.float32)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def reward_log_return(portfolio_return: float, **kwargs) -> float:
    return float(portfolio_return)


def reward_sharpe(
    returns_history: list,
    portfolio_return: float,
    rf_daily: float = 0.05/252,
    **kwargs,
) -> float:
    if len(returns_history) < 10:
        return float(portfolio_return)
    arr = np.array(returns_history[-63:])   # rolling 63-day
    excess = arr - rf_daily
    if arr.std() < 1e-8:
        return 0.0
    return float(np.mean(excess) / arr.std() * np.sqrt(252) / 252)


def reward_sortino(
    returns_history: list,
    portfolio_return: float,
    rf_daily: float = 0.05/252,
    **kwargs,
) -> float:
    if len(returns_history) < 10:
        return float(portfolio_return)
    arr      = np.array(returns_history[-63:])
    excess   = arr - rf_daily
    downside = arr[arr < rf_daily]
    if len(downside) < 2 or downside.std() < 1e-8:
        return float(np.mean(excess) * np.sqrt(252) / 252)
    return float(np.mean(excess) / downside.std() * np.sqrt(252) / 252)


def reward_drawdown(
    returns_history: list,
    portfolio_return: float,
    drawdown_penalty: float = 0.1,
    **kwargs,
) -> float:
    base = reward_sharpe(returns_history, portfolio_return)
    if len(returns_history) < 5:
        return base
    cum = np.cumprod(1 + np.array(returns_history))
    mdd = float((cum / np.maximum.accumulate(cum) - 1).min())
    return base + drawdown_penalty * mdd   # mdd is negative → reduces reward


REWARD_FNS = {
    "log_return": reward_log_return,
    "sharpe":     reward_sharpe,
    "sortino":    reward_sortino,
    "drawdown":   reward_drawdown,
}


# ---------------------------------------------------------------------------
# Trading Environment
# ---------------------------------------------------------------------------

class TradingEnv:
    """
    Portfolio management environment (Gym-compatible interface).

    State  : (window × n_assets × 6 features) + portfolio weights + value
    Action : target portfolio weights (n_assets + 1 cash), softmax-normalised
    Reward : configurable (log_return / sharpe / sortino / drawdown)
    """

    def __init__(self, prices: pd.DataFrame, config: EnvConfig = None):
        self.prices = prices
        self.config = config or EnvConfig(n_assets=prices.shape[1])
        self.n_assets = prices.shape[1]
        self.n_days   = len(prices)

        # Pre-compute features
        self.features = compute_features(prices, window=self.config.window)

        # Gym-style spaces info
        feat_dim          = self.n_assets * 6
        self.obs_dim      = self.config.window * feat_dim + self.n_assets + 1 + 1
        self.action_dim   = self.n_assets + 1   # +1 for cash

        self._reset_state()

    def _reset_state(self):
        self.t              = self.config.window
        self.portfolio_val  = self.config.initial_capital
        self.weights        = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.weights[-1]    = 1.0   # start fully in cash
        self.returns_hist   = []
        self.value_hist     = [self.config.initial_capital]
        self.weights_hist   = [self.weights.copy()]
        self.done           = False
        self.peak_val       = self.config.initial_capital

    def reset(self) -> np.ndarray:
        """Reset environment to a random starting point."""
        self._reset_state()
        # Random start within valid range
        max_start = self.n_days - self.config.episode_length - 2
        if max_start > self.config.window:
            self.t = np.random.randint(self.config.window, max(max_start, self.config.window + 1))
        else:
            self.t = self.config.window
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        # Feature window: (window, n_assets*6) → flatten
        feat_window = self.features[self.t - self.config.window : self.t]
        feat_flat   = feat_window.flatten().astype(np.float32)

        # Portfolio state
        weights_obs = self.weights.astype(np.float32)
        val_norm    = np.array(
            [self.portfolio_val / self.config.initial_capital], dtype=np.float32
        )

        return np.concatenate([feat_flat, weights_obs, val_norm])

    def _apply_action(self, action: np.ndarray) -> float:
        """
        Apply portfolio rebalancing action.

        1. Softmax-normalise action to get target weights
        2. Compute transaction costs from weight changes
        3. Apply market returns
        4. Update portfolio value and weights

        Returns portfolio return for this step.
        """
        # Softmax normalisation
        if self.config.allow_short:
            target = np.tanh(action)
            target = target / (np.abs(target).sum() + 1e-8)
        else:
            exp_a  = np.exp(action - action.max())
            target = exp_a / exp_a.sum()

        target = target.astype(np.float32)

        # Transaction costs (proportional)
        weight_change = np.abs(target[:-1] - self.weights[:-1])
        costs = weight_change.sum() * self.config.transaction_cost

        # Market returns for this step
        prices_t   = self.prices.values[self.t]
        prices_t1  = self.prices.values[self.t - 1]
        asset_rets = (prices_t - prices_t1) / prices_t1

        # Portfolio return = weighted sum of asset returns
        portfolio_ret = float(np.dot(target[:-1], asset_rets)) - costs

        # Update
        self.portfolio_val *= (1 + portfolio_ret)
        self.weights        = target.copy()
        self.peak_val       = max(self.peak_val, self.portfolio_val)

        return portfolio_ret

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : array of shape (n_assets + 1,) — target portfolio weights

        Returns
        -------
        obs    : next observation
        reward : scalar reward
        done   : episode termination flag
        info   : dict with portfolio metrics
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")

        # Apply action and get portfolio return
        port_ret = self._apply_action(action)
        self.returns_hist.append(port_ret)
        self.value_hist.append(self.portfolio_val)
        self.weights_hist.append(self.weights.copy())

        # Compute reward
        reward_fn = REWARD_FNS.get(self.config.reward_type, reward_sharpe)
        reward = reward_fn(
            returns_history=self.returns_hist,
            portfolio_return=port_ret,
            drawdown_penalty=self.config.drawdown_penalty,
        )

        # Advance time
        self.t += 1
        ep_start = self.config.window
        ep_done  = (self.t >= min(ep_start + self.config.episode_length, self.n_days - 1))
        bankrupt = self.portfolio_val < self.config.initial_capital * 0.05

        self.done = bool(ep_done or bankrupt)

        # Info dict
        drawdown = (self.portfolio_val - self.peak_val) / self.peak_val
        info = {
            "portfolio_value":  round(self.portfolio_val, 2),
            "portfolio_return": round(port_ret, 6),
            "drawdown":         round(drawdown, 6),
            "weights":          self.weights.tolist(),
            "step":             self.t,
        }

        obs = self._get_obs() if not self.done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, float(reward), self.done, info

    def portfolio_metrics(self) -> dict:
        """Compute full portfolio performance metrics at episode end."""
        if len(self.returns_hist) < 2:
            return {}
        rets = np.array(self.returns_hist)
        cum  = np.cumprod(1 + rets)
        mdd  = float((cum / np.maximum.accumulate(cum) - 1).min())

        total_ret = float(cum[-1] - 1)
        ann_ret   = float((1 + total_ret) ** (252 / max(len(rets), 1)) - 1)
        ann_vol   = float(rets.std() * np.sqrt(252))
        sharpe    = float((rets.mean() - 0.05/252) / (rets.std() + 1e-8) * np.sqrt(252))

        down  = rets[rets < 0.05/252]
        sort_ = float((rets.mean() - 0.05/252) / (down.std() + 1e-8) * np.sqrt(252)) \
                if len(down) > 1 else 0.0
        calmar = float(ann_ret / abs(mdd)) if mdd != 0 else 0.0

        return {
            "total_return":   round(total_ret * 100, 2),
            "ann_return":     round(ann_ret   * 100, 2),
            "ann_vol":        round(ann_vol   * 100, 2),
            "sharpe":         round(sharpe, 3),
            "sortino":        round(sort_, 3),
            "max_drawdown":   round(mdd   * 100, 2),
            "calmar":         round(calmar, 3),
            "final_value":    round(self.portfolio_val, 2),
            "n_steps":        len(self.returns_hist),
        }


# ---------------------------------------------------------------------------
# Random agent (baseline)
# ---------------------------------------------------------------------------

def random_agent_episode(env: TradingEnv) -> dict:
    """Run one episode with a random agent. Returns performance metrics."""
    obs = env.reset()
    done = False
    while not done:
        action = np.random.randn(env.action_dim).astype(np.float32)
        obs, reward, done, info = env.step(action)
    return env.portfolio_metrics()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Trading Environment Demo")
    print("=" * 60)

    prices = generate_market_data(n_assets=5, n_days=800)
    config = EnvConfig(
        n_assets=5, window=30, episode_length=252,
        reward_type="sharpe", transaction_cost=0.001,
    )
    env = TradingEnv(prices, config)

    print(f"\n  Assets       : {list(prices.columns)}")
    print(f"  Obs dim      : {env.obs_dim}")
    print(f"  Action dim   : {env.action_dim}")
    print(f"  Episode len  : {config.episode_length} days")

    # Random agent baseline
    print("\n  Running random agent baseline...")
    metrics = random_agent_episode(env)
    print(f"\n  Random Agent Performance:")
    for k, v in metrics.items():
        unit = "%" if "return" in k or "vol" in k or "drawdown" in k else ""
        print(f"    {k:<20}: {v}{unit}")

    # Sanity check: obs shape
    obs = env.reset()
    action = np.random.randn(env.action_dim)
    obs2, reward, done, info = env.step(action)
    print(f"\n  Obs shape    : {obs.shape}")
    print(f"  Step reward  : {reward:.6f}")
    print(f"  Portfolio val: ${info['portfolio_value']:,.2f}")
