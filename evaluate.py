"""
Walk-Forward Evaluation & Benchmark Comparison
================================================
Evaluates portfolio strategies using walk-forward methodology.

Benchmarks:
    1. Equal-weight buy-and-hold
    2. Random agent
    3. 60/40 portfolio (60% equity, 40% cash)
    4. Momentum strategy

Metrics: Sharpe, Sortino, Calmar, MDD, VaR, CVaR, Win Rate
"""

import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from env.trading_env import TradingEnv, EnvConfig, generate_market_data


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def compute_metrics(returns: np.ndarray, rf: float = 0.05, periods: int = 252) -> dict:
    if len(returns) < 2:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
                "total_return_pct": 0.0, "ann_return_pct": 0.0,
                "ann_vol_pct": 0.0, "max_drawdown_pct": 0.0,
                "var_95_pct": 0.0, "cvar_95_pct": 0.0, "win_rate_pct": 0.0}

    rf_d    = rf / periods
    excess  = returns - rf_d
    total_r = float((1 + returns).prod() - 1)
    ann_r   = float((1 + total_r) ** (periods / len(returns)) - 1)
    ann_vol = float(returns.std() * np.sqrt(periods))
    sharpe  = float(excess.mean() / returns.std() * np.sqrt(periods)) if returns.std() > 1e-8 else 0.0
    down    = returns[returns < rf_d]
    sortino = float(excess.mean() / down.std() * np.sqrt(periods)) if len(down) > 1 and down.std() > 1e-8 else 0.0
    cum     = (1 + returns).cumprod()
    peak    = np.maximum.accumulate(cum)
    dd      = (cum - peak) / peak
    max_dd  = float(dd.min())
    calmar  = float(ann_r / abs(max_dd)) if max_dd != 0 else 0.0
    var95   = float(np.percentile(returns, 5))
    cvar95  = float(returns[returns <= var95].mean()) if (returns <= var95).any() else var95

    return {
        "total_return_pct": round(total_r * 100, 3),
        "ann_return_pct":   round(ann_r * 100, 3),
        "ann_vol_pct":      round(ann_vol * 100, 3),
        "sharpe":           round(sharpe, 4),
        "sortino":          round(sortino, 4),
        "calmar":           round(calmar, 4),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "var_95_pct":       round(var95 * 100, 4),
        "cvar_95_pct":      round(cvar95 * 100, 4),
        "win_rate_pct":     round(float((returns > 0).mean()) * 100, 2),
        "n_days":           len(returns),
    }


def rolling_sharpe(returns: np.ndarray, window: int = 63, rf: float = 0.05) -> np.ndarray:
    rf_d = rf / 252
    out  = np.full(len(returns), np.nan)
    for i in range(window, len(returns)):
        r = returns[i-window:i]
        if r.std() > 1e-8:
            out[i] = (r.mean() - rf_d) / r.std() * np.sqrt(252)
    return out


def monthly_returns_table(returns: np.ndarray, dates) -> pd.DataFrame:
    sr = pd.Series(returns, index=dates[:len(returns)])
    monthly = sr.resample("ME").apply(lambda x: (1+x).prod()-1)
    df = pd.DataFrame({"ret": monthly, "year": monthly.index.year, "month": monthly.index.month})
    pivot = df.pivot(index="year", columns="month", values="ret")
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [months[m-1] for m in pivot.columns]
    return pivot.round(4)


# ---------------------------------------------------------------------------
# Benchmark strategies
# ---------------------------------------------------------------------------

def _run_strategy(prices, config, weight_fn):
    """Generic strategy runner."""
    env  = TradingEnv(prices=prices, config=config)
    obs  = env.reset()
    done = False
    rets = []
    step = 0
    while not done:
        w = weight_fn(step, env)
        obs, _, done, info = env.step(w)
        rets.append(info.get("portfolio_return", 0.0))
        step += 1
    return np.array(rets, dtype=np.float32)


def run_buy_and_hold(prices, config, weight_cash=0.0):
    n = prices.shape[1]
    w = np.ones(n+1) / (n+1)
    if weight_cash > 0:
        w[:-1] = (1-weight_cash) / n
        w[-1]  = weight_cash
    return _run_strategy(prices, config, lambda s, e: w)


def run_random_agent(prices, config, seed=0):
    rng = np.random.default_rng(seed)
    n   = prices.shape[1] + 1
    return _run_strategy(prices, config, lambda s, e: rng.dirichlet(np.ones(n)))


def run_momentum(prices, config, lookback=21):
    n  = prices.shape[1]
    pv = prices.values

    def weight_fn(step, env):
        idx = step + config.window
        if idx >= lookback:
            hist = pv[max(0,idx-lookback):idx]
            if len(hist) >= 2:
                mom = np.log(hist[-1] / hist[0] + 1e-8)
                exp_m = np.exp(mom - mom.max())
                aw = exp_m / exp_m.sum() * 0.9
                return np.append(aw, 0.1)
        return np.ones(n+1)/(n+1)

    return _run_strategy(prices, config, weight_fn)


def run_equal_sharpe(prices, config, lookback=63):
    """Inverse-volatility weighting."""
    n  = prices.shape[1]
    pv = prices.values

    def weight_fn(step, env):
        idx = step + config.window
        if idx >= lookback:
            hist = pv[max(0,idx-lookback):idx]
            if len(hist) >= 2:
                rets = np.log(hist[1:] / hist[:-1])
                vols = rets.std(axis=0) + 1e-8
                iv   = 1 / vols
                aw   = iv / iv.sum() * 0.95
                return np.append(aw, 0.05)
        return np.ones(n+1)/(n+1)

    return _run_strategy(prices, config, weight_fn)


# ---------------------------------------------------------------------------
# Full comparison
# ---------------------------------------------------------------------------

def full_comparison(test_prices, env_cfg, agent=None):
    """Run all strategies on the same test period."""
    strategies = {
        "Buy & Hold":    lambda: run_buy_and_hold(test_prices, env_cfg),
        "Random":        lambda: run_random_agent(test_prices, env_cfg),
        "Momentum":      lambda: run_momentum(test_prices, env_cfg),
        "Inv-Vol":       lambda: run_equal_sharpe(test_prices, env_cfg),
        "60/40":         lambda: run_buy_and_hold(test_prices, env_cfg, weight_cash=0.4),
    }
    if agent is not None:
        from agents.ppo_agent import run_ppo_agent
        strategies["PPO Agent"] = lambda: run_ppo_agent(agent, test_prices, env_cfg)

    results = {}
    for name, fn in strategies.items():
        rets = fn()
        results[name] = {
            "returns": rets,
            "equity":  (1 + rets).cumprod() * 100_000,
            "metrics": compute_metrics(rets),
        }
    return results


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    train_window: int = 252
    test_window:  int = 63
    n_folds:      int = 3


class WalkForwardEvaluator:
    def __init__(self, all_data, env_cfg, wf_cfg=None):
        self.all_data = all_data
        self.env_cfg  = env_cfg
        self.wf_cfg   = wf_cfg or WalkForwardConfig()
        self.results  = []

    def run(self, verbose=True):
        wf = self.wf_cfg
        n  = len(self.all_data)

        for fold in range(wf.n_folds):
            train_start = fold * wf.test_window
            train_end   = train_start + wf.train_window
            test_end    = train_end + wf.test_window
            if test_end > n:
                break

            test_data = self.all_data.iloc[train_end:test_end]
            if verbose:
                print(f"  Fold {fold+1}: test [{train_end}:{test_end}] ({len(test_data)} days)")

            fold_r = {"fold": fold+1}
            for name, fn in [
                ("buy_and_hold", lambda: run_buy_and_hold(test_data, self.env_cfg)),
                ("random",       lambda: run_random_agent(test_data, self.env_cfg, seed=fold)),
                ("momentum",     lambda: run_momentum(test_data, self.env_cfg)),
                ("inv_vol",      lambda: run_equal_sharpe(test_data, self.env_cfg)),
                ("60_40",        lambda: run_buy_and_hold(test_data, self.env_cfg, weight_cash=0.4)),
            ]:
                rets = fn()
                m = compute_metrics(rets)
                fold_r[name] = m
                if verbose:
                    print(f"    {name:<14} Sharpe={m['sharpe']:.3f}  Return={m['total_return_pct']:.2f}%")

            self.results.append(fold_r)
        return self.results

    def summary_table(self):
        if not self.results:
            return pd.DataFrame()
        strategies = ["buy_and_hold","random","momentum","inv_vol","60_40"]
        rows = []
        for s in strategies:
            sharpes = [f[s]["sharpe"] for f in self.results if s in f]
            returns = [f[s]["total_return_pct"] for f in self.results if s in f]
            mdds    = [f[s]["max_drawdown_pct"] for f in self.results if s in f]
            rows.append({"strategy": s, "avg_sharpe": round(float(np.mean(sharpes)),4),
                         "avg_return%": round(float(np.mean(returns)),3),
                         "avg_mdd%": round(float(np.mean(mdds)),3)})
        return pd.DataFrame(rows).set_index("strategy")


if __name__ == "__main__":
    print("=" * 55)
    print("Walk-Forward Evaluation Demo")
    print("=" * 55)
    data    = generate_market_data(n_assets=5, n_days=756, seed=42)
    env_cfg = EnvConfig(n_assets=5)
    test    = data.iloc[-252:]

    print("\nSingle-period comparison:")
    comp = full_comparison(test, env_cfg)
    print(f"{'Strategy':<14} {'Sharpe':>7} {'Return%':>9} {'MDD%':>7}")
    print("─"*35)
    for name, res in comp.items():
        m = res["metrics"]
        print(f"{name:<14} {m['sharpe']:>7.3f} {m['total_return_pct']:>9.3f} {m['max_drawdown_pct']:>7.3f}")

    print("\nWalk-forward (3 folds):")
    wf  = WalkForwardEvaluator(data, env_cfg, WalkForwardConfig(n_folds=3))
    wf.run(verbose=True)
    print("\nSummary:")
    print(wf.summary_table().to_string())
