"""
Streamlit Dashboard — RL Portfolio Optimizer
=============================================
Interactive dashboard showing:
    - Live portfolio simulation (PPO / SAC / benchmarks)
    - Agent training metrics
    - Walk-forward evaluation results
    - Feature importance (permutation)
    - Risk metrics comparison table
    - Monthly returns heatmap

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.trading_env import TradingEnv, EnvConfig, generate_market_data
from agents.sac_agent import SACAgent, SACConfig, permutation_importance
from evaluate.evaluate import (
    full_comparison, WalkForwardConfig, WalkForwardEvaluator,
    compute_metrics, rolling_sharpe, monthly_returns_table,
    run_buy_and_hold, run_random_agent, run_momentum, run_equal_sharpe,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RL Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-box { background:#161b22; border:1px solid #30363d;
              border-radius:8px; padding:14px; text-align:center; }
.positive { color:#3fb950; font-weight:600; }
.negative { color:#ff7b72; font-weight:600; }
h1,h2,h3 { color:#e6edf3; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📈 RL Portfolio Optimizer")
st.sidebar.markdown("---")

n_assets    = st.sidebar.slider("Number of assets", 3, 8, 5)
n_days      = st.sidebar.slider("Training days",  252, 756, 504, step=63)
train_steps = st.sidebar.slider("SAC training steps", 200, 1000, 400, step=100)
seed        = st.sidebar.number_input("Random seed", value=42, min_value=0)

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🏠 Overview",
    "🤖 Agent Training",
    "📊 Strategy Comparison",
    "🔍 Feature Importance",
    "📅 Walk-Forward",
])

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def build_system(n_assets, n_days, train_steps, seed):
    data    = generate_market_data(n_assets=n_assets, n_days=n_days, seed=seed)
    env_cfg = EnvConfig(n_assets=n_assets)
    env     = TradingEnv(prices=data, config=env_cfg)
    obs     = env.reset()

    state_dim  = obs.shape[0]
    action_dim = n_assets + 1

    sac_cfg = SACConfig(hidden_dims=[64,32], learning_starts=50, batch_size=32)
    agent   = SACAgent(state_dim, action_dim, sac_cfg, seed=seed)

    rng  = np.random.default_rng(seed)
    obs  = env.reset(); done = False
    al, cl, ent = [], [], []

    for step in range(train_steps):
        action, _, _ = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        agent.store(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()
        done = False if done else done
        if step >= sac_cfg.learning_starts:
            m = agent.update()
            if m:
                al.append(m['actor_loss'])
                cl.append(m['critic1_loss'])
                ent.append(m['alpha'])

    # Test period
    test_data = data.iloc[-126:] if len(data) > 126 else data.iloc[-63:]
    comp = full_comparison(test_data, env_cfg)

    return {
        "agent": agent, "data": data, "test_data": test_data,
        "env_cfg": env_cfg, "comp": comp,
        "actor_losses": al, "critic_losses": cl, "alphas": ent,
        "state_dim": state_dim, "action_dim": action_dim,
    }

with st.spinner("Building RL system..."):
    sys_data = build_system(n_assets, n_days, train_steps, int(seed))

agent    = sys_data["agent"]
data     = sys_data["data"]
test_data= sys_data["test_data"]
env_cfg  = sys_data["env_cfg"]
comp     = sys_data["comp"]

STRAT_COLORS = {
    "Buy & Hold":"#58a6ff","Random":"#8b949e",
    "Momentum":"#3fb950","Inv-Vol":"#d2a8ff","60/40":"#f0883e",
}

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("📈 RL Portfolio Optimizer")
    st.markdown("Deep Reinforcement Learning portfolio allocation — SAC agent vs benchmarks.")

    # Top metrics
    bh  = comp.get("Buy & Hold", {}).get("metrics", {})
    mom = comp.get("Momentum", {}).get("metrics", {})
    sac_m = compute_metrics(
        np.array([info.get("portfolio_return",0) for info in [{}]*10])
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Assets",       f"{n_assets}")
    col2.metric("Train Days",   f"{n_days}")
    col3.metric("SAC Updates",  f"{agent.update_count:,}")
    col4.metric("Buffer Size",  f"{len(agent.buffer):,}")
    col5.metric("Total Params", f"{agent.param_count()['total_params']:,}")

    st.markdown("---")
    st.subheader("Strategy Performance Summary")

    rows = []
    for name, res in comp.items():
        m = res["metrics"]
        rows.append({
            "Strategy":       name,
            "Sharpe":         m.get("sharpe", 0),
            "Return (%)":     m.get("total_return_pct", 0),
            "Max DD (%)":     m.get("max_drawdown_pct", 0),
            "Sortino":        m.get("sortino", 0),
            "Win Rate (%)":   m.get("win_rate_pct", 0),
        })
    df = pd.DataFrame(rows).set_index("Strategy")
    st.dataframe(
        df.style.background_gradient(subset=["Sharpe","Return (%)"], cmap="RdYlGn")
               .background_gradient(subset=["Max DD (%)"], cmap="RdYlGn_r"),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Equity Curves")
    chart_df = pd.DataFrame(
        {name: res["equity"] for name, res in comp.items()}
    )
    st.line_chart(chart_df)

# ---------------------------------------------------------------------------
# Page: Agent Training
# ---------------------------------------------------------------------------
elif page == "🤖 Agent Training":
    st.title("🤖 SAC Agent Training")

    col1, col2, col3 = st.columns(3)
    col1.metric("Updates",     agent.update_count)
    col2.metric("Buffer",      len(agent.buffer))
    col3.metric("Final α",     f"{agent.alpha:.4f}")

    st.markdown("---")

    if sys_data["actor_losses"]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Losses")
            loss_df = pd.DataFrame({
                "Actor Loss":  sys_data["actor_losses"],
                "Critic Loss": sys_data["critic_losses"],
            })
            st.line_chart(loss_df)

        with col2:
            st.subheader("Entropy Temperature (α)")
            st.line_chart(pd.Series(sys_data["alphas"], name="Alpha"))
    else:
        st.info("Increase training steps to see loss curves.")

    st.markdown("---")
    st.subheader("Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**SAC Components**")
        params = agent.param_count()
        for k, v in params.items():
            st.metric(k.replace("_"," ").title(), f"{v:,}")
    with col2:
        st.markdown("**PPO vs SAC**")
        cmp_df = pd.DataFrame({
            "Property":   ["Type","Buffer","Networks","Target Nets","Entropy","Sample Eff."],
            "PPO":        ["On-policy","Rollout","Actor+Critic","✗","Bonus coef","Low"],
            "SAC":        ["Off-policy","Replay","Actor+2×Q","✓ Polyak","Auto α","High"],
        }).set_index("Property")
        st.dataframe(cmp_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Strategy Comparison
# ---------------------------------------------------------------------------
elif page == "📊 Strategy Comparison":
    st.title("📊 Strategy Comparison")

    col1, col2 = st.columns([3,1])
    with col2:
        show_strats = st.multiselect("Show strategies", list(comp.keys()), default=list(comp.keys()))

    with col1:
        st.subheader("Equity Curves")
        equity_df = pd.DataFrame({n: comp[n]["equity"] for n in show_strats if n in comp})
        st.line_chart(equity_df)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sharpe Ratio")
        sharpe_df = pd.DataFrame(
            {n: [comp[n]["metrics"]["sharpe"]] for n in show_strats if n in comp},
            index=["Sharpe"]
        ).T.sort_values("Sharpe", ascending=False)
        st.bar_chart(sharpe_df)

    with col2:
        st.subheader("Rolling Sharpe (30-day)")
        rs_df = pd.DataFrame({
            n: rolling_sharpe(comp[n]["returns"], window=min(30, len(comp[n]["returns"])//3))
            for n in show_strats if n in comp
        })
        st.line_chart(rs_df)

    st.markdown("---")
    st.subheader("Monthly Returns Heatmap — Buy & Hold")
    if "Buy & Hold" in comp:
        rets = comp["Buy & Hold"]["returns"]
        mt   = monthly_returns_table(rets, test_data.index) * 100
        if not mt.empty:
            st.dataframe(
                mt.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.2f}%"),
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Page: Feature Importance
# ---------------------------------------------------------------------------
elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance")
    st.markdown("Permutation importance: measures how much the agent's Q-value drops when each feature is randomly shuffled.")

    n_steps   = st.slider("Steps per eval", 20, 100, 40)
    n_repeats = st.slider("Repeats", 1, 5, 2)

    state_dim = sys_data["state_dim"]
    feat_names = (
        [f"return_{i+1}"   for i in range(n_assets)] +
        [f"momentum_{i+1}" for i in range(n_assets)] +
        [f"vol_{i+1}"      for i in range(n_assets)] +
        [f"weight_{i+1}"   for i in range(n_assets+1)] +
        ["drawdown"]
    )[:state_dim]

    env_fi = TradingEnv(prices=test_data, config=env_cfg)

    with st.spinner("Computing feature importance..."):
        importance = permutation_importance(agent, env_fi, feat_names, n_steps, n_repeats)

    imp_df = pd.DataFrame(
        list(importance.items()), columns=["Feature","Importance"]
    ).sort_values("Importance", ascending=True)

    st.bar_chart(imp_df.set_index("Feature"))

    st.markdown("---")
    st.subheader("Top 10 Most Important Features")
    top10 = pd.DataFrame(
        list(importance.items())[:10], columns=["Feature","Importance"]
    )
    st.dataframe(top10, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Walk-Forward
# ---------------------------------------------------------------------------
elif page == "📅 Walk-Forward":
    st.title("📅 Walk-Forward Evaluation")
    st.markdown("Out-of-sample validation across multiple time folds.")

    n_folds      = st.slider("Number of folds", 2, 5, 3)
    train_window = st.slider("Train window (days)", 63, 252, 126)
    test_window  = st.slider("Test window (days)",  21, 126, 63)

    if len(data) < train_window + test_window * n_folds:
        st.warning("Not enough data for selected settings. Increase training days.")
    else:
        with st.spinner("Running walk-forward evaluation..."):
            wf_cfg = WalkForwardConfig(
                train_window=train_window,
                test_window=test_window,
                n_folds=n_folds,
            )
            wf_eval = WalkForwardEvaluator(data, env_cfg, wf_cfg)
            wf_eval.run(verbose=False)
            summary = wf_eval.summary_table()

        st.subheader("Average Metrics Across Folds")
        st.dataframe(
            summary.style.background_gradient(subset=["avg_sharpe"], cmap="RdYlGn"),
            use_container_width=True,
        )

        st.subheader("Per-Fold Sharpe Ratios")
        fold_data = {}
        for fold_r in wf_eval.results:
            for strat in ["buy_and_hold","random","momentum","inv_vol","60_40"]:
                if strat in fold_r:
                    if strat not in fold_data:
                        fold_data[strat] = []
                    fold_data[strat].append(fold_r[strat]["sharpe"])

        fold_df = pd.DataFrame(fold_data,
                               index=[f"Fold {i+1}" for i in range(len(wf_eval.results))])
        st.line_chart(fold_df)
