# RL Portfolio Optimizer — PPO Agent

Deep Reinforcement Learning portfolio allocation using a PPO agent trained in a custom OpenAI Gym environment. The agent learns to allocate capital across a basket of equities with realistic transaction costs, drawdown penalties, and Sharpe-based reward shaping.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-39%20passing-brightgreen)](#testing)

---

## What this does

| Module | Description |
|--------|-------------|
| `env/trading_env.py` | Custom Gym environment — state space, action space, reward functions, episode logic |
| `agents/ppo_agent.py` | PPO agent with actor-critic network, GAE advantage estimation |
| `agents/sac_agent.py` | SAC agent (off-policy alternative for comparison) |
| `train/train.py` | Training loop with TensorBoard logging, checkpointing |
| `evaluate/evaluate.py` | Walk-forward evaluation, benchmark comparison |
| `dashboard/app.py` | Streamlit portfolio dashboard |

---

## Environment

```
State  : price features (returns, vol, momentum, RSI) × n_assets
         + current weights + drawdown + portfolio return
Action : continuous portfolio weights ∈ [0,1]^n  (softmax normalised, sum=1)
Reward : Sharpe ratio (rolling 21d) − drawdown penalty
Episode: ends when data exhausted or max drawdown > 30%
```

---

## Results

| Metric | Value |
|--------|-------|
| PPO Sharpe (test) | TBD |
| Buy-and-hold Sharpe | TBD |
| Max Drawdown | TBD |
| Win Rate | TBD |

---

## Output Samples

![Trading Environment](outputs/trading_environment.png)

---

## Project Structure

```
rl-portfolio-optimizer/
├── env/trading_env.py      # Custom Gym environment
├── agents/
│   ├── ppo_agent.py        # PPO actor-critic
│   └── sac_agent.py        # SAC agent
├── train/train.py          # Training loop
├── evaluate/evaluate.py    # Walk-forward evaluation
├── dashboard/app.py        # Streamlit dashboard
├── tests/
├── outputs/
├── models/
└── data/
```

---

## Installation

```bash
git clone https://github.com/yuvrajsingh1097/rl-portfolio-optimizer
cd rl-portfolio-optimizer
pip install -r requirements.txt
python env/trading_env.py
python -m pytest tests/ -v
streamlit run dashboard/app.py
```

---

## License
MIT
