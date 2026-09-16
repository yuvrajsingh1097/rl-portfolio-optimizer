# RL Portfolio Optimizer — Deep Reinforcement Learning

Portfolio allocation using Soft Actor-Critic (SAC) and PPO agents trained in a custom OpenAI Gym environment. Agents learn to allocate capital across equities with realistic transaction costs, drawdown penalties, and entropy-regularised reward shaping.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-84%20passing-brightgreen)](#testing)

---

## Architecture

```
Market Data → Custom Gym Env → State (returns, vol, momentum, weights, drawdown)
                                        ↓
                         SAC Agent (Actor + Twin Q-Networks)
                         PPO Agent (Actor-Critic + GAE)
                                        ↓
                    Portfolio Weights (softmax → sum=1)
                                        ↓
             Walk-Forward Evaluation vs 5 Benchmarks
             Permutation Feature Importance (SHAP-style)
```

---

## Modules

| File | Description |
|------|-------------|
| `env/trading_env.py` | Custom Gym — state/action/reward, transaction costs, episode logic |
| `agents/ppo_agent.py` | PPO with Dirichlet actor, twin critic, GAE, rollout buffer |
| `agents/sac_agent.py` | SAC with twin Q-networks, replay buffer, auto entropy (numpy) |
| `train/train.py` | Training loop, TensorBoard logging, checkpointing, early stopping |
| `evaluate/evaluate.py` | Walk-forward eval, 5 benchmarks, Sharpe/Sortino/Calmar/MDD/VaR |
| `dashboard/app.py` | Streamlit dashboard — equity curves, features, walk-forward |

---

## Results

| Strategy | Sharpe | Return | Max DD |
|----------|--------|--------|--------|
| SAC Agent | TBD | TBD | TBD |
| Buy & Hold | — | — | — |
| Momentum | — | — | — |
| Inv-Vol | — | — | — |
| 60/40 | — | — | — |

---

## Output Samples

![Dashboard](outputs/dashboard_preview.png)
![SAC Agent](outputs/sac_agent.png)
![Evaluation](outputs/evaluation.png)
![Training Loop](outputs/training_loop.png)
![PPO Agent](outputs/ppo_agent.png)
![Environment](outputs/trading_environment.png)

---

## Installation

```bash
git clone https://github.com/yuvrajsingh1097/rl-portfolio-optimizer
cd rl-portfolio-optimizer
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## License
MIT
