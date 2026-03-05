# RL Learning Project

A hands-on reinforcement learning project using **LunarLander-v2** (Gymnasium). You implement the algorithms; the project provides the environment, training infrastructure, and visualization tools.

## Philosophy

Learn by doing. Each algorithm has:
- A **template** (`algorithms/<name>/agent.py`) with TODO comments guiding your implementation
- A **reference solution** (`solutions/<name>/agent.py`) you can peek at if stuck
- **Neural network helpers** already built for you (`network.py` files)
- A **shared training loop** so you focus purely on the RL logic

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd rl-learning
uv sync

# If box2d fails to install, you may need system dependencies:
# Ubuntu/Debian: sudo apt-get install swig
# macOS: brew install swig
```

## Quick Start

```python
from environment import LunarLanderEnv

env = LunarLanderEnv()
obs, state_info = env.reset()

# state_info gives you named access to every variable
print(f"Position: ({state_info.x_position:.2f}, {state_info.y_position:.2f})")
print(f"Speed: {state_info.speed:.2f}")
print(f"Upright: {state_info.is_upright}")

# Take a step
obs, reward, terminated, truncated, state_info, reward_signals = env.step(2)  # Fire main engine

# reward_signals decomposes the reward for you
print(f"Proximity reward: {reward_signals.proximity_reward:.2f}")
print(f"Fuel penalty: {reward_signals.fuel_penalty:.2f}")
print(f"Default reward: {reward_signals.default_gymnasium_reward:.2f}")
```

## Learning Workflow

1. **Read the theory** — Use `notebooks/rl_theory.ipynb` for notes
2. **Study the template** — Read the docstring and TODOs in `algorithms/<name>/agent.py`
3. **Implement** — Fill in the methods
4. **Train** — `uv run python scripts/train.py --algo <name> --episodes 1000`
5. **Evaluate** — `uv run python scripts/evaluate.py --algo <name> --checkpoint results/checkpoints/<name>_final.pt`
6. **Compare** — `uv run python scripts/compare.py --algos q_learning dqn ppo`
7. **Check solution** — If stuck, look at `solutions/<name>/agent.py`

## Algorithm Progression

| # | Algorithm | Type | Key Concept | Difficulty |
|---|-----------|------|-------------|------------|
| 1 | Q-Learning | Value-based | Q-table, Bellman equation | Easy |
| 2 | DQN | Value-based | Neural Q-function, replay buffer, target net | Medium |
| 3 | REINFORCE | Policy gradient | Policy gradient theorem, Monte Carlo returns | Medium |
| 4 | Actor-Critic | Hybrid | TD advantage, separate actor & critic | Medium |
| 5 | A2C | Hybrid | N-step returns, entropy bonus, shared network | Medium+ |
| 6 | A3C | Hybrid | Async workers, shared memory, gradient pushing | Hard |
| 7 | PPO | Policy gradient | Clipped objective, GAE, mini-batch updates | Hard |
| 8 | DPO | LLM alignment | Preference optimization (text domain) | Medium |
| 9 | GRPO | LLM alignment | Group-relative advantages (text domain) | Hard |

Start with Q-Learning and work your way down. Each algorithm builds on concepts from the previous ones.

## CLI Commands

### Train
```bash
uv run python scripts/train.py --algo dqn --episodes 1000
uv run python scripts/train.py --algo ppo --episodes 500 --use-solution  # run the reference
```

### Evaluate
```bash
uv run python scripts/evaluate.py --algo dqn --checkpoint results/checkpoints/dqn_final.pt
```

### Compare
```bash
uv run python scripts/compare.py --algos q_learning dqn reinforce a2c ppo
```

### Animate
```bash
uv run python scripts/animate.py --algo ppo --episodes 5
```

### TensorBoard
```bash
tensorboard --logdir results/tensorboard
```

## Visualization Tools

```python
from visualization import plot_training_history, compare_training_curves, create_dashboard
from visualization import animate_agent

# Plot a single algorithm's training
plot_training_history(history, title="DQN Training", save_path="results/plots/dqn.png")

# Compare multiple algorithms
compare_training_curves({"DQN": dqn_hist, "PPO": ppo_hist}, save_path="results/plots/compare.png")

# Full dashboard
create_dashboard({"DQN": dqn_hist, "PPO": ppo_hist}, save_path="results/plots/dashboard.png")

# Record agent as video
animate_agent(agent, n_episodes=3, output_path="results/videos/")
```

## Project Structure

```
rl-learning/
├── environment/          # Wrapped LunarLander with rich state/reward info
├── core/                 # Training loop, evaluator, replay buffer, base agent
├── visualization/        # Plotting, animation, dashboard tools
├── algorithms/           # YOUR IMPLEMENTATIONS (templates with TODOs)
├── solutions/            # Reference implementations (don't peek early!)
├── notebooks/            # Theory notes
├── scripts/              # CLI tools (train, evaluate, compare, animate)
└── results/              # Auto-created: checkpoints, videos, plots
```

## Key Files to Keep Open

- `environment/README.md` — State variables, actions, rewards reference card
- `algorithms/README.md` — Learning order and workflow guide
- `core/base_agent.py` — The interface your agents must implement
