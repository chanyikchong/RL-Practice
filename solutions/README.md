# Reference Solutions

**Don't peek until you've tried implementing the algorithm yourself!**

Each solution is a complete, working implementation that can solve LunarLander-v2.
They use the same class names and interfaces as the templates, so you can
drop them in as replacements.

## How to Use a Solution

```bash
# Run training with a solution instead of your implementation
uv run python scripts/train.py --algo dqn --use-solution --episodes 1000
```

Or copy a solution to compare side-by-side with your implementation.

## Expected Performance

| Algorithm | Episodes to Solve | Final Avg Reward | Notes |
|-----------|-------------------|------------------|-------|
| Q-Learning | ~2000+ | ~150-200 | Limited by discretization |
| DQN | ~400-600 | 250+ | Reliable with target net |
| REINFORCE | ~800-1200 | 200+ | High variance |
| Actor-Critic | ~600-900 | 220+ | Lower variance than REINFORCE |
| A2C | ~400-700 | 240+ | Fast, stable |
| A3C | ~300-500 | 240+ | Fastest wall-clock time |
| PPO | ~400-600 | 260+ | Most robust |
