# Algorithm Implementations

This is where YOU implement each RL algorithm! Each folder contains:

- `agent.py` — A template with `TODO` comments guiding you through the implementation
- `network.py` — Pre-built neural network architectures (provided, don't modify)

## Recommended Learning Order

| # | Algorithm | Type | Difficulty | Key Concept |
|---|-----------|------|-----------|-------------|
| 1 | **Q-Learning** | Value-based | Easy | Bellman equation, Q-table, epsilon-greedy |
| 2 | **DQN** | Value-based | Medium | Neural Q-function, experience replay, target network |
| 3 | **REINFORCE** | Policy gradient | Medium | Policy gradient theorem, Monte Carlo returns |
| 4 | **Actor-Critic** | Hybrid | Medium | TD advantage, separate actor/critic |
| 5 | **A2C** | Hybrid | Medium+ | N-step returns, entropy bonus, shared network |
| 6 | **A3C** | Hybrid | Hard | Asynchronous workers, shared memory, gradient pushing |
| 7 | **PPO** | Policy gradient | Hard | Clipped objective, GAE, mini-batch updates |
| 8 | **DPO** | LLM alignment | Medium | Preference optimization (different paradigm!) |
| 9 | **GRPO** | LLM alignment | Hard | Group-relative advantages (different paradigm!) |

## Workflow for Each Algorithm

1. Read the docstring at the top of `agent.py` — it explains the key idea
2. Study the `TODO` comments — they walk you through each method
3. Implement each method one at a time
4. Test with: `uv run python scripts/train.py --algo <name> --episodes 500`
5. If stuck, peek at the corresponding `solutions/<name>/agent.py`

## Tips

- **Keep `environment/README.md` open** — it has the state/action/reward reference
- Start with small episode counts (100-200) to check your code runs
- Use TensorBoard to monitor training: `tensorboard --logdir results/tensorboard`
- The `core/trainer.py` handles the training loop — you only write the agent
- Q-Learning will be slow to converge (continuous state discretization is lossy)
- DQN and beyond should solve LunarLander (avg reward > 200) within 500-1000 episodes
