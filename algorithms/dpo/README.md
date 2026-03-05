# DPO — Direct Preference Optimization

## Different Paradigm!

DPO is **not** a control algorithm like the others in this project. It's designed
for **aligning language models** to human preferences. We include it here because
understanding the RL→LLM alignment pipeline is essential for modern ML.

## How It Differs

| Aspect | Traditional RL (LunarLander) | DPO (LLM Alignment) |
|--------|------------------------------|----------------------|
| State | 8-dim float vector | Text prompt |
| Action | 4 discrete options | Token generation |
| Reward | Numeric score per step | Preference pairs (chosen vs rejected) |
| Goal | Maximize cumulative reward | Match human preferences |

## The Key Insight

Standard RLHF pipeline:
1. Train reward model from preference data
2. Use PPO to optimize LM against reward model

DPO shortcut:
1. Directly optimize LM from preference data (no reward model needed!)

The DPO loss implicitly learns the reward and optimizes the policy simultaneously.

## Implementation

This mini-project uses:
- **Model**: `facebook/opt-125m` (small, trainable on a single GPU)
- **Task**: Text summarization preferences
- **Dataset**: A small preference dataset

The agent template guides you through implementing the DPO loss function and
training loop.
