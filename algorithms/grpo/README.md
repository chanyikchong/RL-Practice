# GRPO — Group Relative Policy Optimization

## Different Paradigm!

Like DPO, GRPO is for **LLM alignment**, not control tasks. It was introduced
by DeepSeek as an alternative to PPO for LLM training.

## How GRPO Works

1. For each prompt, generate **multiple responses** (a "group")
2. Score each response with a reward function
3. Compute **group-relative advantages**: A_i = (r_i - mean) / std
4. Update policy using a PPO-style clipped objective

## Why GRPO Over PPO for LLMs?

| PPO for LLMs | GRPO |
|--------------|------|
| Needs a learned critic (value network) | No critic needed |
| Baseline = V(s) from critic | Baseline = group mean reward |
| Single response per prompt | Multiple responses per prompt |
| More parameters to train | Simpler architecture |

## The Group-Relative Advantage

Instead of learning V(s) to compute advantages, GRPO uses the group itself:
- Generate G responses for each prompt
- Advantage of response i = how much better it is than the group average
- This is a Monte Carlo estimate of the advantage without a learned baseline

## Implementation

This mini-project uses:
- **Model**: `facebook/opt-125m`
- **Task**: Text generation with simple reward functions
- **Group size**: 4 responses per prompt
