"""
Group Relative Policy Optimization (GRPO) Agent
==================================================

DIFFERENT PARADIGM: Like DPO, GRPO is for LLM alignment, not control tasks.

Key idea: GRPO generates multiple responses per prompt, scores them using
a reward model (or rule-based rewards), and uses the group's relative
rankings to compute advantages. This avoids needing a separate critic
network (unlike PPO for LLMs).

For each prompt x, GRPO:
1. Generates G responses {y_1, ..., y_G} from the current policy
2. Scores each response with a reward function: {r_1, ..., r_G}
3. Computes group-relative advantages:
   A_i = (r_i - mean(r)) / std(r)
4. Updates policy with a clipped objective (like PPO):
   L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) + β * KL(π || π_ref)

Recommended reading: Shao et al. 2024 "DeepSeekMath" (GRPO section)

See algorithms/grpo/README.md for more context.
"""

from __future__ import annotations

import numpy as np
import torch

from core.base_agent import BaseAgent


class GRPOAgent(BaseAgent):
    """GRPO agent for LLM alignment (text domain, not LunarLander).

    Args:
        model_name: HuggingFace model name.
        group_size: Number of responses to generate per prompt.
        clip_epsilon: PPO-style clipping parameter.
        beta: KL penalty coefficient.
        learning_rate: Optimizer learning rate.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        group_size: int = 4,
        clip_epsilon: float = 0.2,
        beta: float = 0.04,
        learning_rate: float = 5e-5,
        max_length: int = 128,
    ):
        # TODO: Store hyperparameters
        # TODO: Load tokenizer and model
        # TODO: Create a frozen reference model (same as DPO)
        # TODO: Create optimizer
        # TODO: Define or load a reward function for scoring responses
        #   For the learning exercise, use a simple length-based or
        #   keyword-matching reward (see solutions for a real example)
        raise NotImplementedError("Implement __init__: load models and reward function")

    def generate_group(self, prompt: str) -> list[str]:
        """Generate group_size responses for a given prompt.

        Uses sampling (temperature > 0) to get diverse responses.
        """
        # TODO: Generate self.group_size different completions
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # responses = []
        # for _ in range(self.group_size):
        #     output = self.model.generate(
        #         **inputs, max_new_tokens=50,
        #         do_sample=True, temperature=0.8, top_p=0.9
        #     )
        #     responses.append(self.tokenizer.decode(output[0], skip_special_tokens=True))
        # return responses
        raise NotImplementedError("Implement generate_group")

    def compute_rewards(self, prompt: str, responses: list[str]) -> list[float]:
        """Score each response using the reward function.

        For the learning exercise, implement a simple reward like:
        - Length penalty (prefer concise responses)
        - Relevance bonus (contains key terms from prompt)
        - Fluency heuristic
        """
        # TODO: Implement reward scoring
        raise NotImplementedError("Implement compute_rewards")

    def compute_grpo_loss(
        self,
        prompt: str,
        responses: list[str],
        rewards: list[float],
    ) -> torch.Tensor:
        """Compute GRPO loss using group-relative advantages.

        Steps:
        1. Compute group-relative advantages: A_i = (r_i - mean(r)) / (std(r) + eps)
        2. Get log probs from policy and reference models
        3. Compute probability ratios
        4. Clipped surrogate objective (like PPO)
        5. KL penalty between policy and reference
        6. Total loss = clipped_loss + beta * KL
        """
        # TODO: Implement GRPO loss
        #
        # rewards_t = torch.tensor(rewards)
        # advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
        #
        # total_loss = 0
        # for response, advantage in zip(responses, advantages):
        #     # Tokenize
        #     tokens = self.tokenizer(response, return_tensors="pt", truncation=True)
        #
        #     # Get log probs from policy and reference
        #     # ... (similar to DPO log prob computation)
        #
        #     # Ratio and clipped objective
        #     # ratio = exp(new_log_prob - old_log_prob)
        #     # surr1 = ratio * advantage
        #     # surr2 = clip(ratio, 1-eps, 1+eps) * advantage
        #     # loss = -min(surr1, surr2)
        #
        #     # KL penalty
        #     # kl = policy_log_prob - ref_log_prob
        #
        #     # total_loss += loss + beta * kl
        #
        # return total_loss / len(responses)
        raise NotImplementedError("Implement compute_grpo_loss")

    def train_on_prompts(self, prompts: list[str], n_iterations: int = 100) -> list[float]:
        """Main training loop for GRPO.

        For each iteration:
        1. Sample a prompt
        2. Generate a group of responses
        3. Score with reward function
        4. Compute GRPO loss and update

        Returns:
            List of losses per iteration.
        """
        # TODO: Implement GRPO training loop
        raise NotImplementedError("Implement train_on_prompts")

    # BaseAgent interface (compatibility stubs)
    def select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError("GRPO operates on text, not discrete actions")

    def update(self, *args, **kwargs) -> dict:
        return {}

    def save(self, path: str) -> None:
        # TODO: self.model.save_pretrained(path)
        raise NotImplementedError("Implement save")

    def load(self, path: str) -> None:
        # TODO: Load model from path
        raise NotImplementedError("Implement load")
