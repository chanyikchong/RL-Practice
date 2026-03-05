"""
Direct Preference Optimization (DPO) Agent
=============================================

DIFFERENT PARADIGM: DPO is NOT a traditional RL algorithm for control tasks.
It is used for aligning language models to human preferences.

Key idea: Instead of learning a separate reward model and then doing RL
(like RLHF), DPO directly optimizes the policy using preference pairs
(chosen vs rejected responses). The loss function implicitly defines
a reward and optimizes the policy in closed form:

    L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

where:
- y_w = preferred/chosen response
- y_l = rejected response
- π_ref = reference (initial) model
- β = temperature parameter

This agent works with TEXT, not LunarLander. See algorithms/dpo/README.md.

Recommended reading: Rafailov et al. 2023 "Direct Preference Optimization"
"""

from __future__ import annotations

import numpy as np
import torch

from core.base_agent import BaseAgent


class DPOAgent(BaseAgent):
    """DPO agent for LLM alignment (text domain, not LunarLander).

    This uses a small language model (e.g. facebook/opt-125m) and a
    preference dataset to demonstrate the DPO algorithm.

    Args:
        model_name: HuggingFace model name.
        beta: Temperature parameter for DPO loss.
        learning_rate: Optimizer learning rate.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        beta: float = 0.1,
        learning_rate: float = 5e-5,
        max_length: int = 128,
    ):
        # TODO: Store hyperparameters
        # TODO: Load tokenizer: AutoTokenizer.from_pretrained(model_name)
        # TODO: Load policy model: AutoModelForCausalLM.from_pretrained(model_name)
        # TODO: Load reference model: same, but frozen (no gradients)
        #   ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        #   ref_model.eval()
        #   for p in ref_model.parameters(): p.requires_grad = False
        # TODO: Create optimizer for policy model only
        raise NotImplementedError("Implement __init__: load models and tokenizer")

    def compute_dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the DPO loss for a batch of preference pairs.

        Steps:
        1. Get log probabilities from policy model for chosen and rejected
        2. Get log probabilities from reference model (no grad)
        3. Compute log ratios: log π(y|x) - log π_ref(y|x) for both
        4. DPO loss: -log σ(β * (chosen_ratio - rejected_ratio))
        """
        # TODO: Implement DPO loss computation
        #
        # def get_log_probs(model, input_ids, attention_mask):
        #     outputs = model(input_ids, attention_mask=attention_mask)
        #     logits = outputs.logits[:, :-1]  # shift
        #     labels = input_ids[:, 1:]         # shift
        #     log_probs = F.log_softmax(logits, dim=-1)
        #     token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        #     mask = attention_mask[:, 1:]
        #     return (token_log_probs * mask).sum(-1)
        #
        # # Policy log probs
        # pi_chosen = get_log_probs(self.model, chosen_ids, chosen_mask)
        # pi_rejected = get_log_probs(self.model, rejected_ids, rejected_mask)
        #
        # # Reference log probs (no grad)
        # with torch.no_grad():
        #     ref_chosen = get_log_probs(self.ref_model, chosen_ids, chosen_mask)
        #     ref_rejected = get_log_probs(self.ref_model, rejected_ids, rejected_mask)
        #
        # # Log ratios
        # chosen_ratio = pi_chosen - ref_chosen
        # rejected_ratio = pi_rejected - ref_rejected
        #
        # # DPO loss
        # loss = -F.logsigmoid(self.beta * (chosen_ratio - rejected_ratio)).mean()
        # return loss
        raise NotImplementedError("Implement compute_dpo_loss")

    def train_on_preferences(self, dataset, n_epochs: int = 3, batch_size: int = 4) -> list[float]:
        """Train on a preference dataset.

        This is the main training entry point (replaces the Trainer for DPO).

        Args:
            dataset: HuggingFace dataset with 'chosen' and 'rejected' columns.
            n_epochs: Number of training epochs.
            batch_size: Batch size.

        Returns:
            List of average losses per epoch.
        """
        # TODO: Implement training loop over preference pairs
        raise NotImplementedError("Implement train_on_preferences")

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text using the trained policy model."""
        # TODO: Tokenize prompt and generate
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        raise NotImplementedError("Implement generate")

    # BaseAgent interface (compatibility stubs)
    def select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError("DPO operates on text, not discrete actions")

    def update(self, *args, **kwargs) -> dict:
        return {}

    def save(self, path: str) -> None:
        # TODO: self.model.save_pretrained(path)
        raise NotImplementedError("Implement save")

    def load(self, path: str) -> None:
        # TODO: self.model = AutoModelForCausalLM.from_pretrained(path)
        raise NotImplementedError("Implement load")
