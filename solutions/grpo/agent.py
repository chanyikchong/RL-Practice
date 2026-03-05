"""Group Relative Policy Optimization (GRPO) agent for text generation.

GRPO (Shao et al., 2024) is a reinforcement-learning-from-human-feedback
variant that avoids learning a separate value/critic network.  Instead it:

1. Generates a *group* of G responses for each prompt.
2. Scores every response with a reward function (or reward model).
3. Computes **group-relative advantages** -- each response's advantage is
   its reward minus the group mean, divided by the group standard deviation.
   This removes the need for a learned baseline / value function.
4. Optimises a clipped surrogate objective (similar to PPO) using the
   group-relative advantages, plus a KL penalty against a reference model.

Advantage computation
---------------------
For prompt x and responses {y_1, ..., y_G}:

    A_i = (r_i - mean(r)) / (std(r) + eps)

This is the "group-relative" part -- the baseline is just the group
statistics, so no critic network is needed.

Objective (per response)
------------------------
    L_i = min( rho_i * A_i,  clip(rho_i, 1-eps, 1+eps) * A_i )
          - beta * KL(pi_theta || pi_ref)

where rho_i = pi_theta(y_i|x) / pi_old(y_i|x) is the importance ratio.

This file implements GRPO on facebook/opt-125m with simple heuristic
rewards for demonstration purposes.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from core.base_agent import BaseAgent


class GRPOAgent(BaseAgent):
    """GRPO agent that fine-tunes a causal LM with group-relative advantages.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    group_size : int
        Number of responses to sample per prompt (G).
    clip_eps : float
        PPO-style clipping epsilon for the surrogate objective.
    beta : float
        KL penalty coefficient against the reference policy.
    lr : float
        Learning rate for AdamW.
    max_length : int
        Maximum token length for generation / encoding.
    device : str | None
        Torch device string (auto-detects CUDA when None).
    """

    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        group_size: int = 4,
        clip_eps: float = 0.2,
        beta: float = 0.04,
        lr: float = 5e-5,
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.group_size = group_size
        self.clip_eps = clip_eps
        self.beta = beta
        self.lr = lr
        self.max_length = max_length

        # ---- Tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Policy model (trainable) ----
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(
            self.device
        )

        # ---- Reference model (frozen) ----
        # Anchors the KL penalty so the policy does not collapse.
        self.ref_model = copy.deepcopy(self.policy_model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # ---- Optimizer ----
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_group(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
    ) -> list[str]:
        """Sample ``self.group_size`` diverse completions for a single prompt.

        Parameters
        ----------
        prompt : str
            The input text.
        max_new_tokens : int
            Maximum tokens to generate per response.
        temperature : float
            Sampling temperature (higher -> more diverse).

        Returns
        -------
        list[str]
            Group of G completions (prompt text stripped).
        """
        self.policy_model.eval()
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = enc["input_ids"].shape[1]

        completions: list[str] = []
        with torch.no_grad():
            for _ in range(self.group_size):
                output_ids = self.policy_model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                text = self.tokenizer.decode(
                    output_ids[0][prompt_len:], skip_special_tokens=True
                )
                completions.append(text)

        return completions

    # ------------------------------------------------------------------
    # Reward heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_rewards(
        prompt: str, completions: list[str]
    ) -> list[float]:
        """Score each completion with simple heuristic rewards.

        In a real system this would be a learned reward model or human
        ratings.  Here we use lightweight proxies:

        1. **Length reward** -- prefer responses that are neither too short
           nor too long (target ~20-60 words).
        2. **Relevance reward** -- fraction of prompt content words that
           also appear in the completion (crude overlap measure).
        3. **Fluency penalty** -- penalise excessive repetition.

        Each sub-reward is in [0, 1]; the total is their average.

        Parameters
        ----------
        prompt : str
            The input prompt.
        completions : list[str]
            The G completions to score.

        Returns
        -------
        list[float]
            Reward for each completion.
        """
        # Extract content words from the prompt (lowercase, len >= 3).
        prompt_words = {
            w.lower()
            for w in prompt.split()
            if len(w) >= 3 and w.isalpha()
        }

        rewards: list[float] = []
        for comp in completions:
            words = comp.split()
            n_words = len(words)

            # 1. Length reward: Gaussian-ish bump centred at 40 words.
            length_reward = float(np.exp(-0.5 * ((n_words - 40) / 20) ** 2))

            # 2. Relevance reward: word overlap with prompt.
            if prompt_words:
                comp_words = {w.lower() for w in words if w.isalpha()}
                relevance_reward = len(prompt_words & comp_words) / len(prompt_words)
            else:
                relevance_reward = 0.0

            # 3. Fluency penalty: fraction of unique bigrams.
            if n_words > 1:
                bigrams = [(words[i], words[i + 1]) for i in range(n_words - 1)]
                unique_ratio = len(set(bigrams)) / len(bigrams)
                fluency_reward = unique_ratio  # 1.0 = all unique, 0.0 = all repeated
            else:
                fluency_reward = 0.5

            total = (length_reward + relevance_reward + fluency_reward) / 3.0
            rewards.append(total)

        return rewards

    # ------------------------------------------------------------------
    # Log-probability computation
    # ------------------------------------------------------------------

    def _sequence_logprobs(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int,
    ) -> torch.Tensor:
        """Compute total log P(completion | prompt) for a batch.

        Parameters
        ----------
        model : AutoModelForCausalLM
        input_ids : Tensor [B, L]
        attention_mask : Tensor [B, L]
        prompt_length : int
            Number of tokens that belong to the prompt (shared across batch).

        Returns
        -------
        Tensor [B]  -- sum of log-probs over completion tokens.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, L, V]

        # Shift for next-token prediction.
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logps = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [B, L-1]

        # Mask: only count completion tokens (positions >= prompt_length).
        mask = torch.zeros_like(token_logps)
        # shift_labels index is offset by 1, so completion starts at prompt_length-1.
        comp_start = max(prompt_length - 1, 0)
        mask[:, comp_start:] = 1.0
        # Also zero out padding.
        mask = mask * attention_mask[:, 1:]

        return (token_logps * mask).sum(dim=-1)  # [B]

    # ------------------------------------------------------------------
    # GRPO loss
    # ------------------------------------------------------------------

    def compute_grpo_loss(
        self,
        prompt: str,
        completions: list[str],
        rewards: list[float],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the GRPO clipped surrogate loss for one prompt group.

        Steps
        -----
        1. Compute group-relative advantages:
               A_i = (r_i - mean(r)) / (std(r) + eps)
        2. For each response compute the importance ratio:
               rho_i = exp( logp_policy - logp_old )
           Here the "old" policy is the policy *before* this gradient step,
           which for a single-step update equals the current policy
           (rho = 1).  For multi-step inner loops you would store the old
           log-probs.  We use the reference model as a stand-in for the
           old policy for simplicity in this educational implementation.
        3. Clipped surrogate:
               L_clip = min(rho * A, clip(rho, 1-eps, 1+eps) * A)
        4. KL penalty:
               KL_i = logp_policy - logp_ref
           (approximated per-sequence).
        5. Final loss (maximise):
               L = mean( L_clip - beta * KL )
           We *minimise* the negative of this.

        Parameters
        ----------
        prompt : str
        completions : list[str]   length G
        rewards : list[float]     length G

        Returns
        -------
        loss : Tensor (scalar)
        metrics : dict
        """
        G = len(completions)
        assert G == len(rewards)

        # ---- Group-relative advantage ----
        # This is the key GRPO idea: normalise rewards within the group
        # so the mean response is the baseline.  Responses better than
        # the group average get positive advantage; worse ones get negative.
        r = np.array(rewards, dtype=np.float64)
        r_mean = r.mean()
        r_std = r.std() + 1e-8  # avoid division by zero
        advantages = (r - r_mean) / r_std  # [G]
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # ---- Tokenize all (prompt + completion) pairs ----
        full_texts = [prompt + c for c in completions]
        enc = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        prompt_enc = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length
        )
        prompt_length = len(prompt_enc["input_ids"])

        input_ids = enc["input_ids"]       # [G, L]
        attention_mask = enc["attention_mask"]

        # ---- Policy log-probs ----
        self.policy_model.train()
        policy_logps = self._sequence_logprobs(
            self.policy_model, input_ids, attention_mask, prompt_length
        )  # [G]

        # ---- Reference (old) log-probs ----
        with torch.no_grad():
            ref_logps = self._sequence_logprobs(
                self.ref_model, input_ids, attention_mask, prompt_length
            )  # [G]

        # ---- Importance ratio ----
        # rho = pi_theta / pi_old;  in log space: log rho = logp_policy - logp_ref
        log_rho = policy_logps - ref_logps
        rho = torch.exp(log_rho)  # [G]

        # ---- Clipped surrogate objective ----
        # Unclipped term: rho * A
        surr1 = rho * adv_tensor
        # Clipped term: clip(rho, 1-eps, 1+eps) * A
        surr2 = torch.clamp(rho, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_tensor
        # Take the minimum (pessimistic bound, as in PPO).
        clipped_obj = torch.min(surr1, surr2)

        # ---- KL penalty ----
        # Approximate per-sequence KL divergence.
        # KL(pi_theta || pi_ref) ~ logp_policy - logp_ref  (first-order approx)
        kl = log_rho  # [G]

        # ---- Total objective (we want to maximise, so negate for loss) ----
        objective = clipped_obj - self.beta * kl  # [G]
        loss = -objective.mean()

        # ---- Metrics ----
        with torch.no_grad():
            metrics = {
                "grpo_loss": loss.item(),
                "mean_reward": float(r_mean),
                "std_reward": float(r.std()),
                "mean_advantage": float(advantages.mean()),
                "mean_kl": float(kl.mean().item()),
                "mean_rho": float(rho.mean().item()),
            }

        return loss, metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_on_prompts(
        self,
        prompts: list[str],
        epochs: int = 3,
        max_new_tokens: int = 64,
    ) -> list[dict[str, float]]:
        """Main GRPO training loop.

        For each prompt:
        1. Generate a group of G responses.
        2. Score them with the reward function.
        3. Compute group-relative advantages and the clipped loss.
        4. Update the policy.

        Parameters
        ----------
        prompts : list[str]
            Training prompts.
        epochs : int
            Number of passes over the prompt set.
        max_new_tokens : int
            Max tokens per generated response.

        Returns
        -------
        history : list[dict]
            Per-step metrics.
        """
        history: list[dict[str, float]] = []

        for epoch in range(epochs):
            indices = np.random.permutation(len(prompts))
            for idx in indices:
                prompt = prompts[idx]

                # Step 1 -- Generate a group of responses.
                completions = self.generate_group(
                    prompt, max_new_tokens=max_new_tokens
                )

                # Step 2 -- Score every response.
                rewards = self.compute_rewards(prompt, completions)

                # Step 3 + 4 -- Compute loss and update.
                loss, metrics = self.compute_grpo_loss(prompt, completions, rewards)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), max_norm=1.0
                )
                self.optimizer.step()

                metrics["epoch"] = epoch
                history.append(metrics)

        return history

    # ------------------------------------------------------------------
    # Text generation (post-training)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
    ) -> str:
        """Generate a single completion from the trained policy."""
        self.policy_model.eval()
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = self.policy_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # BaseAgent interface stubs
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """Stub -- GRPO operates on text, not discrete-action MDPs."""
        return 0

    def update(self, *args, **kwargs) -> dict:
        """Stub -- use :meth:`train_on_prompts` for GRPO training."""
        return {}

    def save(self, path: str) -> None:
        """Persist the policy model and tokenizer."""
        self.policy_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        """Restore a saved policy model."""
        self.policy_model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)


# ----------------------------------------------------------------------
# Quick demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    prompts = [
        "Explain why the sky is blue in simple terms: ",
        "Write a short product description for a reusable water bottle: ",
        "Summarize the benefits of regular exercise: ",
        "Describe what machine learning is to a beginner: ",
    ]

    print("Initialising GRPO agent (facebook/opt-125m) ...")
    agent = GRPOAgent(device="cpu", group_size=4)

    print(f"Training on {len(prompts)} prompts with group_size={agent.group_size} ...")
    history = agent.train_on_prompts(prompts, epochs=1, max_new_tokens=48)

    for step, m in enumerate(history):
        print(
            f"  step {step}: loss={m['grpo_loss']:.4f}  "
            f"reward={m['mean_reward']:.3f}  "
            f"kl={m['mean_kl']:.4f}  "
            f"rho={m['mean_rho']:.3f}"
        )

    print("\nGenerating a response ...")
    out = agent.generate("Explain gravity in one sentence: ")
    print(f"  -> {out}")
