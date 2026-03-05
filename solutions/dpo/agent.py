"""Direct Preference Optimization (DPO) agent for text summarization.

DPO (Rafailov et al., 2023) turns the RLHF objective into a simple
classification loss over preference pairs, eliminating the need for a
separate reward model and PPO training loop.

Key insight
-----------
The optimal policy under a KL-constrained reward-maximization objective
satisfies:

    r(x, y) = beta * log[ pi(y|x) / pi_ref(y|x) ] + const

So instead of fitting a reward model and then running RL, we can directly
optimize the policy with a binary cross-entropy style loss over
(chosen, rejected) pairs:

    L_DPO = -E[ log sigmoid( beta * (log_ratio_chosen - log_ratio_rejected) ) ]

where log_ratio = log pi_theta(y|x) - log pi_ref(y|x).

This file implements DPO on a small causal LM (facebook/opt-125m) for a
text-summarization preference task.
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


class DPOAgent(BaseAgent):
    """DPO agent that fine-tunes a causal language model from preference data.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: ``"facebook/opt-125m"``.
    beta : float
        KL penalty coefficient. Controls how far the policy can drift from
        the reference model.  Higher beta -> more conservative updates.
    lr : float
        Learning rate for AdamW.
    max_length : int
        Maximum token length for inputs.
    device : str
        Torch device string.
    """

    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        beta: float = 0.1,
        lr: float = 5e-5,
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        # ---- Device selection ----
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.beta = beta
        self.lr = lr
        self.max_length = max_length

        # ---- Load tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # OPT does not ship with a pad token; reuse the EOS token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Policy model (the one we train) ----
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(
            self.device
        )

        # ---- Reference model (frozen copy) ----
        # The reference model is a snapshot of the policy at initialisation.
        # It is never updated; it anchors the KL constraint.
        self.ref_model = copy.deepcopy(self.policy_model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # ---- Optimizer ----
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # Core DPO helpers
    # ------------------------------------------------------------------

    def _get_per_token_logps(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log-probabilities for *completion* tokens.

        Parameters
        ----------
        model : AutoModelForCausalLM
            Either the policy or reference model.
        input_ids : Tensor [B, L]
            Full sequence (prompt + completion).
        attention_mask : Tensor [B, L]
            1 for real tokens, 0 for padding.
        labels : Tensor [B, L]
            Token ids where prompt positions are set to -100 so they are
            ignored when computing the log-probability of the completion.

        Returns
        -------
        Tensor [B]
            Sum of log-probabilities over completion tokens for each item
            in the batch.
        """
        with torch.no_grad() if not model.training else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # logits shape: [B, L, V]
            logits = outputs.logits

        # Shift logits and labels so that token t predicts token t+1.
        # logits[:, :-1]  ->  predictions for positions 1..L
        # labels[:, 1:]   ->  ground-truth tokens at positions 1..L
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # Per-token log-probabilities via log-softmax + gather.
        log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, L-1, V]
        gathered = log_probs.gather(
            dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)  # [B, L-1]

        # Mask out prompt tokens (label == -100) and padding.
        loss_mask = (shift_labels != -100).float()
        gathered = gathered * loss_mask

        # Sum over the completion length to get total log P(completion | prompt).
        return gathered.sum(dim=-1)  # [B]

    def _tokenize_pair(
        self, prompts: list[str], completions: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize prompt+completion and create labels with prompt masked.

        Returns (input_ids, attention_mask, labels) all on self.device.
        """
        # Build full text sequences.
        full_texts = [p + c for p, c in zip(prompts, completions)]

        enc = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Build labels: copy of input_ids, but set prompt tokens to -100.
        labels = input_ids.clone()
        for i, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer(
                prompt, truncation=True, max_length=self.max_length
            )["input_ids"]
            prompt_len = len(prompt_ids)
            labels[i, :prompt_len] = -100

        # Also mask padding tokens.
        labels[attention_mask == 0] = -100

        return input_ids, attention_mask, labels

    def compute_dpo_loss(
        self,
        prompts: list[str],
        chosen_completions: list[str],
        rejected_completions: list[str],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the DPO loss for a batch of preference pairs.

        The DPO loss is:
            L = -log sigmoid( beta * (log_ratio_chosen - log_ratio_rejected) )

        where:
            log_ratio_w = log pi_theta(y_w | x) - log pi_ref(y_w | x)

        This encourages the policy to increase the likelihood of chosen
        completions relative to rejected ones, while staying close to the
        reference model (controlled by beta).

        Parameters
        ----------
        prompts : list[str]
            The input prompts / contexts.
        chosen_completions : list[str]
            Human-preferred completions (y_w).
        rejected_completions : list[str]
            Non-preferred completions (y_l).

        Returns
        -------
        loss : Tensor (scalar)
        metrics : dict
        """
        # ---- Tokenize chosen and rejected pairs ----
        c_ids, c_mask, c_labels = self._tokenize_pair(prompts, chosen_completions)
        r_ids, r_mask, r_labels = self._tokenize_pair(prompts, rejected_completions)

        # ---- Policy log-probs ----
        self.policy_model.train()
        policy_logp_chosen = self._get_per_token_logps(
            self.policy_model, c_ids, c_mask, c_labels
        )
        policy_logp_rejected = self._get_per_token_logps(
            self.policy_model, r_ids, r_mask, r_labels
        )

        # ---- Reference log-probs (no gradient) ----
        with torch.no_grad():
            ref_logp_chosen = self._get_per_token_logps(
                self.ref_model, c_ids, c_mask, c_labels
            )
            ref_logp_rejected = self._get_per_token_logps(
                self.ref_model, r_ids, r_mask, r_labels
            )

        # ---- Log-ratios ----
        # log_ratio = log pi_theta(y|x) - log pi_ref(y|x)
        # Positive log_ratio means the policy assigns *more* probability
        # than the reference.
        log_ratio_chosen = policy_logp_chosen - ref_logp_chosen
        log_ratio_rejected = policy_logp_rejected - ref_logp_rejected

        # ---- DPO loss ----
        # We want log_ratio_chosen > log_ratio_rejected, i.e. the policy
        # should prefer the chosen completion *more* than the reference does.
        # The logistic loss converts this margin into a smooth 0-1 objective.
        logits = self.beta * (log_ratio_chosen - log_ratio_rejected)
        loss = -F.logsigmoid(logits).mean()

        # ---- Metrics for logging ----
        with torch.no_grad():
            reward_margin = (log_ratio_chosen - log_ratio_rejected).mean().item()
            accuracy = (logits > 0).float().mean().item()

        metrics = {
            "dpo_loss": loss.item(),
            "reward_margin": reward_margin,
            "accuracy": accuracy,
        }
        return loss, metrics

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_preferences(
        self,
        preference_data: list[dict[str, str]],
        epochs: int = 3,
        batch_size: int = 2,
    ) -> list[dict[str, float]]:
        """Train the policy model on a preference dataset.

        Parameters
        ----------
        preference_data : list[dict]
            Each dict has keys: ``"prompt"``, ``"chosen"``, ``"rejected"``.
        epochs : int
            Number of passes through the dataset.
        batch_size : int
            Mini-batch size.

        Returns
        -------
        history : list[dict]
            Per-step training metrics.
        """
        history: list[dict[str, float]] = []
        n = len(preference_data)

        for epoch in range(epochs):
            # Shuffle at the start of each epoch.
            indices = np.random.permutation(n)

            for start in range(0, n, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch = [preference_data[i] for i in batch_idx]

                prompts = [item["prompt"] for item in batch]
                chosen = [item["chosen"] for item in batch]
                rejected = [item["rejected"] for item in batch]

                # Forward + loss
                loss, metrics = self.compute_dpo_loss(prompts, chosen, rejected)

                # Backward + step
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
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from the fine-tuned policy model.

        Parameters
        ----------
        prompt : str
            Input text / context.
        max_new_tokens : int
            Maximum number of tokens to generate.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus sampling threshold.

        Returns
        -------
        str
            The generated completion (prompt stripped).
        """
        self.policy_model.eval()
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.policy_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens.
        generated = self.tokenizer.decode(
            output_ids[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return generated

    # ------------------------------------------------------------------
    # BaseAgent interface stubs
    # ------------------------------------------------------------------
    # DPO operates in the text/preference domain rather than the typical
    # state-action MDP loop, so select_action and update are thin stubs.

    def select_action(self, state: np.ndarray) -> int:
        """Stub -- DPO is not used in discrete-action MDPs.

        For text generation use :meth:`generate` instead.
        """
        return 0

    def update(self, *args, **kwargs) -> dict:
        """Stub -- use :meth:`train_on_preferences` for DPO training."""
        return {}

    def save(self, path: str) -> None:
        """Save the policy model and tokenizer to *path*."""
        self.policy_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        """Load a previously saved policy model from *path*."""
        self.policy_model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)


# ----------------------------------------------------------------------
# Quick demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # A tiny synthetic preference dataset for demonstration.
    # In practice you would load a real dataset (e.g. from HuggingFace).
    preference_data = [
        {
            "prompt": "Summarize: The cat sat on the mat. It was warm and sunny. ",
            "chosen": "A cat rested comfortably on a mat in warm weather.",
            "rejected": "Cat mat warm sunny sat.",
        },
        {
            "prompt": "Summarize: The stock market rose 2% today on strong earnings. ",
            "chosen": "Stocks climbed 2% driven by positive earnings reports.",
            "rejected": "Stocks went up a lot today because of stuff.",
        },
        {
            "prompt": "Summarize: Researchers discovered a new species of frog in the Amazon. ",
            "chosen": "A new frog species was found in the Amazon rainforest.",
            "rejected": "Frog. Amazon. New. Discovered.",
        },
        {
            "prompt": "Summarize: The city council approved a new park in the downtown area. ",
            "chosen": "City council greenlit a downtown park project.",
            "rejected": "Council did park thing downtown area approved yes.",
        },
    ]

    print("Initialising DPO agent (facebook/opt-125m) ...")
    agent = DPOAgent(device="cpu")

    print("Training on preference pairs ...")
    history = agent.train_on_preferences(preference_data, epochs=2, batch_size=2)
    for step, m in enumerate(history):
        print(
            f"  step {step}: loss={m['dpo_loss']:.4f}  "
            f"margin={m['reward_margin']:.4f}  acc={m['accuracy']:.2f}"
        )

    print("\nGenerating a summary ...")
    output = agent.generate(
        "Summarize: Heavy rain caused flooding in several coastal towns. "
    )
    print(f"  -> {output}")
