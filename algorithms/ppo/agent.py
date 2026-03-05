"""
Proximal Policy Optimization (PPO) Agent
==========================================

Key idea: PPO improves on vanilla policy gradient by preventing large
policy updates that can destabilize training. It uses a clipped surrogate
objective that limits how much the policy can change in each update.

The clipped objective:
    L_CLIP = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)

where r(θ) = π_new(a|s) / π_old(a|s) is the probability ratio.

PPO collects a batch of experience, then performs multiple epochs of
mini-batch updates on that data (reusing data efficiently).

Recommended reading: Schulman et al. 2017 "Proximal Policy Optimization"

See algorithms/ppo/network.py for PPOActorCriticNetwork with evaluate_action().
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core.base_agent import BaseAgent
from .network import PPOActorCriticNetwork


class PPOAgent(BaseAgent):
    """PPO agent with clipped objective and mini-batch updates.

    Args:
        state_dim: Dimension of observation space (8).
        n_actions: Number of discrete actions (4).
        hidden_dim: Hidden layer size.
        learning_rate: Optimizer learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_epsilon: PPO clipping parameter (typically 0.2).
        n_steps: Steps to collect before each update.
        n_epochs: Number of optimization epochs per batch.
        mini_batch_size: Size of mini-batches within each epoch.
        value_coeff: Weight for critic loss.
        entropy_coeff: Weight for entropy bonus.
        max_grad_norm: Maximum gradient norm for clipping.
    """

    def __init__(
        self,
        state_dim: int = 8,
        n_actions: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        n_steps: int = 128,
        n_epochs: int = 4,
        mini_batch_size: int = 32,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        # TODO: Store all hyperparameters
        # TODO: Create PPOActorCriticNetwork
        # TODO: Create optimizer
        # TODO: Initialize rollout storage:
        #   self.states = []
        #   self.actions = []
        #   self.rewards = []
        #   self.log_probs = []  # old log probs (before update)
        #   self.values = []
        #   self.dones = []
        raise NotImplementedError("Implement __init__: create network and storage")

    def select_action(self, state: np.ndarray) -> int:
        """Sample action and store old log probability for later ratio computation."""
        # TODO: Implement action selection
        # state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # with torch.no_grad():
        #     action, log_prob, value = self.network.get_action_and_value(state_tensor)
        # self.states.append(state)
        # self.actions.append(action)
        # self.log_probs.append(log_prob)
        # self.values.append(value.squeeze())
        # return action
        raise NotImplementedError("Implement select_action")

    def update(self, state, action, reward, next_state, done) -> dict:
        """Store transition; perform PPO update every N steps.

        When updating:
        1. Compute GAE advantages and returns
        2. For each epoch:
           a. Shuffle data into mini-batches
           b. For each mini-batch:
              - Compute new log probs and values
              - Compute probability ratio: r = exp(new_log_prob - old_log_prob)
              - Clipped surrogate loss:
                surr1 = r * advantage
                surr2 = clip(r, 1-ε, 1+ε) * advantage
                actor_loss = -min(surr1, surr2).mean()
              - Value loss: MSE(new_value, return)
              - Entropy bonus
              - Total loss and backprop
        3. Clear storage

        Returns:
            {"loss": float, "actor_loss": float, "critic_loss": float,
             "entropy": float, "clip_fraction": float} when updating, {} otherwise
        """
        # TODO: Implement PPO update
        #
        # self.rewards.append(reward)
        # self.dones.append(done)
        #
        # if len(self.states) < self.n_steps and not done:
        #     return {}
        #
        # # Compute GAE
        # with torch.no_grad():
        #     if done:
        #         next_value = 0.0
        #     else:
        #         _, nv = self.network(torch.FloatTensor(next_state).unsqueeze(0))
        #         next_value = nv.squeeze().item()
        #
        # # GAE computation
        # advantages = []
        # gae = 0
        # values_list = [v.item() for v in self.values] + [next_value]
        # for t in reversed(range(len(self.rewards))):
        #     delta = self.rewards[t] + self.gamma * values_list[t+1] * (1 - float(self.dones[t])) - values_list[t]
        #     gae = delta + self.gamma * self.gae_lambda * (1 - float(self.dones[t])) * gae
        #     advantages.insert(0, gae)
        #
        # advantages = torch.tensor(advantages, dtype=torch.float32)
        # returns = advantages + torch.tensor(values_list[:-1], dtype=torch.float32)
        #
        # # Prepare batch tensors
        # states_t = torch.FloatTensor(np.array(self.states))
        # actions_t = torch.tensor(self.actions, dtype=torch.long)
        # old_log_probs_t = torch.stack(self.log_probs)
        #
        # # Normalize advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        #
        # # PPO epochs
        # total_loss = 0
        # clip_fracs = []
        # for epoch in range(self.n_epochs):
        #     # Shuffle indices
        #     indices = torch.randperm(len(states_t))
        #     for start in range(0, len(indices), self.mini_batch_size):
        #         mb_idx = indices[start:start + self.mini_batch_size]
        #
        #         mb_states = states_t[mb_idx]
        #         mb_actions = actions_t[mb_idx]
        #         mb_old_log_probs = old_log_probs_t[mb_idx]
        #         mb_advantages = advantages[mb_idx]
        #         mb_returns = returns[mb_idx]
        #
        #         # New forward pass
        #         new_log_probs, new_values, entropy = self.network.evaluate_action(mb_states, mb_actions)
        #
        #         # Ratio
        #         ratio = torch.exp(new_log_probs - mb_old_log_probs.detach())
        #
        #         # Clipped surrogate
        #         surr1 = ratio * mb_advantages
        #         surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
        #         actor_loss = -torch.min(surr1, surr2).mean()
        #
        #         # Value loss
        #         critic_loss = F.mse_loss(new_values.squeeze(), mb_returns)
        #
        #         # Total loss
        #         loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy.mean()
        #
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        #         self.optimizer.step()
        #
        #         total_loss += loss.item()
        #         clip_fracs.append(((ratio - 1).abs() > self.clip_epsilon).float().mean().item())
        #
        # # Clear storage
        # self.states.clear(); self.actions.clear(); self.rewards.clear()
        # self.log_probs.clear(); self.values.clear(); self.dones.clear()
        #
        # return {"loss": total_loss, "clip_fraction": np.mean(clip_fracs) if clip_fracs else 0}
        raise NotImplementedError("Implement update: PPO clipped objective")

    def save(self, path: str) -> None:
        # TODO: torch.save(self.network.state_dict(), path)
        raise NotImplementedError("Implement save")

    def load(self, path: str) -> None:
        # TODO: self.network.load_state_dict(torch.load(path))
        raise NotImplementedError("Implement load")

    def set_eval_mode(self) -> None:
        # TODO: self.network.eval()
        raise NotImplementedError("Implement set_eval_mode")

    def set_train_mode(self) -> None:
        # TODO: self.network.train()
        raise NotImplementedError("Implement set_train_mode")
