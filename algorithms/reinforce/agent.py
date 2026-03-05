"""
REINFORCE (Monte Carlo Policy Gradient) Agent
===============================================

Key idea: Directly optimize a parameterized policy π(a|s;θ) by performing
gradient ascent on the expected return. Unlike Q-learning/DQN which learn
value functions, REINFORCE learns the policy directly.

The policy gradient theorem gives us:
    ∇J(θ) = E[Σ_t ∇log π(a_t|s_t;θ) * G_t]

where G_t = Σ_{k=t}^T γ^(k-t) * r_k is the discounted return from step t.

This is an on-policy, Monte Carlo method: we must complete an entire episode
before updating, and we use the actual returns (not bootstrapped estimates).

Recommended reading: Sutton & Barto, Chapter 13 (Policy Gradient Methods)

See algorithms/reinforce/network.py for the provided PolicyNetwork.
"""

from __future__ import annotations

import numpy as np
import torch

from core.base_agent import BaseAgent
from .network import PolicyNetwork


class REINFORCEAgent(BaseAgent):
    """REINFORCE agent with optional baseline subtraction.

    Args:
        state_dim: Dimension of observation space (8).
        n_actions: Number of discrete actions (4).
        hidden_dim: Hidden layer size.
        learning_rate: Optimizer learning rate.
        gamma: Discount factor.
    """

    def __init__(
        self,
        state_dim: int = 8,
        n_actions: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
    ):
        # TODO: Store hyperparameters
        # TODO: Create PolicyNetwork(state_dim, n_actions, hidden_dim)
        # TODO: Create optimizer: torch.optim.Adam(...)
        # TODO: Initialize episode storage lists:
        #   self.log_probs = []   # log π(a|s) for each step
        #   self.rewards = []     # reward at each step
        raise NotImplementedError("Implement __init__: create policy network and storage")

    def select_action(self, state: np.ndarray) -> int:
        """Sample action from policy and store the log probability.

        Steps:
        1. Convert state to tensor
        2. Get action distribution from policy network
        3. Sample action from distribution
        4. Store log_prob for later gradient computation
        5. Return action
        """
        # TODO: Implement stochastic action selection
        # state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # action, log_prob = self.policy.get_action(state_tensor)
        # self.log_probs.append(log_prob)
        # return action
        raise NotImplementedError("Implement select_action: sample from policy")

    def update(self, state, action, reward, next_state, done) -> dict:
        """Store reward; if episode is done, compute policy gradient and update.

        REINFORCE requires a complete episode before updating.

        When done:
        1. Compute discounted returns G_t for each timestep
        2. Normalize returns (subtract mean, divide by std) for stability
        3. Compute policy gradient loss: -Σ log_prob * G_t
        4. Backprop and optimize
        5. Clear episode storage

        Returns:
            {"loss": float} when episode ends, {} otherwise
        """
        # TODO: Implement REINFORCE update
        #
        # self.rewards.append(reward)
        #
        # if not done:
        #     return {}
        #
        # # Compute discounted returns
        # returns = []
        # G = 0
        # for r in reversed(self.rewards):
        #     G = r + self.gamma * G
        #     returns.insert(0, G)
        # returns = torch.tensor(returns)
        #
        # # Normalize returns for training stability
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        #
        # # Compute policy gradient loss
        # loss = 0
        # for log_prob, G in zip(self.log_probs, returns):
        #     loss -= log_prob * G  # negative because we do gradient ASCENT
        #
        # # Optimize
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        #
        # # Clear episode storage
        # self.log_probs.clear()
        # self.rewards.clear()
        #
        # return {"loss": loss.item()}
        raise NotImplementedError("Implement update: Monte Carlo policy gradient")

    def save(self, path: str) -> None:
        """Save policy network weights."""
        # TODO: torch.save(self.policy.state_dict(), path)
        raise NotImplementedError("Implement save")

    def load(self, path: str) -> None:
        """Load policy network weights."""
        # TODO: self.policy.load_state_dict(torch.load(path))
        raise NotImplementedError("Implement load")

    def set_eval_mode(self) -> None:
        """Set policy to eval mode (still stochastic but no dropout/batchnorm)."""
        # TODO: self.policy.eval()
        raise NotImplementedError("Implement set_eval_mode")

    def set_train_mode(self) -> None:
        """Set policy back to train mode."""
        # TODO: self.policy.train()
        raise NotImplementedError("Implement set_train_mode")
