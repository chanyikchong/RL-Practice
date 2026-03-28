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
from torch import optim

from core.base_agent import BaseAgent
from core.replay_episodes import ReplayEpisodes
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
        batch_size: int = 64,
        baseline: str = "mean",
        replay_length: int = 1000,
    ):
        assert baseline in ["mean", "norm", "time"], ValueError("Baseline must be 'mean', 'norm' or 'time'.")
        self.policy_net = PolicyNetwork(state_dim, n_actions, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.batch_size = batch_size
        self.baseline = baseline
        self.states = list()
        self.actions = list()
        self.rewards = list()
        self.replay_episodes = ReplayEpisodes(capacity=replay_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)

    def select_action(self, state: np.ndarray) -> int:
        """Sample action from policy and store the log probability.

        Steps:
        1. Convert state to tensor
        2. Get action distribution from policy network
        3. Sample action from distribution
        4. Store log_prob for later gradient computation
        5. Return action
        """
        # state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # action, log_prob = self.policy.get_action(state_tensor)
        # self.log_probs.append(log_prob)
        # return action
        self.policy_net.eval()
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action(torch.tensor(state, dtype=torch.float, device=self.device))
        return action

    def compute_baseline(self, gts, lengths):
        if self.baseline == "mean":
            return gts - torch.mean(gts)
        elif self.baseline == "norm":
            return (gts - gts.mean()) / (gts.std() + 1e-8)
        elif self.baseline == 'time':
            # make time idx
            t_idx = torch.cat([torch.arange(L, device=self.device) for L in lengths], dim=0)
            T_max = max(lengths)
            b_t = torch.empty(T_max, device=self.device, dtype=gts.dtype)
            for t in range(T_max):
                b_t[t] = gts[t_idx == t].mean()
            baseline_per_sample = b_t[t_idx]
            return gts - baseline_per_sample
        else:
            raise ValueError("Baseline must be 'mean' or 'time'.")

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
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if not done:
            return {}

        gt = list()
        g = 0
        for r in reversed(self.rewards):
            g = r + self.gamma * g
            gt.insert(0, g)

        self.replay_episodes.push(self.states, self.actions, self.rewards, gt)
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

        if len(self.replay_episodes) < self.batch_size:
            return {"loss": 0}

        batch_sample = self.replay_episodes.sample(self.batch_size)

        states= batch_sample["states"].to(self.device)
        actions = batch_sample["actions"].to(self.device)
        gts= batch_sample["gts"].to(self.device)
        lengths = batch_sample["lengths"]
        # mask = batch_sample["mask"].to(self.device)

        self.policy_net.train()
        cater_logits = self.policy_net(states)
        log_probs = cater_logits.log_prob(actions)
        gts = self.compute_baseline(gts, lengths)
        loss = -(log_probs * gts).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.replay_episodes._buffer.clear()

        return {'loss': loss.item()}

    def save(self, path: str) -> None:
        """Save policy network weights."""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load policy network weights."""
        self.policy_net.load_state_dict(torch.load(path))

    def set_eval_mode(self) -> None:
        """Set policy to eval mode (still stochastic but no dropout/batchnorm)."""
        self.policy_net.eval()

    def set_train_mode(self) -> None:
        """Set policy back to train mode."""
        self.policy_net.train()
