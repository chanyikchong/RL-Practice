"""
Actor-Critic Agent
====================

Key idea: Combine policy gradient (actor) with a learned value function (critic).
The critic estimates V(s) and provides a baseline to reduce variance in the
policy gradient, while the actor updates the policy using the advantage:

    A(s,a) = r + γV(s') - V(s)   (TD advantage)

Unlike REINFORCE, Actor-Critic updates at EVERY step (not just episode end),
using bootstrapped estimates instead of Monte Carlo returns.

Actor loss:  -log π(a|s) * A(s,a)
Critic loss: MSE(V(s), r + γV(s'))

Recommended reading: Sutton & Barto, Chapter 13.5

See algorithms/actor_critic/network.py for ActorNetwork and CriticNetwork.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core.base_agent import BaseAgent
from .network import ActorNetwork, CriticNetwork


class ActorCriticAgent(BaseAgent):
    """One-step Actor-Critic agent with separate actor and critic networks.

    Args:
        state_dim: Dimension of observation space (8).
        n_actions: Number of discrete actions (4).
        hidden_dim: Hidden layer size.
        actor_lr: Learning rate for the actor (policy).
        critic_lr: Learning rate for the critic (value function).
        gamma: Discount factor.
    """

    def __init__(
        self,
        state_dim: int = 8,
        n_actions: int = 4,
        hidden_dim: int = 128,
        actor_lr: float = 1e-3,
        critic_lr: float = 5e-3,
        gamma: float = 0.99,
    ):
        # TODO: Store hyperparameters
        # TODO: Create ActorNetwork and CriticNetwork
        # TODO: Create separate optimizers for actor and critic
        # TODO: self._last_log_prob = None (to store between select_action and update)
        raise NotImplementedError("Implement __init__: create actor, critic, optimizers")

    def select_action(self, state: np.ndarray) -> int:
        """Sample action from actor and store log probability.

        Steps:
        1. Convert state to tensor
        2. Get action distribution from actor
        3. Sample action
        4. Store log_prob in self._last_log_prob
        """
        # TODO: Implement action selection
        # state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # dist = self.actor(state_tensor)
        # action = dist.sample()
        # self._last_log_prob = dist.log_prob(action)
        # return action.item()
        raise NotImplementedError("Implement select_action")

    def update(self, state, action, reward, next_state, done) -> dict:
        """One-step Actor-Critic update using TD advantage.

        Steps:
        1. Compute V(s) and V(s') using the critic
        2. Compute TD target: r + γV(s') (0 if done)
        3. Compute advantage: A = TD_target - V(s)
        4. Critic loss: MSE(V(s), TD_target)
        5. Actor loss: -log_prob * advantage.detach()
           (detach advantage so critic gradients don't flow into actor)
        6. Update both networks

        Returns:
            {"actor_loss": float, "critic_loss": float, "advantage": float}
        """
        # TODO: Implement actor-critic update
        #
        # state_t = torch.FloatTensor(state).unsqueeze(0)
        # next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        #
        # # Critic: compute values
        # value = self.critic(state_t).squeeze()
        # with torch.no_grad():
        #     next_value = self.critic(next_state_t).squeeze() if not done else torch.tensor(0.0)
        #     td_target = reward + self.gamma * next_value
        #
        # # Advantage
        # advantage = td_target - value.detach()
        #
        # # Critic loss
        # critic_loss = F.mse_loss(value, td_target)
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()
        #
        # # Actor loss
        # actor_loss = -self._last_log_prob * advantage
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()
        #
        # return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
        #         "advantage": advantage.item()}
        raise NotImplementedError("Implement update: TD actor-critic")

    def save(self, path: str) -> None:
        """Save both actor and critic weights."""
        # TODO: torch.save({"actor": ..., "critic": ...}, path)
        raise NotImplementedError("Implement save")

    def load(self, path: str) -> None:
        """Load both actor and critic weights."""
        # TODO: checkpoint = torch.load(path)
        # self.actor.load_state_dict(checkpoint["actor"])
        # self.critic.load_state_dict(checkpoint["critic"])
        raise NotImplementedError("Implement load")

    def set_eval_mode(self) -> None:
        # TODO: self.actor.eval(); self.critic.eval()
        raise NotImplementedError("Implement set_eval_mode")

    def set_train_mode(self) -> None:
        # TODO: self.actor.train(); self.critic.train()
        raise NotImplementedError("Implement set_train_mode")
