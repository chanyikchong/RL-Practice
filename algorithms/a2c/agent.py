"""
Advantage Actor-Critic (A2C) Agent
====================================

Key idea: A2C is the synchronous version of A3C. It collects N steps of
experience, computes advantages using Generalized Advantage Estimation (GAE)
or N-step returns, and updates the shared actor-critic network.

Key differences from basic Actor-Critic:
1. N-step returns instead of 1-step TD (reduces bias, increases variance)
2. Entropy bonus to encourage exploration
3. Combined loss: actor_loss + value_coeff * critic_loss - entropy_coeff * entropy

The update happens after every N steps (not every single step).

Recommended reading: Mnih et al. 2016 "Asynchronous Methods for Deep RL"

See algorithms/a2c/network.py for the provided ActorCriticNetwork.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core.base_agent import BaseAgent
from .network import ActorCriticNetwork


class A2CAgent(BaseAgent):
    """Advantage Actor-Critic with N-step returns and entropy regularization.

    Args:
        state_dim: Dimension of observation space (8).
        n_actions: Number of discrete actions (4).
        hidden_dim: Hidden layer size.
        learning_rate: Optimizer learning rate.
        gamma: Discount factor.
        n_steps: Number of steps to collect before updating.
        value_coeff: Weight for critic loss in combined loss.
        entropy_coeff: Weight for entropy bonus (encourages exploration).
    """

    def __init__(
        self,
        state_dim: int = 8,
        n_actions: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 7e-4,
        gamma: float = 0.99,
        n_steps: int = 5,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
    ):
        # TODO: Store hyperparameters
        # TODO: Create ActorCriticNetwork
        # TODO: Create optimizer
        # TODO: Initialize N-step storage:
        #   self.states = []
        #   self.actions = []
        #   self.rewards = []
        #   self.log_probs = []
        #   self.values = []
        #   self.dones = []
        raise NotImplementedError("Implement __init__: create network and storage")

    def select_action(self, state: np.ndarray) -> int:
        """Sample action from policy, storing log_prob and value estimate.

        Uses the combined ActorCriticNetwork to get both the action
        distribution and value estimate in one forward pass.
        """
        # TODO: Implement action selection
        # state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # action, log_prob, value = self.network.get_action_and_value(state_tensor)
        # self.log_probs.append(log_prob)
        # self.values.append(value.squeeze())
        # return action
        raise NotImplementedError("Implement select_action")

    def update(self, state, action, reward, next_state, done) -> dict:
        """Store transition; update every N steps or at episode end.

        When updating:
        1. Compute N-step returns:
           G_t = r_t + γr_{t+1} + ... + γ^(n-1)r_{t+n-1} + γ^n V(s_{t+n})
        2. Compute advantages: A_t = G_t - V(s_t)
        3. Actor loss: -mean(log_prob * advantage)
        4. Critic loss: MSE(V(s_t), G_t)
        5. Entropy bonus: mean(entropy)
        6. Total loss: actor + value_coeff * critic - entropy_coeff * entropy
        7. Clear storage after update

        Returns:
            {"loss": float, "actor_loss": float, "critic_loss": float,
             "entropy": float} when updating, {} otherwise
        """
        # TODO: Implement A2C update
        #
        # self.states.append(state)
        # self.actions.append(action)
        # self.rewards.append(reward)
        # self.dones.append(done)
        #
        # if len(self.states) < self.n_steps and not done:
        #     return {}
        #
        # # Bootstrap value for last state (0 if done)
        # if done:
        #     next_value = torch.tensor(0.0)
        # else:
        #     with torch.no_grad():
        #         _, next_value = self.network(torch.FloatTensor(next_state).unsqueeze(0))
        #         next_value = next_value.squeeze()
        #
        # # Compute returns backwards
        # returns = []
        # G = next_value
        # for r, d in zip(reversed(self.rewards), reversed(self.dones)):
        #     G = r + self.gamma * G * (1 - float(d))
        #     returns.insert(0, G)
        # returns = torch.stack(returns) if isinstance(returns[0], torch.Tensor) else torch.tensor(returns)
        #
        # # Stack values and log_probs
        # values = torch.stack(self.values)
        # log_probs = torch.stack(self.log_probs)
        #
        # # Advantages
        # advantages = returns - values.detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        #
        # # Compute entropy from current policy
        # states_t = torch.FloatTensor(np.array(self.states))
        # actions_t = torch.tensor(self.actions)
        # dist, _ = self.network(states_t)
        # entropy = dist.entropy().mean()
        #
        # # Losses
        # actor_loss = -(log_probs * advantages).mean()
        # critic_loss = F.mse_loss(values, returns)
        # total_loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
        #
        # self.optimizer.zero_grad()
        # total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        # self.optimizer.step()
        #
        # # Clear storage
        # self.states.clear(); self.actions.clear(); self.rewards.clear()
        # self.log_probs.clear(); self.values.clear(); self.dones.clear()
        #
        # return {"loss": total_loss.item(), "actor_loss": actor_loss.item(),
        #         "critic_loss": critic_loss.item(), "entropy": entropy.item()}
        raise NotImplementedError("Implement update: N-step A2C")

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
