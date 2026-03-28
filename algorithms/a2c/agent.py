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
from torch import optim

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
        baseline: str = "norm",
    ):
        self.gamma = gamma
        self.n_steps = n_steps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.baseline = baseline
        self.net = ActorCriticNetwork(state_dim, n_actions, hidden_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        # self.log_probs = list()
        # self.rewards = list()
        # self.values = list()
        # self.entropies = list()
        # self.dones = list()
        self.states = list()
        self.actions = list()
        self.rewards = list()
        self.dones = list()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def select_action(self, state: np.ndarray) -> int:
        """Sample action from policy, storing log_prob and value estimate.

        Uses the combined ActorCriticNetwork to get both the action
        distribution and value estimate in one forward pass.
        """
        # state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # action, log_prob, value = self.network.get_action_and_value(state_tensor)
        # self.log_probs.append(log_prob)
        # self.values.append(value.squeeze())
        # return action

        # state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # action, log_prob, entropy, value = self.net.get_action_and_value(state)
        # self.log_probs.append(log_prob.squeeze())
        # self.values.append(value.squeeze())
        # self.entropies.append(entropy.squeeze())

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            action, _, _, _ = self.net.get_action_and_value(state)
        return action

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

        # self.rewards.append(reward)
        # self.dones.append(done)
        # if len(self.rewards) < self.n_steps and not done:
        #     return {}
        #
        # if done:
        #     next_value = torch.tensor(0.).to(self.device)
        # else:
        #     next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
        #     with torch.no_grad():
        #         _, next_value = self.net(next_state)
        #         next_value = next_value.squeeze()
        #
        # current_values = torch.stack(self.values).to(self.device)
        # log_probs = torch.stack(self.log_probs).to(self.device)
        # entropies = torch.stack(self.entropies).to(self.device)
        #
        # self.net.train()
        # gts = list()
        # g = next_value
        # for r, d in zip(reversed(self.rewards), reversed(self.dones)):
        #     g = r + self.gamma * g * (1 - d)
        #     gts.insert(0, g)
        #
        # targets = torch.stack(gts)
        # try:
        #     advantages = targets - current_values
        # except:
        #     print("in")
        # advantages = self.compute_baseline(advantages, len(advantages))
        # actor_loss = (-log_probs * advantages.detach()).mean()
        # critic_loss = F.mse_loss(current_values, targets)
        # entropy = entropies.mean()
        #
        # loss = actor_loss + critic_loss * self.value_coeff - entropy * self.entropy_coeff
        # self.optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        # self.optimizer.step()
        #
        # self.log_probs.clear()
        # self.values.clear()
        # self.rewards.clear()
        # self.dones.clear()
        # self.entropies.clear()
        # return {
        #     "actor_loss": actor_loss.item(),
        #     "critic_loss": critic_loss.item(),
        #     "entropy": entropy.item(),
        # }

        # Original implementation
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        if len(self.states) < self.n_steps and not done:
            return {}

        self.states.append(next_state)
        self.actions.append(0)  ## to make align with the output of NN
        states = torch.from_numpy(np.array(self.states)).float().to(self.device)
        actions = torch.tensor(self.actions).long().to(self.device)
        rewards = torch.tensor(self.rewards).float().to(self.device)
        dones = torch.tensor(self.dones).long().to(self.device)
        self.net.train()
        self.optimizer.zero_grad()

        dists, values = self.net(states)
        log_probs = dists.log_prob(actions)[:-1]
        current_values = values[:-1].squeeze(1)

        # Compute n-step returns
        gts = list()
        g = torch.tensor(0) if done else values[-1].detach()
        for r, d in zip(reversed(rewards), reversed(dones)):
            g = r + self.gamma * g * (1- d)
            gts.insert(0, g)
        targets = torch.stack(gts).float().to(self.device)
        if len(targets) > 1:
            targets = targets.squeeze(-1)

        advantages = targets - current_values.detach()
        advantages = self.compute_baseline(advantages, len(advantages))

        critic_loss = F.mse_loss(current_values, targets)
        actor_loss = -(log_probs * advantages).mean()

        entropy = dists.entropy()[:-1].mean()
        loss = actor_loss + critic_loss * self.value_coeff - entropy * self.entropy_coeff

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy,
        }

    def compute_baseline(self, gts, lengths):
        if self.baseline == "mean":
            return gts - torch.mean(gts)
        elif self.baseline == "norm":
            if len(gts) == 1:
                return gts - torch.mean(gts)
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

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path))

    def set_eval_mode(self) -> None:
        self.net.eval()
        # self.log_probs.clear()
        # self.values.clear()
        # self.rewards.clear()
        # self.dones.clear()
        # self.entropies.clear()

    def set_train_mode(self) -> None:
        self.net.train()
        # self.log_probs.clear()
        # self.values.clear()
        # self.rewards.clear()
        # self.dones.clear()
        # self.entropies.clear()
