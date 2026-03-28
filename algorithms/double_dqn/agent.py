"""
Double Deep Q-Network (DQN) Agent
============================

Key idea: Replace the Q-table with a neural network Q(s,a;θ) that maps
states to action values. Use experience replay and a target network for
stable training.

Two critical innovations over tabular Q-learning:
1. Experience Replay: Store transitions in a buffer and train on random
   mini-batches to break temporal correlations.
2. Target Network: Use a separate, slowly-updated copy of the Q-network
   to compute TD targets, preventing oscillation.

Loss: MSE between Q(s,a;θ) and [r + γ Q(s', max_a' Q(s',a';θ); θ⁻)]
where θ⁻ are the target network's frozen parameters.

Recommended reading: Hasselt et al. 2015 "Deep Reinforcement Learning with Double Q-learning"

See algorithms/double_dqn/network.py for the provided QNetwork architecture.
"""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn.functional as F

from core.base_agent import BaseAgent
from core.replay_buffer import ReplayBuffer
from .network import QNetwork


class DoubleDQNAgent(BaseAgent):
    """Double DQN agent with experience replay and target network.

    Args:
        state_dim: Dimension of observation space (8 for LunarLander).
        n_actions: Number of discrete actions (4 for LunarLander).
        hidden_dim: Hidden layer size for Q-network.
        learning_rate: Optimizer learning rate.
        gamma: Discount factor.
        epsilon: Initial exploration rate.
        epsilon_decay: Multiplicative decay per step.
        epsilon_min: Minimum exploration rate.
        buffer_capacity: Replay buffer maximum size.
        batch_size: Mini-batch size for training.
        update_tau: Update tau parameter.
        target_update_every: Steps between target network syncs.
    """

    def __init__(
        self,
        state_dim: int = 8,
        n_actions: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_every: int = 1000,
        update_tau: float = 0.01,
        clip_norm: float = 1.0,
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.target_update_every = target_update_every
        self.update_tau = update_tau
        self.clip_norm = clip_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_counter = 0
        self._train_epsilon = epsilon

        self.online_net = QNetwork(state_dim, n_actions, hidden_dim)
        self.target_net = QNetwork(state_dim, n_actions, hidden_dim)
        self.online_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection using the online Q-network.

        With probability epsilon, return random action.
        Otherwise, pass state through Q-network and return argmax.
        """
        # 1. if random < epsilon: return random action
        # 2. else:
        #    - Convert state to tensor: torch.FloatTensor(state).unsqueeze(0)
        #    - with torch.no_grad(): q_values = self.online_net(state_tensor)
        #    - return q_values.argmax(dim=1).item()
        if random.random() < self.epsilon:
            return int(random.randint(0, self.n_actions - 1))
        action = self.online_net(torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)).argmax()
        return int(action.item())

    def update(self, state, action, reward, next_state, done) -> dict:
        """Store transition and train on a mini-batch if buffer is ready.

        Steps:
        1. Push transition to replay buffer
        2. If buffer has < 1000 transitions, skip training
        3. Sample a mini-batch from the buffer
        4. Compute TD targets: r + gamma * max_a'(Q_target(s', a'))
        5. Compute current Q-values: Q_online(s, a)
        6. Loss = MSE(current_Q, target_Q)
        7. Backprop and optimize
        8. Periodically sync target network

        Returns:
            {"loss": float, "epsilon": float, "buffer_size": int}
        """
        # Step 1: self.buffer.push(state, action, reward, next_state, done)
        #
        # Step 2: if not self.buffer.is_ready: return {"loss": 0, ...}
        #
        # Step 3: batch = self.buffer.sample(self.batch_size)
        #         states = batch["states"]
        #         actions = batch["actions"]
        #         rewards = batch["rewards"]
        #         next_states = batch["next_states"]
        #         dones = batch["dones"]
        #
        # Step 4: with torch.no_grad():
        #             next_actions = self.online_net(next_states).max(dim=1)[1].unsqueeze(1)
        #             next_q = self.target_net(next_states).gather(1, next_actions)
        #             targets = rewards + self.gamma * next_q * (1 - dones)
        #
        # Step 5: current_q = self.online_net(states)
        #         current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        #
        # Step 6: loss = F.mse_loss(current_q, targets)
        #
        # Step 7: self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()
        #
        # Step 8: self.step_count += 1
        #         if self.step_count % self.target_update_every == 0:
        #             self.target_net.load_state_dict(self.online_net.state_dict())
        #
        # Don't forget to decay epsilon!
        self.replay_buffer.push(state, action, reward, next_state, done)
        if not self.replay_buffer.is_ready:
            return {
                "loss": 0.0,
                "epsilon": self.epsilon,
                "buffer_size": len(self.replay_buffer),
            }

        batch = self.replay_buffer.sample(self.batch_size)
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)

        with torch.no_grad():
            self.online_net.eval()
            next_actions = self.online_net(next_states).max(1)[1].unsqueeze(1)
            next_q_value = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * next_q_value * (1 - dones)

        self.online_net.train()
        self.optimizer.zero_grad()
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        advantages = F.mse_loss(current_q, target)
        advantages.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.clip_norm)
        self.optimizer.step()

        self.step_counter += 1
        if self.step_counter % self.target_update_every == 0:
            with torch.no_grad():
                for param_a, param_b in zip(self.target_net.parameters(), self.online_net.parameters()):
                    param_a.mul_(self.update_tau).add_(param_b, alpha=1 - self.update_tau)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return {
            "loss": advantages.item(),
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
        }

    def save(self, path: str) -> None:
        """Save online network weights."""
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load online network weights and sync to target."""
        self.online_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.online_net.state_dict())

    def set_eval_mode(self) -> None:
        """Set epsilon to 0 and network to eval mode."""
        self._train_epsilon = self.epsilon
        self.epsilon = 0.0
        self.online_net.eval()

    def set_train_mode(self) -> None:
        """Restore epsilon and set network to train mode."""
        self.epsilon = self._train_epsilon
        self.online_net.train()
