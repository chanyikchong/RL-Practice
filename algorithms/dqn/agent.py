"""
Deep Q-Network (DQN) Agent
============================

Key idea: Replace the Q-table with a neural network Q(s,a;θ) that maps
states to action values. Use experience replay and a target network for
stable training.

Two critical innovations over tabular Q-learning:
1. Experience Replay: Store transitions in a buffer and train on random
   mini-batches to break temporal correlations.
2. Target Network: Use a separate, slowly-updated copy of the Q-network
   to compute TD targets, preventing oscillation.

Loss: MSE between Q(s,a;θ) and [r + γ max_a' Q(s',a';θ⁻)]
where θ⁻ are the target network's frozen parameters.

Recommended reading: Mnih et al. 2015 "Human-level control through deep RL"

See algorithms/dqn/network.py for the provided QNetwork architecture.
"""

from __future__ import annotations

import numpy as np
import torch

from core.base_agent import BaseAgent
from core.replay_buffer import ReplayBuffer
from .network import QNetwork


class DQNAgent(BaseAgent):
    """DQN agent with experience replay and target network.

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
    ):
        # TODO: Store hyperparameters
        # TODO: Create the online Q-network: QNetwork(state_dim, n_actions, hidden_dim)
        # TODO: Create the target Q-network: QNetwork(...) — same architecture
        # TODO: Copy online weights to target: target_net.load_state_dict(online_net.state_dict())
        # TODO: Create optimizer: torch.optim.Adam(online_net.parameters(), lr=learning_rate)
        # TODO: Create replay buffer: ReplayBuffer(buffer_capacity)
        # TODO: Initialize step counter for target updates
        raise NotImplementedError("Implement __init__: create networks, optimizer, buffer")

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection using the online Q-network.

        With probability epsilon, return random action.
        Otherwise, pass state through Q-network and return argmax.
        """
        # TODO: Implement epsilon-greedy with neural network
        # 1. if random < epsilon: return random action
        # 2. else:
        #    - Convert state to tensor: torch.FloatTensor(state).unsqueeze(0)
        #    - with torch.no_grad(): q_values = self.online_net(state_tensor)
        #    - return q_values.argmax(dim=1).item()
        raise NotImplementedError("Implement select_action: epsilon-greedy with Q-network")

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
        # TODO: Implement DQN update
        #
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
        #             next_q = self.target_net(next_states).max(dim=1)[0]
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
        raise NotImplementedError("Implement update: experience replay + target network")

    def save(self, path: str) -> None:
        """Save online network weights."""
        # TODO: torch.save(self.online_net.state_dict(), path)
        raise NotImplementedError("Implement save")

    def load(self, path: str) -> None:
        """Load online network weights and sync to target."""
        # TODO: self.online_net.load_state_dict(torch.load(path))
        # TODO: self.target_net.load_state_dict(self.online_net.state_dict())
        raise NotImplementedError("Implement load")

    def set_eval_mode(self) -> None:
        """Set epsilon to 0 and network to eval mode."""
        # TODO: self._train_epsilon = self.epsilon
        # TODO: self.epsilon = 0.0
        # TODO: self.online_net.eval()
        raise NotImplementedError("Implement set_eval_mode")

    def set_train_mode(self) -> None:
        """Restore epsilon and set network to train mode."""
        # TODO: self.epsilon = self._train_epsilon
        # TODO: self.online_net.train()
        raise NotImplementedError("Implement set_train_mode")
