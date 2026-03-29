"""
Deep Q-Network (DQN) Agent -- SOLUTION
========================================

DQN replaces the Q-table with a neural network Q(s, a; theta) that
generalises across similar states. Two key innovations stabilise training:

1. **Experience Replay** -- Transitions are stored in a circular buffer and
   sampled randomly for training. This breaks the temporal correlation
   between consecutive samples that would otherwise destabilise SGD.

2. **Target Network** -- A slowly-updated copy of the Q-network provides
   the TD targets. Because the targets change less frequently, the
   optimisation landscape is more stationary, preventing oscillation.

Loss at each gradient step:

    L = MSE( Q(s, a; theta),  r + gamma * max_a' Q(s', a'; theta^-) )

where theta^- are the frozen target-network parameters.

Reference: Mnih et al. 2015, "Human-level control through deep RL"
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core.base_agent import BaseAgent
from core.replay_buffer import ReplayBuffer
from algorithms.dqn.network import QNetwork


class DQNAgent(BaseAgent):
    """DQN agent with experience replay and a periodic target-network sync.

    Args:
        state_dim: Dimension of observation space (8 for LunarLander).
        n_actions: Number of discrete actions (4 for LunarLander).
        hidden_dim: Width of hidden layers in the Q-network.
        learning_rate: Adam optimiser learning rate.
        gamma: Discount factor.
        epsilon: Initial exploration probability.
        epsilon_decay: Multiplicative decay applied to epsilon each step.
        epsilon_min: Minimum exploration probability.
        buffer_capacity: Maximum number of transitions in the replay buffer.
        batch_size: Mini-batch size sampled from the buffer each update.
        target_update_every: Number of gradient steps between target syncs.
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
    ) -> None:
        # -- Hyperparameters --------------------------------------------------
        self.n_actions: int = n_actions
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self.batch_size: int = batch_size
        self.target_update_every: int = target_update_every

        # -- Networks ---------------------------------------------------------
        # The *online* network is the one we optimise with gradient descent.
        self.online_net = QNetwork(state_dim, n_actions, hidden_dim)
        # The *target* network is a frozen copy used to compute TD targets.
        self.target_net = QNetwork(state_dim, n_actions, hidden_dim)
        # Initialise target with the same weights as online.
        self.target_net.load_state_dict(self.online_net.state_dict())
        # We never compute gradients through the target network.
        self.target_net.eval()

        # -- Optimiser --------------------------------------------------------
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=learning_rate
        )

        # -- Replay buffer ----------------------------------------------------
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        # -- Step counter (for scheduling target-network updates) -------------
        self.step_count: int = 0

        # -- Eval / train bookkeeping -----------------------------------------
        self._train_epsilon: float = epsilon

    # ---------------------------------------------------------------------- #
    #  Action selection                                                        #
    # ---------------------------------------------------------------------- #

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection using the online Q-network.

        With probability ``epsilon`` a uniformly random action is returned
        (exploration). Otherwise the action with the highest predicted
        Q-value is selected (exploitation).
        """
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))

        # Convert numpy state to a batched tensor: shape (1, state_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)  # (1, n_actions)
        return int(q_values.argmax(dim=1).item())

    # ---------------------------------------------------------------------- #
    #  Learning update                                                         #
    # ---------------------------------------------------------------------- #

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, terminated: bool, truncated: bool, done: bool) -> dict:
        """Store transition, sample a mini-batch, and perform a gradient step.

        The update only runs once the replay buffer has accumulated enough
        transitions (``buffer.is_ready`` checks for >= 1000).

        Steps:
            1. Push the new transition into the replay buffer.
            2. If the buffer is not yet ready, return early.
            3. Sample a random mini-batch of transitions.
            4. Compute TD targets using the *target* network.
            5. Compute current Q-values using the *online* network.
            6. Minimise the MSE loss between current Q and TD targets.
            7. Clip gradients to prevent large updates.
            8. Periodically copy online weights to the target network.
            9. Decay epsilon.

        Returns:
            Dict with ``loss``, ``epsilon``, and ``buffer_size``.
        """
        # Step 1: store transition
        # Store terminated (not done) so the TD target is bootstrapped correctly
        # for truncated episodes (time limit) rather than zeroing out future value.
        self.buffer.push(state, action, reward, next_state, terminated)

        # Step 2: skip training until we have enough data
        if not self.buffer.is_ready:
            return {
                "loss": 0.0,
                "epsilon": self.epsilon,
                "buffer_size": len(self.buffer),
            }

        # Step 3: sample a mini-batch (returned as torch tensors)
        batch = self.buffer.sample(self.batch_size)
        states = batch["states"]           # (B, state_dim)
        actions = batch["actions"]         # (B,) int64
        rewards = batch["rewards"]         # (B,)
        next_states = batch["next_states"] # (B, state_dim)
        dones = batch["dones"]             # (B,) float 0/1

        # Step 4: compute TD targets with the *frozen* target network
        with torch.no_grad():
            # Best Q-value at each next state according to target net
            next_q_values = self.target_net(next_states).max(dim=1)[0]  # (B,)
            # If the episode ended, there is no future value
            td_targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        # Step 5: Q-values from the online network for the actions we took
        all_q = self.online_net(states)  # (B, n_actions)
        # gather selects Q(s, a) for the action actually taken
        current_q = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Step 6: MSE loss between predicted Q and TD target
        loss = F.mse_loss(current_q, td_targets)

        # Step 7: gradient descent with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilise training (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Step 8: periodically copy online weights -> target network
        self.step_count += 1
        if self.step_count % self.target_update_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Step 9: decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "buffer_size": len(self.buffer),
        }

    # ---------------------------------------------------------------------- #
    #  Persistence                                                             #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save the online network weights to disk."""
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load network weights and sync the target network."""
        self.online_net.load_state_dict(torch.load(path))
        # Keep target in sync after loading
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ---------------------------------------------------------------------- #
    #  Eval / train mode                                                       #
    # ---------------------------------------------------------------------- #

    def set_eval_mode(self) -> None:
        """Disable exploration and set network to eval mode.

        Eval mode disables dropout / batch-norm updates (not used in the
        default QNetwork, but good practice).
        """
        self._train_epsilon = self.epsilon
        self.epsilon = 0.0
        self.online_net.eval()

    def set_train_mode(self) -> None:
        """Re-enable exploration and set network to train mode."""
        self.epsilon = self._train_epsilon
        self.online_net.train()
