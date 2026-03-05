"""
Tabular Q-Learning Agent
=========================

Key idea: Maintain a table Q[state][action] representing the expected
cumulative reward for taking an action in a given state. Update using
the Bellman equation after each step:

    Q(s,a) <- Q(s,a) + lr * [r + gamma * max_a'(Q(s',a')) - Q(s,a)]

Note: LunarLander has a continuous state space (8 floats), so you must
discretize it into bins. Fewer bins = faster learning but coarser policy.
More bins = finer control but exponentially larger Q-table.

Recommended reading: Sutton & Barto, Chapter 6 (TD Learning)

See environment/README.md for state variable ranges and discretization tips.
"""

from __future__ import annotations

import numpy as np

from core.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Tabular Q-Learning agent with state discretization.

    Args:
        n_bins: Number of bins per continuous state dimension.
        learning_rate: Step size for Q-value updates (alpha).
        gamma: Discount factor for future rewards.
        epsilon: Initial exploration rate (probability of random action).
        epsilon_decay: Multiplicative decay applied to epsilon each episode.
        epsilon_min: Minimum exploration rate.
    """

    def __init__(
        self,
        n_bins: int = 6,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        # TODO: Store hyperparameters
        # TODO: Define bin edges for each of the 8 state dimensions
        #   Hint: Use the ranges from environment/README.md
        #   x_position: [-1.5, 1.5], y_position: [0, 1.5], etc.
        #   For leg contacts (dims 6,7), use [0, 1] (already discrete)
        # TODO: Initialize Q-table with zeros
        #   Shape: (n_bins,) * 6 + (2, 2) + (4,)
        #   The first 6 dims are continuous (discretized), dims 6-7 are binary,
        #   and the last dim is for the 4 actions
        raise NotImplementedError("Implement __init__: create Q-table and bin edges")

    def _discretize(self, state: np.ndarray) -> tuple:
        """Convert continuous 8-dim state to discrete bin indices.

        Steps:
        1. For each continuous dimension (0-5), clip to known range
        2. Use np.digitize to find which bin the value falls into
        3. For dimensions 6-7 (leg contacts), just round to 0 or 1

        Returns:
            Tuple of integers indexing into the Q-table.
        """
        # TODO: Implement discretization
        # Hint: np.digitize(value, bin_edges) returns the bin index
        # Hint: np.clip(value, low, high) keeps values in range
        raise NotImplementedError("Implement _discretize: convert state to bin indices")

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection.

        With probability epsilon, return a random action (0-3).
        Otherwise, return the action with the highest Q-value
        for the discretized state.
        """
        # TODO: Implement epsilon-greedy
        # 1. discretized = self._discretize(state)
        # 2. if np.random.random() < self.epsilon: return random action
        # 3. else: return np.argmax(self.q_table[discretized])
        raise NotImplementedError("Implement select_action: epsilon-greedy policy")

    def update(self, state, action, reward, next_state, done) -> dict:
        """Apply the Q-learning update rule.

        Q(s,a) <- Q(s,a) + lr * [r + gamma * max_a'(Q(s',a')) - Q(s,a)]

        If done is True, there is no next state value (terminal).
        Also decay epsilon after each update.

        Returns:
            {"td_error": float, "epsilon": float}
        """
        # TODO: Implement Q-learning update
        # 1. Discretize both state and next_state
        # 2. Compute TD target: r + gamma * max(Q[next_state]) (0 if done)
        # 3. Compute TD error: target - Q[state][action]
        # 4. Update: Q[state][action] += learning_rate * td_error
        # 5. Decay epsilon: self.epsilon = max(epsilon_min, epsilon * epsilon_decay)
        raise NotImplementedError("Implement update: Bellman equation update")

    def save(self, path: str) -> None:
        """Save Q-table to disk using np.save."""
        # TODO: np.save(path, self.q_table)
        raise NotImplementedError("Implement save")

    def load(self, path: str) -> None:
        """Load Q-table from disk using np.load."""
        # TODO: self.q_table = np.load(path)
        raise NotImplementedError("Implement load")

    def set_eval_mode(self) -> None:
        """Store current epsilon and set to 0 (pure greedy)."""
        # TODO: self._train_epsilon = self.epsilon; self.epsilon = 0.0
        raise NotImplementedError("Implement set_eval_mode")

    def set_train_mode(self) -> None:
        """Restore training epsilon."""
        # TODO: self.epsilon = self._train_epsilon
        raise NotImplementedError("Implement set_train_mode")
