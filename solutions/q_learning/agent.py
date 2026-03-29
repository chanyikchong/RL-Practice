"""
Tabular Q-Learning Agent -- SOLUTION
======================================

Q-Learning is the simplest model-free RL algorithm. It maintains a lookup
table Q[state][action] that estimates the expected cumulative reward for
taking a given action in a given state, then following the optimal policy.

The core update rule is the Bellman equation:

    Q(s,a) <- Q(s,a) + lr * [r + gamma * max_a'(Q(s',a')) - Q(s,a)]

Because LunarLander has a continuous 8-dimensional state space, we must
discretize it into bins so we can index into the Q-table. This trades off
resolution (more bins = finer policy) against table size (exponential in
the number of dimensions).

State dimensions and their typical ranges:
    0: x position      [-1.5,  1.5]
    1: y position      [ 0.0,  1.5]
    2: x velocity      [-2.0,  2.0]
    3: y velocity      [-2.0,  2.0]
    4: angle           [-pi/2, pi/2]
    5: angular velocity[-2.0,  2.0]
    6: left leg contact  {0, 1}
    7: right leg contact {0, 1}

Actions: 0=noop, 1=fire left, 2=fire main, 3=fire right
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from core.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Tabular Q-Learning agent with state discretization.

    The continuous 8-dim state is discretized into bins so it can serve as
    an index into a numpy Q-table. Epsilon-greedy exploration is used, with
    epsilon decaying multiplicatively after every update step.

    Args:
        n_bins: Number of bins per continuous state dimension.
        learning_rate: Step size for Q-value updates (alpha).
        gamma: Discount factor for future rewards.
        epsilon: Initial exploration rate (probability of random action).
        epsilon_decay: Multiplicative decay applied to epsilon each step.
        epsilon_min: Floor on exploration rate.
    """

    def __init__(
        self,
        n_bins: int = 6,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.01,
    ) -> None:
        # -- Hyperparameters --------------------------------------------------
        self.n_bins: int = n_bins
        self.learning_rate: float = learning_rate
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min

        # -- Bin edges for each continuous dimension --------------------------
        # np.linspace creates n_bins-1 interior edges, which np.digitize uses
        # to map a value into one of n_bins buckets (indices 0 .. n_bins-1).
        self.bin_edges: List[np.ndarray] = [
            np.linspace(-1.5, 1.5, n_bins - 1),       # dim 0: x position
            np.linspace(0.0, 1.5, n_bins - 1),        # dim 1: y position
            np.linspace(-2.0, 2.0, n_bins - 1),       # dim 2: x velocity
            np.linspace(-2.0, 2.0, n_bins - 1),       # dim 3: y velocity
            np.linspace(-math.pi / 2, math.pi / 2, n_bins - 1),  # dim 4: angle
            np.linspace(-2.0, 2.0, n_bins - 1),       # dim 5: angular velocity
        ]

        # -- Q-table ----------------------------------------------------------
        # Shape: (n_bins,)*6 for the continuous dims, (2,2) for the two binary
        # leg-contact dims, and (4,) for the four discrete actions.
        # Total entries: 6^6 * 2^2 * 4 = 746_496 (manageable).
        self.q_table: np.ndarray = np.zeros(
            (n_bins,) * 6 + (2, 2, 4), dtype=np.float64
        )

        # -- Eval / train mode bookkeeping ------------------------------------
        self._train_epsilon: float = epsilon

    # ---------------------------------------------------------------------- #
    #  State discretization                                                    #
    # ---------------------------------------------------------------------- #

    def _discretize(self, state: np.ndarray) -> Tuple[int, ...]:
        """Convert a continuous 8-dim state vector into a tuple of bin indices.

        For the 6 continuous dimensions we clip to the known range and use
        ``np.digitize`` to find the bin index. For the 2 binary leg-contact
        dimensions we simply round to 0 or 1.

        Returns:
            Tuple of 8 integers that can directly index into ``self.q_table``.
        """
        discrete: list[int] = []

        # Continuous dimensions 0-5: digitize into bins
        for i in range(6):
            # Clip to the range covered by our bin edges to avoid out-of-range
            low = self.bin_edges[i][0]
            high = self.bin_edges[i][-1]
            clipped = np.clip(state[i], low, high)
            # np.digitize returns values in [0, n_bins-1] when the value is
            # within the range of the edges.
            bin_idx = int(np.digitize(clipped, self.bin_edges[i]))
            # Clamp to valid index range [0, n_bins-1]
            bin_idx = min(bin_idx, self.n_bins - 1)
            discrete.append(bin_idx)

        # Binary dimensions 6-7 (leg contacts): round to 0 or 1
        discrete.append(int(round(np.clip(state[6], 0, 1))))
        discrete.append(int(round(np.clip(state[7], 0, 1))))

        return tuple(discrete)

    # ---------------------------------------------------------------------- #
    #  Action selection                                                        #
    # ---------------------------------------------------------------------- #

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection.

        With probability ``epsilon`` pick a uniformly random action (explore).
        Otherwise pick the action with the highest Q-value for the current
        discretized state (exploit).
        """
        if np.random.random() < self.epsilon:
            # Exploration: random action from {0, 1, 2, 3}
            return int(np.random.randint(4))

        # Exploitation: greedy w.r.t. the Q-table
        discrete_state = self._discretize(state)
        q_values = self.q_table[discrete_state]  # shape (4,)
        return int(np.argmax(q_values))

    # ---------------------------------------------------------------------- #
    #  Learning update                                                         #
    # ---------------------------------------------------------------------- #

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, terminated: bool, truncated: bool, done: bool) -> dict:
        """Apply the one-step Q-learning (off-policy TD(0)) update.

        Bellman update:
            Q(s,a) <- Q(s,a) + lr * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        If the episode is done, the target is simply ``r`` (no future value).
        After the update, epsilon is decayed multiplicatively.

        Returns:
            Dictionary with ``td_error`` and current ``epsilon``.
        """
        s = self._discretize(state)
        s_next = self._discretize(next_state)

        # Current Q-value for the (state, action) pair
        current_q = self.q_table[s + (action,)]

        # TD target: immediate reward + discounted best future value
        if terminated:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[s_next])

        # TD error measures how "surprising" this transition was
        td_error = td_target - current_q

        # Bellman update
        self.q_table[s + (action,)] += self.learning_rate * td_error

        # Decay exploration rate (but never below the minimum)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {"td_error": float(td_error), "epsilon": self.epsilon}

    # ---------------------------------------------------------------------- #
    #  Persistence                                                             #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save the Q-table to disk as a ``.npy`` file."""
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        """Load a Q-table from disk."""
        self.q_table = np.load(path)

    # ---------------------------------------------------------------------- #
    #  Eval / train mode                                                       #
    # ---------------------------------------------------------------------- #

    def set_eval_mode(self) -> None:
        """Switch to pure greedy policy (epsilon = 0) for evaluation."""
        self._train_epsilon = self.epsilon
        self.epsilon = 0.0

    def set_train_mode(self) -> None:
        """Restore the training epsilon so exploration resumes."""
        self.epsilon = self._train_epsilon
