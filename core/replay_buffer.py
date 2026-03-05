"""Experience replay buffer for off-policy algorithms (DQN, etc.)."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    """A single experience transition."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size circular buffer storing transitions for experience replay.

    Args:
        capacity: Maximum number of transitions to store.
        seed: Random seed for reproducibility.
    """

    def __init__(self, capacity: int = 100_000, seed: int | None = None):
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Add a transition to the buffer."""
        self._buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a random batch and return as tensors.

        Returns:
            Dict with keys: states, actions, rewards, next_states, dones.
            All values are torch.Tensors.
        """
        transitions = self._rng.sample(list(self._buffer), batch_size)

        states = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        return {
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "rewards": torch.from_numpy(rewards),
            "next_states": torch.from_numpy(next_states),
            "dones": torch.from_numpy(dones),
        }

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True if buffer has at least 1000 transitions (minimum before training)."""
        return len(self._buffer) >= 1000
