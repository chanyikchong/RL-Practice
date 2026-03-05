"""Abstract base class for all RL agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    """Base class that all algorithm implementations must inherit from.

    Every agent must implement:
    - ``select_action``: choose an action given a state
    - ``update``: perform a learning update and return metrics
    - ``save`` / ``load``: persist and restore agent state

    Optional overrides:
    - ``set_eval_mode`` / ``set_train_mode``: switch between greedy and exploratory policies
    """

    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """Given a state observation, return an action (0-3)."""
        ...

    @abstractmethod
    def update(self, *args, **kwargs) -> dict:
        """Perform a learning update.

        Returns:
            Dict of metrics, e.g. {"loss": 0.42, "epsilon": 0.3}
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent weights/Q-table to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent weights/Q-table from disk."""
        ...

    def set_eval_mode(self) -> None:
        """Switch to greedy/deterministic policy for evaluation."""
        pass

    def set_train_mode(self) -> None:
        """Switch back to exploration policy for training."""
        pass
