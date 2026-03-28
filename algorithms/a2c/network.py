"""A2C Combined Actor-Critic Network (fully implemented)."""
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticNetwork(nn.Module):
    """Combined actor-critic with shared feature layers.

    Architecture:
        Shared: state_dim -> 128 -> 128
        Actor head: 128 -> n_actions (softmax)
        Critic head: 128 -> 1 (state value)
    """
    def __init__(self, state_dim: int = 8, n_actions: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """Returns (action distribution, state value)."""
        features = self.shared(x)
        dist = Categorical(logits=self.actor_head(features))
        value = self.critic_head(features)
        return dist, value

    def get_action_and_value(self, state: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob, entropy, value)."""
        dist, value = self.forward(state)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy(), value


class ActorNetwork(nn.Module):
    """Policy network (actor) mapping states to action probabilities."""
    def __init__(self, state_dim: int = 8, n_actions: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> Categorical:
        logits = self.net(x)
        return Categorical(logits=logits)

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        logits = self.forward(state)
        action = logits.sample()
        return action.item(), logits.log_prob(action)


class CriticNetwork(nn.Module):
    """Value network (critic) mapping states to state values V(s)."""
    def __init__(self, state_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns state value V(s): shape (batch, 1)."""
        return self.net(x)