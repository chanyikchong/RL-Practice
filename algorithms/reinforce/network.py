"""REINFORCE Policy Network (fully implemented)."""
import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """Maps states to action probabilities.

    Architecture: state_dim -> 128 -> 128 -> n_actions (softmax)
    """
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
        """Returns a Categorical distribution over actions."""
        logits = self.net(x)
        return Categorical(logits=logits)

    def get_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Sample action and return (action, log_prob)."""
        dist = self.forward(state)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
