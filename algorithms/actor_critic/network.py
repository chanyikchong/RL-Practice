"""Actor-Critic Networks (fully implemented)."""
import torch
import torch.nn as nn
from torch.distributions import Categorical

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
