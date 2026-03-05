"""DQN Q-Network architecture (fully implemented)."""
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Feed-forward Q-network mapping states to action values.

    Architecture: state_dim -> 128 -> 128 -> n_actions
    Uses ReLU activations.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions: shape (batch, n_actions)."""
        return self.net(x)
