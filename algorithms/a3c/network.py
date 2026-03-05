"""A3C Shared Actor-Critic Network (fully implemented)."""
import torch
import torch.nn as nn
from torch.distributions import Categorical

class SharedActorCriticNetwork(nn.Module):
    """Combined actor-critic for A3C with weight sharing utilities.

    Same architecture as A2C but includes methods for syncing
    weights between global and local networks.
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
        features = self.shared(x)
        dist = Categorical(logits=self.actor_head(features))
        value = self.critic_head(features)
        return dist, value

    def get_action_and_value(self, state: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(state)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def sync_from(self, source: "SharedActorCriticNetwork") -> None:
        """Copy all parameters from source network (used by workers)."""
        self.load_state_dict(source.state_dict())
