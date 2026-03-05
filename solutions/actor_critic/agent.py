"""
Actor-Critic Agent -- SOLUTION
================================

Actor-Critic combines the best of value-based and policy-based methods:

- The **actor** is a policy network pi(a|s; theta) that selects actions.
- The **critic** is a value network V(s; w) that estimates the expected
  return from a state.

Unlike REINFORCE, which must wait until the end of an episode to compute
returns, Actor-Critic updates at *every* step using the one-step TD error
as a signal. This reduces variance significantly (at the cost of some bias
from the bootstrapped estimate).

At each step the advantage is:

    A(s, a) = r + gamma * V(s') - V(s)

This tells us how much better (or worse) the action was compared to what
the critic expected. The actor is pushed toward actions with positive
advantage and away from actions with negative advantage.

Losses:
    Actor:  L_actor  = -log pi(a|s) * A(s,a).detach()
    Critic: L_critic = MSE( V(s),  r + gamma * V(s') )

Note that we detach the advantage when computing the actor loss so that
gradients from the actor do not flow into the critic.

Reference: Sutton & Barto, Chapter 13.5
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core.base_agent import BaseAgent
from algorithms.actor_critic.network import ActorNetwork, CriticNetwork


class ActorCriticAgent(BaseAgent):
    """One-step Actor-Critic with separate actor and critic networks.

    The actor and critic have independent optimisers so their learning
    rates can be tuned separately. A higher critic learning rate helps
    the value estimate converge faster, giving the actor a more stable
    training signal.

    Args:
        state_dim: Dimension of observation space (8 for LunarLander).
        n_actions: Number of discrete actions (4 for LunarLander).
        hidden_dim: Width of hidden layers.
        actor_lr: Learning rate for the actor (policy) network.
        critic_lr: Learning rate for the critic (value) network.
        gamma: Discount factor.
    """

    def __init__(
        self,
        state_dim: int = 8,
        n_actions: int = 4,
        hidden_dim: int = 128,
        actor_lr: float = 1e-3,
        critic_lr: float = 5e-3,
        gamma: float = 0.99,
    ) -> None:
        # -- Hyperparameters --------------------------------------------------
        self.gamma: float = gamma

        # -- Networks ---------------------------------------------------------
        self.actor = ActorNetwork(state_dim, n_actions, hidden_dim)
        self.critic = CriticNetwork(state_dim, hidden_dim)

        # -- Separate optimisers ----------------------------------------------
        # The critic typically uses a higher learning rate because accurate
        # value estimates are a prerequisite for a good policy gradient signal.
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        # -- Inter-method state -----------------------------------------------
        # select_action stores the log-probability here so that update() can
        # use it to compute the actor loss without re-running the forward pass.
        self._last_log_prob: torch.Tensor | None = None

    # ---------------------------------------------------------------------- #
    #  Action selection                                                        #
    # ---------------------------------------------------------------------- #

    def select_action(self, state: np.ndarray) -> int:
        """Sample an action from the actor and cache the log probability.

        The cached ``_last_log_prob`` will be consumed by the next call to
        ``update()``.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Actor outputs a Categorical distribution over the 4 actions
        dist = self.actor(state_tensor)
        action = dist.sample()

        # Cache the log-probability for the actor-loss computation in update()
        self._last_log_prob = dist.log_prob(action)

        return action.item()

    # ---------------------------------------------------------------------- #
    #  Learning update                                                         #
    # ---------------------------------------------------------------------- #

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> dict:
        """One-step TD Actor-Critic update.

        Performed at *every* environment step (not just episode boundaries).

        1. Compute V(s) using the critic.
        2. Compute V(s') (0 if terminal) and the TD target: r + gamma*V(s').
        3. Advantage A = TD_target - V(s).
        4. Update the critic to minimise MSE(V(s), TD_target).
        5. Update the actor using -log pi(a|s) * A.detach().

        Returns:
            Dict with ``actor_loss``, ``critic_loss``, and ``advantage``.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)

        # ---- Critic: estimate state values ----
        value = self.critic(state_t).squeeze()  # V(s), scalar

        with torch.no_grad():
            # If the episode ended, there is no next-state value
            if done:
                next_value = torch.tensor(0.0)
            else:
                next_value = self.critic(next_state_t).squeeze()  # V(s')
            # TD target = r + gamma * V(s')
            td_target = reward + self.gamma * next_value

        # ---- Advantage = TD_target - V(s) ----
        # Positive advantage means the action was better than expected;
        # negative means worse. We detach so critic gradients don't leak
        # into the actor update.
        advantage = td_target - value.detach()

        # ---- Critic loss: make V(s) closer to the TD target ----
        critic_loss = F.mse_loss(value, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- Actor loss: reinforce actions with positive advantage ----
        actor_loss = -self._last_log_prob * advantage
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "advantage": advantage.item(),
        }

    # ---------------------------------------------------------------------- #
    #  Persistence                                                             #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save both actor and critic weights in a single checkpoint."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load actor and critic weights from a checkpoint."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

    # ---------------------------------------------------------------------- #
    #  Eval / train mode                                                       #
    # ---------------------------------------------------------------------- #

    def set_eval_mode(self) -> None:
        """Set both networks to eval mode (disables dropout / batch-norm)."""
        self.actor.eval()
        self.critic.eval()

    def set_train_mode(self) -> None:
        """Set both networks back to training mode."""
        self.actor.train()
        self.critic.train()
