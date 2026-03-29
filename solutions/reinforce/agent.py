"""
REINFORCE (Monte Carlo Policy Gradient) Agent -- SOLUTION
==========================================================

REINFORCE is the simplest policy-gradient algorithm. Instead of learning a
value function and deriving a policy from it (like Q-learning / DQN), it
directly parameterises and optimises the policy pi(a|s; theta).

The policy gradient theorem tells us the direction to update theta:

    nabla J(theta) = E[ sum_t  nabla log pi(a_t | s_t; theta) * G_t ]

where G_t is the *discounted return* from timestep t onward:

    G_t = sum_{k=t}^{T} gamma^{k-t} * r_k

Because G_t can only be computed after the episode finishes, REINFORCE is
a **Monte Carlo** method -- it cannot update mid-episode.

To reduce variance we normalise the returns (subtract mean, divide by std).
This acts as a simple baseline and dramatically improves learning stability.

Reference: Sutton & Barto, Chapter 13 (Policy Gradient Methods)
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from core.base_agent import BaseAgent
from algorithms.reinforce.network import PolicyNetwork


class REINFORCEAgent(BaseAgent):
    """REINFORCE agent with return normalisation.

    The agent collects an entire episode of (log_prob, reward) pairs, then
    computes the discounted returns and performs a single policy-gradient
    update at the end of the episode.

    Args:
        state_dim: Dimension of observation space (8 for LunarLander).
        n_actions: Number of discrete actions (4 for LunarLander).
        hidden_dim: Width of hidden layers in the policy network.
        learning_rate: Adam optimiser learning rate.
        gamma: Discount factor for computing returns.
    """

    def __init__(
        self,
        state_dim: int = 8,
        n_actions: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
    ) -> None:
        # -- Hyperparameters --------------------------------------------------
        self.gamma: float = gamma

        # -- Policy network ---------------------------------------------------
        # Maps states to a Categorical distribution over actions.
        self.policy = PolicyNetwork(state_dim, n_actions, hidden_dim)

        # -- Optimiser --------------------------------------------------------
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )

        # -- Episode storage --------------------------------------------------
        # We accumulate log-probabilities and rewards throughout the episode,
        # then use them in the update at the end.
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    # ---------------------------------------------------------------------- #
    #  Action selection                                                        #
    # ---------------------------------------------------------------------- #

    def select_action(self, state: np.ndarray) -> int:
        """Sample an action from the policy and store its log probability.

        The stored log_prob is needed later to compute the policy gradient.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # PolicyNetwork.get_action returns (action_int, log_prob_tensor)
        action, log_prob = self.policy.get_action(state_tensor)
        # Store for the update at episode end
        self.log_probs.append(log_prob)
        return action

    # ---------------------------------------------------------------------- #
    #  Learning update                                                         #
    # ---------------------------------------------------------------------- #

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, terminated: bool, truncated: bool, done: bool) -> dict:
        """Accumulate rewards; update the policy when the episode finishes.

        REINFORCE is a Monte Carlo method, so the gradient step only happens
        once we have the full trajectory.

        At episode end:
            1. Compute the discounted return G_t for every timestep t.
            2. Normalise returns (zero-mean, unit-variance) to reduce variance.
            3. Compute the policy-gradient loss:
                   L = -sum_t [ log pi(a_t|s_t) * G_t_normalised ]
               The negative sign is because optimisers *minimise*, but we want
               to *maximise* the expected return.
            4. Backpropagate and step the optimiser.
            5. Clear episode storage for the next episode.

        Returns:
            ``{"loss": float}`` when the episode ends; ``{}`` otherwise.
        """
        # Accumulate this step's reward
        self.rewards.append(reward)

        # If the episode hasn't ended yet, nothing to do
        if not done:  # done = terminated or truncated
            return {}

        # ---- Compute discounted returns for each timestep ----
        returns: List[float] = []
        G: float = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # ---- Normalise returns (variance-reduction baseline) ----
        # Subtracting the mean acts as a simple baseline: actions that led
        # to above-average returns are reinforced, below-average are weakened.
        returns_tensor = (
            (returns_tensor - returns_tensor.mean())
            / (returns_tensor.std() + 1e-8)
        )

        # ---- Policy gradient loss ----
        # L = -sum_t log pi(a_t|s_t) * G_t
        loss = torch.zeros(1)
        for log_prob, G_norm in zip(self.log_probs, returns_tensor):
            # Negative because we want gradient *ascent* on returns
            loss = loss - log_prob * G_norm

        # ---- Gradient step ----
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ---- Cleanup for next episode ----
        self.log_probs.clear()
        self.rewards.clear()

        return {"loss": loss.item()}

    # ---------------------------------------------------------------------- #
    #  Persistence                                                             #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save the policy network weights."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the policy network weights."""
        self.policy.load_state_dict(torch.load(path))

    # ---------------------------------------------------------------------- #
    #  Eval / train mode                                                       #
    # ---------------------------------------------------------------------- #

    def set_eval_mode(self) -> None:
        """Set the policy network to evaluation mode.

        The policy is still stochastic (it samples from a Categorical), but
        eval mode disables any dropout or batch-norm layers.
        """
        self.policy.eval()

    def set_train_mode(self) -> None:
        """Set the policy network back to training mode."""
        self.policy.train()
