"""
Advantage Actor-Critic (A2C) Agent -- SOLUTION
================================================

A2C is the synchronous variant of A3C. Instead of updating every single step
(like vanilla Actor-Critic), A2C collects N steps of experience before
performing a single update. This introduces a spectrum between one-step TD
(low variance, high bias) and full Monte Carlo (high variance, low bias).

Key components:
1. N-step returns -- collect N transitions, then bootstrap from V(s_{t+n}).
2. Entropy bonus -- prevents the policy from collapsing to a deterministic
   mapping too early, encouraging continued exploration.
3. Combined loss -- a single optimizer handles both actor and critic via:
       L = L_actor + 0.5 * L_critic - 0.01 * H(pi)

Reference: Mnih et al. 2016, "Asynchronous Methods for Deep RL"
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core.base_agent import BaseAgent
from algorithms.a2c.network import ActorCriticNetwork


class A2CAgent(BaseAgent):
    """Advantage Actor-Critic with N-step returns and entropy regularization.

    Args:
        state_dim: Dimension of observation space (default 8 for LunarLander).
        n_actions: Number of discrete actions (default 4).
        hidden_dim: Hidden layer size for the shared network.
        learning_rate: Optimizer learning rate.
        gamma: Discount factor for future rewards.
        n_steps: Number of steps to collect before each parameter update.
        value_coeff: Weight of critic (value) loss in the combined objective.
        entropy_coeff: Weight of the entropy bonus (higher = more exploration).
    """

    def __init__(
        self,
        state_dim: int = 8,
        n_actions: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 7e-4,
        gamma: float = 0.99,
        n_steps: int = 5,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
    ):
        # -- Hyperparameters --
        self.gamma = gamma
        self.n_steps = n_steps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

        # -- Network: shared feature extractor with separate actor/critic heads --
        self.network = ActorCriticNetwork(state_dim, n_actions, hidden_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        # -- N-step rollout storage --
        # We accumulate transitions here and flush them every n_steps (or at
        # the end of an episode). log_probs and values come from the forward
        # pass during select_action so the computation graph is preserved for
        # backpropagation.
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.dones: list[bool] = []

    # --------------------------------------------------------------------- #
    # Action selection
    # --------------------------------------------------------------------- #

    def select_action(self, state: np.ndarray) -> int:
        """Sample an action from the current policy.

        A single forward pass through the shared network yields both the
        action distribution (actor head) and the state-value estimate
        (critic head). We store log_prob and value for the upcoming update.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # get_action_and_value returns (action_int, log_prob, value)
        action, log_prob, value = self.network.get_action_and_value(state_tensor)

        # Keep tensors attached to the graph so gradients flow during update
        self.log_probs.append(log_prob)
        self.values.append(value.squeeze())

        return action

    # --------------------------------------------------------------------- #
    # Learning update
    # --------------------------------------------------------------------- #

    def update(self, state, action, reward, next_state, done) -> dict:
        """Store a transition and, every N steps (or at episode end), update.

        The update performs the following:
        1. Compute N-step returns by bootstrapping from V(s_{t+n}):
               G_t = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1}
                     + gamma^n * V(s_{t+n})
           When an episode terminates, the bootstrap value is 0.

        2. Compute advantages: A_t = G_t - V(s_t)
           Advantages are normalised to stabilize training.

        3. Compute three loss terms:
           - Actor loss:  -mean(log_prob * advantage)
           - Critic loss: MSE between predicted V(s) and computed returns
           - Entropy:     mean entropy of the policy distribution

        4. Combine:  L = L_actor + value_coeff * L_critic - entropy_coeff * H

        5. Clip gradient norms at 0.5 to prevent exploding gradients.

        Returns:
            Dict of training metrics when an update is performed, else {}.
        """
        # -- Store the transition --
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

        # Only update every n_steps, or when the episode ends
        if len(self.states) < self.n_steps and not done:
            return {}

        # ---- Bootstrap value for the state after the last stored step ----
        # If the episode ended, the future value is 0; otherwise we estimate
        # it with the critic.
        if done:
            next_value = torch.tensor(0.0)
        else:
            with torch.no_grad():
                _, next_value = self.network(
                    torch.FloatTensor(next_state).unsqueeze(0)
                )
                next_value = next_value.squeeze()

        # ---- Compute N-step returns backwards ----
        # Starting from the bootstrap, we walk backwards through the stored
        # rewards:  G_t = r_t + gamma * G_{t+1}  (with masking for dones)
        returns: list[torch.Tensor] = []
        G = next_value
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            G = r + self.gamma * G * (1.0 - float(d))
            returns.insert(0, G)

        # Convert to tensors
        returns_t = (
            torch.stack(returns)
            if isinstance(returns[0], torch.Tensor)
            else torch.tensor(returns, dtype=torch.float32)
        )

        # Stack the log_probs and values collected during select_action
        values_t = torch.stack(self.values)
        log_probs_t = torch.stack(self.log_probs)

        # ---- Advantages ----
        # A_t = G_t - V(s_t).  We detach values so critic gradients come
        # only from the critic loss, not through the advantage in the actor loss.
        advantages = returns_t - values_t.detach()

        # Normalise advantages (zero mean, unit std) to reduce sensitivity to
        # reward scale and improve training stability.
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- Entropy ----
        # Re-run forward pass on collected states to obtain the current policy
        # distribution (needed for the entropy bonus).
        states_t = torch.FloatTensor(np.array(self.states))
        dist, _ = self.network(states_t)
        entropy = dist.entropy().mean()
        # Entropy of a Categorical measures how "spread out" the distribution
        # is. Maximising it discourages premature convergence to a single action.

        # ---- Losses ----
        # Actor (policy gradient):  -E[log pi(a|s) * A(s,a)]
        actor_loss = -(log_probs_t * advantages).mean()

        # Critic (value regression): MSE between V(s) predictions and targets
        critic_loss = F.mse_loss(values_t, returns_t)

        # Combined objective.  The minus sign on entropy means we *maximise* it.
        total_loss = (
            actor_loss
            + self.value_coeff * critic_loss
            - self.entropy_coeff * entropy
        )

        # ---- Gradient step ----
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping prevents catastrophic updates from noisy rollouts
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)

        self.optimizer.step()

        # ---- Clear rollout storage for the next batch ----
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

        return {
            "loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
        }

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save network weights to disk."""
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        """Load network weights from disk."""
        self.network.load_state_dict(torch.load(path))

    # --------------------------------------------------------------------- #
    # Train / eval mode
    # --------------------------------------------------------------------- #

    def set_eval_mode(self) -> None:
        """Switch to evaluation mode (disables dropout, batchnorm training)."""
        self.network.eval()

    def set_train_mode(self) -> None:
        """Switch back to training mode."""
        self.network.train()
