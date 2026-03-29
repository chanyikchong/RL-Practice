"""
Proximal Policy Optimization (PPO) Agent -- SOLUTION
======================================================

PPO is one of the most popular and practical deep RL algorithms. It addresses
a fundamental tension in policy gradient methods: we want to take the largest
possible improvement step, but steps that are *too* large can catastrophically
degrade the policy (performance collapse).

PPO's solution is elegantly simple -- clip the probability ratio:

    L_CLIP = min( r(theta) * A,  clip(r(theta), 1-eps, 1+eps) * A )

where r(theta) = pi_new(a|s) / pi_old(a|s).

- When the advantage is positive (good action), the ratio is clipped at
  (1+eps), preventing the new policy from assigning *too much* more
  probability to this action.
- When the advantage is negative (bad action), the ratio is clipped at
  (1-eps), preventing the policy from reducing probability too aggressively.

PPO also uses:
- GAE (Generalized Advantage Estimation): a weighted blend of n-step
  advantage estimates that trades off bias vs. variance via lambda.
- Multiple epochs of mini-batch updates on the same collected data,
  making much more efficient use of experience than vanilla PG.

Reference: Schulman et al. 2017, "Proximal Policy Optimization Algorithms"
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core.base_agent import BaseAgent
from algorithms.ppo.network import PPOActorCriticNetwork


class PPOAgent(BaseAgent):
    """PPO agent with clipped surrogate objective and GAE.

    Args:
        state_dim: Observation dimensionality (8 for LunarLander).
        n_actions: Number of discrete actions (4).
        hidden_dim: Hidden layer size.
        learning_rate: Adam learning rate.
        gamma: Discount factor.
        gae_lambda: Lambda for Generalized Advantage Estimation.
            - lambda=0 gives pure one-step TD advantage (low variance, high bias).
            - lambda=1 gives Monte Carlo advantage (high variance, low bias).
            - lambda=0.95 is a common sweet spot.
        clip_epsilon: PPO clipping threshold (typically 0.2).
        n_steps: Number of environment steps to collect per rollout.
        n_epochs: Number of optimisation passes over the collected batch.
        mini_batch_size: Size of each mini-batch within an epoch.
        value_coeff: Weight of critic loss in the combined objective.
        entropy_coeff: Weight of the entropy bonus.
        max_grad_norm: Maximum gradient norm for clipping.
    """

    def __init__(
        self,
        state_dim: int = 8,
        n_actions: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        n_steps: int = 128,
        n_epochs: int = 4,
        mini_batch_size: int = 32,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        # -- Hyperparameters --
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm

        # -- Network and optimizer --
        self.network = PPOActorCriticNetwork(state_dim, n_actions, hidden_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        # -- Rollout buffer --
        # We store transitions here during data collection.  Crucially, we
        # also store the OLD log-probabilities so we can compute the
        # importance-sampling ratio r(theta) = pi_new / pi_old later.
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.log_probs: list[torch.Tensor] = []   # old (behaviour) log probs
        self.values: list[torch.Tensor] = []
        self.dones: list[bool] = []

    # --------------------------------------------------------------------- #
    # Action selection
    # --------------------------------------------------------------------- #

    def select_action(self, state: np.ndarray) -> int:
        """Sample an action and record old log-prob for the PPO ratio.

        We use torch.no_grad() here because these log_probs are the OLD
        (behaviour policy) values -- they are treated as constants during
        optimisation. Gradients will flow through the NEW log_probs computed
        inside evaluate_action during the update.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value = self.network.get_action_and_value(state_tensor)

        # Store for the upcoming update
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value.squeeze())

        return action

    # --------------------------------------------------------------------- #
    # Learning update
    # --------------------------------------------------------------------- #

    def update(self, state, action, reward, next_state, terminated, truncated, done) -> dict:
        """Store a transition; perform a full PPO update every n_steps.

        The PPO update proceeds as follows:

        1. Compute GAE advantages and corresponding returns.
        2. For each of n_epochs:
           a. Randomly shuffle the batch into mini-batches.
           b. For each mini-batch:
              - Re-evaluate actions with the CURRENT policy to get new
                log_probs, values, and entropy.
              - Compute the probability ratio:
                    r = exp(log_prob_new - log_prob_old)
              - Compute the clipped surrogate objective.
              - Compute the value loss and entropy bonus.
              - Backprop and clip gradients.
        3. Clear the rollout buffer.

        Returns:
            Dict of metrics when updating, empty dict otherwise.
        """
        # -- Store the transition --
        self.rewards.append(reward)
        # Only mask future returns at true termination (crash/land).
        # Truncation (time limit) still has future value — bootstrap instead.
        self.dones.append(terminated)

        # Wait until we have n_steps transitions (or episode ended)
        if len(self.states) < self.n_steps and not done:
            return {}

        # ================================================================ #
        # Step 1: Compute GAE (Generalized Advantage Estimation)
        # ================================================================ #
        with torch.no_grad():
            if terminated:
                # True end of episode: no future value
                next_value = 0.0
            else:
                _, nv = self.network(
                    torch.FloatTensor(next_state).unsqueeze(0)
                )
                next_value = nv.squeeze().item()

        # GAE builds advantages as an exponentially-weighted sum of TD errors:
        #
        #   delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)          (one-step TD error)
        #   A_t     = delta_t + (gamma * lambda) * delta_{t+1}
        #             + (gamma * lambda)^2 * delta_{t+2} + ...
        #
        # This is computed efficiently by iterating backwards:
        #   gae = delta_t + (gamma * lambda) * gae_{t+1}
        #
        # The parameter lambda controls the bias-variance trade-off:
        # - lambda=0 => A_t = delta_t (pure TD, low variance, higher bias)
        # - lambda=1 => A_t = sum of discounted rewards - V(s_t) (MC, high variance)
        values_list = [v.item() for v in self.values] + [next_value]

        advantages = []
        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            # Mask: if the episode ended at step t, there is no future reward
            not_done = 1.0 - float(self.dones[t])

            # One-step TD error
            delta = (
                self.rewards[t]
                + self.gamma * values_list[t + 1] * not_done
                - values_list[t]
            )

            # Accumulate the GAE
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages.insert(0, gae)

        advantages_t = torch.tensor(advantages, dtype=torch.float32)

        # Returns = advantages + baseline values  (used as critic targets)
        returns_t = advantages_t + torch.tensor(
            values_list[:-1], dtype=torch.float32
        )

        # ================================================================ #
        # Step 2: Prepare batch tensors
        # ================================================================ #
        states_t = torch.FloatTensor(np.array(self.states))
        actions_t = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs_t = torch.stack(self.log_probs)

        # Normalize advantages and returns.
        # Advantages: zero-mean, unit-variance across the rollout batch.
        # Returns: same normalization so that critic MSE stays ~O(1) and
        # doesn't dwarf the actor loss (~250:1 ratio without this).
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std() + 1e-8
            )

        # ================================================================ #
        # Step 3: Multiple epochs of mini-batch updates
        # ================================================================ #
        batch_size = len(states_t)
        total_loss_accum = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        clip_fractions: list[float] = []
        num_updates = 0

        for _epoch in range(self.n_epochs):
            # Shuffle indices for this epoch
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                # Slice the mini-batch
                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                # ---- New forward pass with current policy ----
                # evaluate_action returns (new_log_probs, new_values, entropy)
                new_log_probs, new_values, entropy = self.network.evaluate_action(
                    mb_states, mb_actions
                )

                # ---- Probability ratio ----
                # r(theta) = pi_new(a|s) / pi_old(a|s)
                # In log space: log(r) = log_prob_new - log_prob_old
                # Detach old log probs so no gradient flows through them.
                ratio = torch.exp(new_log_probs - mb_old_log_probs.detach())

                # ---- Clipped surrogate objective ----
                # surr1: the standard policy gradient objective (with IS ratio)
                surr1 = ratio * mb_advantages

                # surr2: same, but with the ratio clipped to [1-eps, 1+eps]
                # This prevents large policy updates. When the advantage is
                # positive, clipping at (1+eps) caps the reward for moving
                # probability towards this action. When negative, clipping at
                # (1-eps) caps the penalty.
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * mb_advantages
                )

                # Take the pessimistic (lower) bound -- this is conservative,
                # ensuring we don't over-optimise in either direction.
                actor_loss = -torch.min(surr1, surr2).mean()

                # ---- Value (critic) loss ----
                critic_loss = F.mse_loss(new_values.squeeze(-1), mb_returns)

                # ---- Entropy bonus ----
                # Higher entropy = more exploration.  We subtract it from the
                # loss so the optimizer *maximises* entropy.
                entropy_mean = entropy.mean()

                # ---- Combined loss ----
                loss = (
                    actor_loss
                    + self.value_coeff * critic_loss
                    - self.entropy_coeff * entropy_mean
                )

                # ---- Gradient step ----
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # ---- Tracking metrics ----
                total_loss_accum += loss.item()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy_mean.item()
                num_updates += 1

                # Clip fraction: proportion of samples where the ratio was
                # actually clipped.  Useful for monitoring -- if this is very
                # high, the policy is changing too fast.
                with torch.no_grad():
                    clip_frac = (
                        ((ratio - 1.0).abs() > self.clip_epsilon)
                        .float()
                        .mean()
                        .item()
                    )
                    clip_fractions.append(clip_frac)

        # ================================================================ #
        # Step 4: Clear rollout buffer
        # ================================================================ #
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

        # Return averaged metrics
        n = max(num_updates, 1)
        return {
            "loss": total_loss_accum / n,
            "actor_loss": total_actor_loss / n,
            "critic_loss": total_critic_loss / n,
            "entropy": total_entropy / n,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
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
        """Switch to evaluation mode (affects dropout/batchnorm if present)."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.network.eval()

    def set_train_mode(self) -> None:
        """Switch back to training mode."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.network.train()
