"""
Asynchronous Advantage Actor-Critic (A3C) Agent
==================================================

Key idea: Run multiple workers in parallel, each interacting with their
own copy of the environment. Each worker computes gradients on a local
network copy, then pushes those gradients to a shared global network.
Workers periodically sync their local weights from the global network.

This achieves two things:
1. Diverse experience: workers explore different parts of state space
2. Stability: decorrelates updates (similar benefit to experience replay)

Architecture:
- GlobalNetwork: shared parameters (must be in shared memory)
- Worker(Process): each has a local network, local env, computes gradients
- Workers push gradients to global optimizer, then sync local <- global

Recommended reading: Mnih et al. 2016 "Asynchronous Methods for Deep RL"

See algorithms/a3c/network.py for SharedActorCriticNetwork with sync_from().
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from core.base_agent import BaseAgent
from .network import SharedActorCriticNetwork


class A3CWorker(mp.Process):
    """A worker process that collects experience and computes gradients.

    Each worker has:
    - Its own copy of the environment
    - A local network (synced from global before each rollout)
    - Access to the global network and optimizer (via shared memory)

    Args:
        worker_id: Unique identifier for this worker.
        global_network: The shared global network (in shared memory).
        global_optimizer: The shared optimizer.
        global_episode_counter: Shared counter for total episodes.
        result_queue: Queue to send training metrics back to main process.
        max_episodes: Total episodes across all workers.
        gamma: Discount factor.
        n_steps: Steps per rollout before computing gradients.
        entropy_coeff: Weight for entropy bonus.
        value_coeff: Weight for critic loss.
        seed: Random seed offset for this worker.
    """

    def __init__(
        self,
        worker_id: int,
        global_network: SharedActorCriticNetwork,
        global_optimizer: torch.optim.Optimizer,
        global_episode_counter: mp.Value,
        result_queue: mp.Queue,
        max_episodes: int = 1000,
        gamma: float = 0.99,
        n_steps: int = 5,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        seed: int = 0,
    ):
        super().__init__()
        # TODO: Store all parameters
        # TODO: Create local network: SharedActorCriticNetwork()
        # NOTE: Do NOT create the environment here — create it in run()
        #       because gym envs don't serialize across processes
        raise NotImplementedError("Implement Worker.__init__")

    def run(self) -> None:
        """Main worker loop: collect experience, compute gradients, push to global.

        Loop:
        1. Sync local network from global: self.local_net.sync_from(self.global_net)
        2. Collect N steps of experience using local network
        3. Compute N-step returns and advantages
        4. Compute combined loss (actor + critic + entropy)
        5. Compute gradients on LOCAL network
        6. Push gradients to GLOBAL network:
           for local_p, global_p in zip(local_net.parameters(), global_net.parameters()):
               global_p._grad = local_p.grad
        7. Global optimizer step
        8. Repeat until max_episodes reached
        """
        # TODO: Implement the worker loop
        #
        # import gymnasium as gym
        # env = gym.make("LunarLander-v3")
        #
        # while self.global_episode_counter.value < self.max_episodes:
        #     # Sync local <- global
        #     self.local_net.sync_from(self.global_net)
        #
        #     # Collect experience
        #     states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        #     obs, _ = env.reset()
        #     done = False
        #     episode_reward = 0
        #
        #     while not done:
        #         state_t = torch.FloatTensor(obs).unsqueeze(0)
        #         action, log_prob, value = self.local_net.get_action_and_value(state_t)
        #
        #         next_obs, reward, terminated, truncated, _ = env.step(action)
        #         done = terminated or truncated
        #
        #         states.append(obs); actions.append(action); rewards.append(reward)
        #         log_probs.append(log_prob); values.append(value.squeeze()); dones.append(done)
        #         episode_reward += reward
        #         obs = next_obs
        #
        #         # Update every n_steps or at episode end
        #         if len(states) >= self.n_steps or done:
        #             # Compute returns
        #             if done:
        #                 R = torch.tensor(0.0)
        #             else:
        #                 with torch.no_grad():
        #                     _, R = self.local_net(torch.FloatTensor(next_obs).unsqueeze(0))
        #                     R = R.squeeze()
        #
        #             returns = []
        #             for r, d in zip(reversed(rewards), reversed(dones)):
        #                 R = r + self.gamma * R * (1 - float(d))
        #                 returns.insert(0, R)
        #             returns = torch.stack(returns) if isinstance(returns[0], torch.Tensor) else torch.tensor(returns)
        #
        #             values_t = torch.stack(values)
        #             log_probs_t = torch.stack(log_probs)
        #             advantages = returns - values_t.detach()
        #
        #             # Entropy
        #             states_t = torch.FloatTensor(np.array(states))
        #             dist, _ = self.local_net(states_t)
        #             entropy = dist.entropy().mean()
        #
        #             # Loss
        #             actor_loss = -(log_probs_t * advantages).mean()
        #             critic_loss = F.mse_loss(values_t, returns)
        #             loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
        #
        #             # Compute local gradients
        #             self.local_net.zero_grad()
        #             loss.backward()
        #
        #             # Push gradients to global
        #             for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
        #                 gp._grad = lp.grad
        #             self.global_optimizer.step()
        #
        #             # Sync again
        #             self.local_net.sync_from(self.global_net)
        #
        #             # Clear storage
        #             states.clear(); actions.clear(); rewards.clear()
        #             log_probs.clear(); values.clear(); dones.clear()
        #
        #     # Episode finished
        #     with self.global_episode_counter.get_lock():
        #         self.global_episode_counter.value += 1
        #     self.result_queue.put(episode_reward)
        #
        # env.close()
        raise NotImplementedError("Implement Worker.run: async gradient computation")


class A3CAgent(BaseAgent):
    """A3C agent that manages global network and spawns worker processes.

    Args:
        state_dim: Dimension of observation space (8).
        n_actions: Number of discrete actions (4).
        hidden_dim: Hidden layer size.
        learning_rate: Optimizer learning rate.
        gamma: Discount factor.
        n_workers: Number of parallel workers.
        n_steps: Steps per worker rollout.
        entropy_coeff: Entropy bonus coefficient.
        value_coeff: Value loss coefficient.
    """

    def __init__(
        self,
        state_dim: int = 8,
        n_actions: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 7e-4,
        gamma: float = 0.99,
        n_workers: int = 4,
        n_steps: int = 5,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
    ):
        # TODO: Store hyperparameters
        # TODO: Create global network and put in shared memory:
        #   self.global_network = SharedActorCriticNetwork(...)
        #   self.global_network.share_memory()
        # TODO: Create shared optimizer (use torch.optim.Adam)
        # TODO: self._eval_mode = False
        raise NotImplementedError("Implement __init__: create global network in shared memory")

    def train_parallel(self, max_episodes: int = 1000) -> list[float]:
        """Launch workers and train in parallel.

        This replaces the normal Trainer loop for A3C.

        Steps:
        1. Create shared episode counter and result queue
        2. Create and start n_workers Worker processes
        3. Collect episode rewards from result queue
        4. Join all workers when done

        Returns:
            List of episode rewards (in order of completion).
        """
        # TODO: Implement parallel training
        #
        # global_ep = mp.Value('i', 0)
        # result_queue = mp.Queue()
        #
        # workers = [
        #     A3CWorker(
        #         worker_id=i,
        #         global_network=self.global_network,
        #         global_optimizer=self.optimizer,
        #         global_episode_counter=global_ep,
        #         result_queue=result_queue,
        #         max_episodes=max_episodes,
        #         gamma=self.gamma,
        #         n_steps=self.n_steps,
        #         entropy_coeff=self.entropy_coeff,
        #         value_coeff=self.value_coeff,
        #         seed=i * 1000,
        #     )
        #     for i in range(self.n_workers)
        # ]
        #
        # for w in workers: w.start()
        # rewards = []
        # while any(w.is_alive() for w in workers) or not result_queue.empty():
        #     if not result_queue.empty():
        #         rewards.append(result_queue.get())
        # for w in workers: w.join()
        # return rewards
        raise NotImplementedError("Implement train_parallel: launch workers")

    def select_action(self, state: np.ndarray) -> int:
        """Use global network for action selection (after training)."""
        # TODO: Forward pass through global_network
        # state_t = torch.FloatTensor(state).unsqueeze(0)
        # with torch.no_grad():
        #     dist, _ = self.global_network(state_t)
        #     if self._eval_mode:
        #         return dist.probs.argmax().item()
        #     return dist.sample().item()
        raise NotImplementedError("Implement select_action")

    def update(self, *args, **kwargs) -> dict:
        """A3C updates happen inside workers. This is a no-op for compatibility."""
        return {}

    def save(self, path: str) -> None:
        # TODO: torch.save(self.global_network.state_dict(), path)
        raise NotImplementedError("Implement save")

    def load(self, path: str) -> None:
        # TODO: self.global_network.load_state_dict(torch.load(path))
        raise NotImplementedError("Implement load")

    def set_eval_mode(self) -> None:
        # TODO: self._eval_mode = True; self.global_network.eval()
        raise NotImplementedError("Implement set_eval_mode")

    def set_train_mode(self) -> None:
        # TODO: self._eval_mode = False; self.global_network.train()
        raise NotImplementedError("Implement set_train_mode")
