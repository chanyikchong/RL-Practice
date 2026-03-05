"""
Asynchronous Advantage Actor-Critic (A3C) Agent -- SOLUTION
=============================================================

A3C uses multiple worker processes, each running its own copy of the
environment. Workers compute gradients locally and push them to a shared
global network, which lives in shared memory. After each gradient push
the worker syncs its local weights from the global network.

Why this works:
- Different workers explore different parts of the state space
  simultaneously, decorrelating the training data (similar benefit to
  experience replay but without the memory cost).
- Asynchronous updates are surprisingly stable because the diversity of
  worker trajectories acts as an implicit regulariser.

Architecture overview:
    Main process:
        - Creates the global SharedActorCriticNetwork (share_memory)
        - Creates the shared Adam optimizer
        - Spawns N worker processes
        - Collects episode rewards via a multiprocessing Queue

    Each worker:
        - Has a LOCAL copy of the network (not shared)
        - Syncs local <- global at the start of each rollout
        - Collects n_steps of experience in its own env instance
        - Computes combined loss & local gradients
        - Copies local gradients into global_param._grad
        - Calls global_optimizer.step()
        - Repeats until the global episode counter reaches max_episodes

Reference: Mnih et al. 2016, "Asynchronous Methods for Deep RL"
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from core.base_agent import BaseAgent
from algorithms.a3c.network import SharedActorCriticNetwork


# ======================================================================== #
# Worker process
# ======================================================================== #

class A3CWorker(mp.Process):
    """A single worker that interacts with its own environment copy.

    Each worker repeatedly:
    1. Syncs its local network from the global network.
    2. Collects a short rollout (n_steps or until episode end).
    3. Computes N-step returns and advantages.
    4. Back-propagates through the LOCAL network to get gradients.
    5. Copies those gradients onto the GLOBAL network parameters.
    6. Steps the shared optimizer (updates the global weights).

    Args:
        worker_id: Unique integer id for this worker.
        global_network: The shared global network (already in shared memory).
        global_optimizer: The optimizer that operates on global_network params.
        global_episode_counter: mp.Value('i') tracking total completed episodes.
        result_queue: mp.Queue where episode rewards are sent to the main process.
        max_episodes: Training stops when global counter reaches this value.
        gamma: Discount factor.
        n_steps: Steps between gradient pushes.
        entropy_coeff: Weight for entropy bonus in the combined loss.
        value_coeff: Weight for critic loss in the combined loss.
        seed: Random seed offset so workers get different trajectories.
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
        self.worker_id = worker_id
        self.global_net = global_network
        self.global_optimizer = global_optimizer
        self.global_episode_counter = global_episode_counter
        self.result_queue = result_queue
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.seed = seed

        # Create a LOCAL network -- same architecture, but weights are NOT
        # shared across processes.  We will sync it from global before each
        # rollout.  NOTE: the environment is created inside run() because
        # gymnasium environments cannot be pickled across process boundaries.
        self.local_net = SharedActorCriticNetwork()

    # ------------------------------------------------------------------ #
    # Main worker loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Entry point executed when the process is started."""
        import gymnasium as gym

        # Each worker gets its own env instance (created here, not in __init__,
        # because gym envs are not safe to fork across processes).
        env = gym.make("LunarLander-v3")
        # Seed for reproducibility -- each worker gets a different seed
        torch.manual_seed(self.seed)

        while self.global_episode_counter.value < self.max_episodes:
            # ---- Sync local weights from the latest global weights ----
            self.local_net.sync_from(self.global_net)

            # ---- Collect one episode, updating every n_steps ----
            obs, _ = env.reset(seed=self.seed + self.global_episode_counter.value)
            done = False
            episode_reward = 0.0

            # Per-rollout storage (flushed after each gradient push)
            states: list[np.ndarray] = []
            actions: list[int] = []
            rewards: list[float] = []
            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            dones: list[bool] = []

            while not done:
                # Forward pass through the LOCAL network
                state_t = torch.FloatTensor(obs).unsqueeze(0)
                action, log_prob, value = self.local_net.get_action_and_value(state_t)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Store transition
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value.squeeze())
                dones.append(done)
                episode_reward += reward
                obs = next_obs

                # ---- Gradient push every n_steps or at episode end ----
                if len(states) >= self.n_steps or done:
                    self._push_gradients(
                        states, actions, rewards, log_probs, values, dones, next_obs, done
                    )
                    # Clear buffers for the next rollout segment
                    states.clear()
                    actions.clear()
                    rewards.clear()
                    log_probs.clear()
                    values.clear()
                    dones.clear()

                    # Re-sync after the global weights have been updated
                    self.local_net.sync_from(self.global_net)

            # ---- Episode finished ----
            with self.global_episode_counter.get_lock():
                self.global_episode_counter.value += 1
            self.result_queue.put(episode_reward)

        env.close()

    # ------------------------------------------------------------------ #
    # Gradient computation & push
    # ------------------------------------------------------------------ #

    def _push_gradients(
        self,
        states: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        log_probs: list[torch.Tensor],
        values: list[torch.Tensor],
        dones: list[bool],
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Compute local gradients and transfer them to the global network.

        Steps:
        1. Compute N-step returns with bootstrap from V(s_{t+n}).
        2. Compute advantages = returns - values.
        3. Compute combined loss (actor + value_coeff*critic - entropy_coeff*entropy).
        4. Backprop on LOCAL network.
        5. Copy local_param.grad -> global_param._grad for every parameter.
        6. Step the shared optimizer.
        """
        # ---- Bootstrap value ----
        if done:
            R = torch.tensor(0.0)
        else:
            with torch.no_grad():
                _, R = self.local_net(torch.FloatTensor(next_obs).unsqueeze(0))
                R = R.squeeze()

        # ---- N-step returns (computed backwards) ----
        returns: list = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1.0 - float(d))
            returns.insert(0, R)

        returns_t = (
            torch.stack(returns)
            if isinstance(returns[0], torch.Tensor)
            else torch.tensor(returns, dtype=torch.float32)
        )

        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)

        # ---- Advantages ----
        advantages = returns_t - values_t.detach()

        # ---- Entropy from current local policy ----
        states_t = torch.FloatTensor(np.array(states))
        dist, _ = self.local_net(states_t)
        entropy = dist.entropy().mean()

        # ---- Losses ----
        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = F.mse_loss(values_t, returns_t)
        loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy

        # ---- Local backprop ----
        self.local_net.zero_grad()
        loss.backward()

        # ---- Transfer gradients: local -> global ----
        # This is the core A3C mechanism: we computed gradients on the local
        # copy and now install them on the global parameters so the shared
        # optimizer can apply them.
        for local_param, global_param in zip(
            self.local_net.parameters(), self.global_net.parameters()
        ):
            global_param._grad = local_param.grad

        # ---- Step the global optimizer ----
        self.global_optimizer.step()


# ======================================================================== #
# Main A3C Agent
# ======================================================================== #

class A3CAgent(BaseAgent):
    """A3C agent that manages the global network and spawns worker processes.

    Usage:
        agent = A3CAgent()
        rewards = agent.train_parallel(max_episodes=1000)
        agent.set_eval_mode()
        action = agent.select_action(obs)

    Args:
        state_dim: Dimension of observation space (8 for LunarLander).
        n_actions: Number of discrete actions (4).
        hidden_dim: Hidden layer size.
        learning_rate: Shared optimizer learning rate.
        gamma: Discount factor.
        n_workers: Number of parallel worker processes.
        n_steps: Steps per worker rollout before gradient push.
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
        self.gamma = gamma
        self.n_workers = n_workers
        self.n_steps = n_steps
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff

        # ---- Global network (shared across all workers) ----
        # share_memory_() moves the underlying storage into shared memory so
        # that all worker processes can read/write the same tensors.
        self.global_network = SharedActorCriticNetwork(state_dim, n_actions, hidden_dim)
        self.global_network.share_memory()

        # ---- Shared optimizer ----
        # torch.optim.Adam works fine with shared-memory parameters.  The
        # optimizer state (momentum buffers, etc.) lives in the main process,
        # but since workers call optimizer.step() after copying their grads
        # onto the global params, this is safe in practice.
        self.optimizer = torch.optim.Adam(
            self.global_network.parameters(), lr=learning_rate
        )

        self._eval_mode = False

    # ------------------------------------------------------------------ #
    # Parallel training
    # ------------------------------------------------------------------ #

    def train_parallel(self, max_episodes: int = 1000) -> list[float]:
        """Launch worker processes and collect training rewards.

        This method replaces the usual Trainer loop. Each worker runs its
        own environment and pushes gradients to the global network
        asynchronously.

        Returns:
            List of episode rewards in order of completion (across all workers).
        """
        global_ep = mp.Value("i", 0)
        result_queue = mp.Queue()

        # Create workers
        workers = [
            A3CWorker(
                worker_id=i,
                global_network=self.global_network,
                global_optimizer=self.optimizer,
                global_episode_counter=global_ep,
                result_queue=result_queue,
                max_episodes=max_episodes,
                gamma=self.gamma,
                n_steps=self.n_steps,
                entropy_coeff=self.entropy_coeff,
                value_coeff=self.value_coeff,
                seed=i * 1000,
            )
            for i in range(self.n_workers)
        ]

        # Start all workers
        for w in workers:
            w.start()

        # Collect rewards from the result queue as workers finish episodes
        rewards: list[float] = []
        while any(w.is_alive() for w in workers) or not result_queue.empty():
            if not result_queue.empty():
                rewards.append(result_queue.get())
            else:
                # Brief sleep to avoid busy-waiting
                time.sleep(0.01)

        # Wait for all workers to terminate
        for w in workers:
            w.join()

        # Drain any remaining items in the queue
        while not result_queue.empty():
            rewards.append(result_queue.get())

        return rewards

    # ------------------------------------------------------------------ #
    # Action selection (uses the global network, typically after training)
    # ------------------------------------------------------------------ #

    def select_action(self, state: np.ndarray) -> int:
        """Select an action using the global network.

        In eval mode: pick the action with highest probability (greedy).
        In train mode: sample from the policy distribution.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.global_network(state_t)
            if self._eval_mode:
                return dist.probs.argmax().item()
            return dist.sample().item()

    def update(self, *args, **kwargs) -> dict:
        """A3C updates happen inside workers -- this is a no-op for API compat."""
        return {}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save the global network weights to disk."""
        torch.save(self.global_network.state_dict(), path)

    def load(self, path: str) -> None:
        """Load global network weights from disk."""
        self.global_network.load_state_dict(torch.load(path))

    # ------------------------------------------------------------------ #
    # Train / eval mode
    # ------------------------------------------------------------------ #

    def set_eval_mode(self) -> None:
        """Switch to greedy action selection."""
        self._eval_mode = True
        self.global_network.eval()

    def set_train_mode(self) -> None:
        """Switch back to stochastic action selection."""
        self._eval_mode = False
        self.global_network.train()
