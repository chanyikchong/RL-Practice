"""
LunarLander-v2 environment wrapper with rich state and reward information.

This wrapper makes the environment's internal signals transparent, so learners
can understand exactly what's happening at each step and experiment with
custom reward shaping.

Actions:
    0: Do nothing
    1: Fire left engine
    2: Fire main engine
    3: Fire right engine
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class StateInfo:
    """Named state variables extracted from the raw observation vector.

    The raw observation is an 8-dimensional float vector. This dataclass
    gives each dimension a human-readable name plus useful derived values.
    """

    x_position: float        # horizontal position (-1 to 1, 0 = center of landing pad)
    y_position: float        # vertical position (0 = ground level)
    x_velocity: float        # horizontal velocity
    y_velocity: float        # vertical velocity
    angle: float             # lander angle in radians (0 = upright)
    angular_velocity: float  # rotation speed
    left_leg_contact: bool   # True if left leg is touching the ground
    right_leg_contact: bool  # True if right leg is touching the ground

    # Derived helpers
    distance_to_target: float  # Euclidean distance from landing pad center (0, 0)
    speed: float               # total speed magnitude
    is_upright: bool           # True if |angle| < 0.1 rad (~5.7 degrees)
    both_legs_down: bool       # True if both legs are touching the ground

    @classmethod
    def from_obs(cls, obs: np.ndarray) -> StateInfo:
        """Create a StateInfo from the raw 8-dim observation array."""
        x, y = float(obs[0]), float(obs[1])
        vx, vy = float(obs[2]), float(obs[3])
        angle = float(obs[4])
        ang_vel = float(obs[5])
        left_leg = bool(obs[6])
        right_leg = bool(obs[7])

        return cls(
            x_position=x,
            y_position=y,
            x_velocity=vx,
            y_velocity=vy,
            angle=angle,
            angular_velocity=ang_vel,
            left_leg_contact=left_leg,
            right_leg_contact=right_leg,
            distance_to_target=math.sqrt(x**2 + y**2),
            speed=math.sqrt(vx**2 + vy**2),
            is_upright=abs(angle) < 0.1,
            both_legs_down=left_leg and right_leg,
        )


@dataclass
class RewardSignals:
    """Decomposed reward components.

    The learner can combine these however they like for custom reward shaping,
    or just use ``default_gymnasium_reward`` to get the standard signal.
    """

    proximity_reward: float          # +reward for being close to pad (higher = closer)
    velocity_penalty: float          # -penalty for moving fast near ground
    angle_penalty: float             # -penalty for tilting away from upright
    fuel_penalty: float              # -penalty for firing engines (action != 0)
    landing_bonus: float             # +100 for successful landing (both legs, low speed)
    crash_penalty: float             # -100 for crashing
    leg_contact_reward: float        # +10 per leg touching ground
    default_gymnasium_reward: float  # raw gymnasium reward, for reference


class LunarLanderEnv:
    """Wrapped LunarLander-v2 that exposes rich state and reward information.

    Usage::

        env = LunarLanderEnv()
        obs, state_info = env.reset()
        obs, reward, terminated, truncated, state_info, reward_signals = env.step(action)

    The ``obs`` array is always the raw 8-dim vector suitable for neural nets.
    ``state_info`` and ``reward_signals`` are dataclasses for inspection.
    """

    # Action names for display
    ACTION_NAMES = {0: "Do nothing", 1: "Fire left", 2: "Fire main", 3: "Fire right"}

    def __init__(self, render_mode: str | None = None, env_id: str = "LunarLander-v3") -> None:
        self._env = gym.make(env_id, render_mode=render_mode)
        self._prev_state_info: StateInfo | None = None

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, StateInfo]:
        """Reset the environment and return (obs, state_info)."""
        obs, info = self._env.reset(seed=seed)
        state_info = StateInfo.from_obs(obs)
        self._prev_state_info = state_info
        return obs, state_info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, StateInfo, RewardSignals]:
        """Take a step and return (obs, reward, terminated, truncated, state_info, reward_signals).

        Args:
            action: 0=nothing, 1=left engine, 2=main engine, 3=right engine

        Returns:
            obs: Raw 8-dim observation array
            reward: The default gymnasium reward (use reward_signals for decomposition)
            terminated: True if episode ended (landed or crashed)
            truncated: True if episode hit time limit
            state_info: Named state variables
            reward_signals: Decomposed reward components
        """
        obs, gym_reward, terminated, truncated, info = self._env.step(action)
        state_info = StateInfo.from_obs(obs)

        # Compute decomposed reward signals
        reward_signals = self._compute_reward_signals(
            state_info, action, gym_reward, terminated
        )

        self._prev_state_info = state_info
        return obs, gym_reward, terminated, truncated, state_info, reward_signals

    def _compute_reward_signals(
        self,
        state: StateInfo,
        action: int,
        gym_reward: float,
        terminated: bool,
    ) -> RewardSignals:
        """Decompose the reward into interpretable components."""
        # Proximity: inversely proportional to distance from pad
        proximity = max(0.0, 1.0 - state.distance_to_target)

        # Velocity penalty: penalize high speed, especially near ground
        height_factor = max(0.0, 1.0 - state.y_position)  # stronger near ground
        vel_penalty = -state.speed * height_factor * 0.1

        # Angle penalty
        angle_penalty = -abs(state.angle) * 0.5

        # Fuel penalty: any engine firing costs fuel
        fuel_penalty = -0.03 if action != 0 else 0.0
        if action == 2:  # main engine costs more
            fuel_penalty = -0.3

        # Terminal bonuses/penalties
        landing_bonus = 0.0
        crash_penalty = 0.0
        if terminated:
            if state.both_legs_down and state.speed < 0.5:
                landing_bonus = 100.0
            elif state.speed > 1.0 or abs(state.angle) > 0.5:
                crash_penalty = -100.0

        # Leg contact
        leg_reward = 0.0
        if state.left_leg_contact:
            leg_reward += 10.0
        if state.right_leg_contact:
            leg_reward += 10.0

        return RewardSignals(
            proximity_reward=proximity,
            velocity_penalty=vel_penalty,
            angle_penalty=angle_penalty,
            fuel_penalty=fuel_penalty,
            landing_bonus=landing_bonus,
            crash_penalty=crash_penalty,
            leg_contact_reward=leg_reward,
            default_gymnasium_reward=gym_reward,
        )

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._env.action_space

    def render_frame(self) -> np.ndarray:
        """Return an RGB frame of the current state (requires render_mode='rgb_array')."""
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
