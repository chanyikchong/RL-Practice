# LunarLander-v2 Environment Reference Card

Keep this open while implementing your algorithms!

## State Space (8 dimensions)

| Index | Name               | Range (approx)   | Description                                 |
|-------|--------------------|-------------------|---------------------------------------------|
| 0     | `x_position`       | [-1.5, 1.5]       | Horizontal position. 0 = landing pad center |
| 1     | `y_position`       | [0, 1.5]          | Vertical position. 0 = ground level         |
| 2     | `x_velocity`       | [-5, 5]           | Horizontal velocity                         |
| 3     | `y_velocity`       | [-5, 5]           | Vertical velocity (negative = falling)      |
| 4     | `angle`            | [-pi, pi]         | Lander angle in radians. 0 = upright        |
| 5     | `angular_velocity` | [-5, 5]           | Rotation speed                              |
| 6     | `left_leg_contact` | {0, 1}            | 1 if left leg touches ground                |
| 7     | `right_leg_contact`| {0, 1}            | 1 if right leg touches ground               |

## Action Space (4 discrete actions)

| Action | Name             | Effect                        |
|--------|------------------|-------------------------------|
| 0      | Do nothing       | No engine fires               |
| 1      | Fire left engine | Pushes lander right           |
| 2      | Fire main engine | Pushes lander up (costs fuel) |
| 3      | Fire right engine| Pushes lander left            |

## Reward Structure (default gymnasium)

The default reward at each step includes:
- **Shaping reward**: Proportional to moving closer to pad and reducing speed
- **Fuel cost**: -0.03 per side engine fire, -0.3 per main engine fire
- **Leg contact**: +10 per leg touching ground
- **Landing**: +100 for landing safely
- **Crash**: -100 for crashing

An episode is **solved** when average reward >= 200 over 100 episodes.

## Derived State Helpers (via `StateInfo`)

| Property            | Type   | Description                                   |
|---------------------|--------|-----------------------------------------------|
| `distance_to_target`| float  | Euclidean distance from (0, 0) landing pad    |
| `speed`             | float  | Total speed magnitude sqrt(vx^2 + vy^2)      |
| `is_upright`        | bool   | True if |angle| < 0.1 rad (~5.7 degrees)      |
| `both_legs_down`    | bool   | True if both legs are touching the ground     |

## Reward Signals (via `RewardSignals`)

These pre-computed components let you build custom rewards:

| Signal               | Sign | Description                                    |
|----------------------|------|------------------------------------------------|
| `proximity_reward`   | +    | Higher when closer to landing pad              |
| `velocity_penalty`   | -    | Penalty for high speed (stronger near ground)  |
| `angle_penalty`      | -    | Penalty for tilting away from upright           |
| `fuel_penalty`       | -    | Cost of firing engines                         |
| `landing_bonus`      | +    | +100 for successful landing                    |
| `crash_penalty`      | -    | -100 for crashing                              |
| `leg_contact_reward` | +    | +10 per leg touching ground                    |
| `default_gymnasium_reward` | +/- | Raw gymnasium reward for reference        |

## Usage Example

```python
from environment import LunarLanderEnv

env = LunarLanderEnv()
obs, state_info = env.reset()

print(f"Starting at x={state_info.x_position:.2f}, y={state_info.y_position:.2f}")

obs, reward, terminated, truncated, state_info, reward_signals = env.step(2)

print(f"Gym reward: {reward_signals.default_gymnasium_reward:.2f}")
print(f"Distance to pad: {state_info.distance_to_target:.2f}")
print(f"Speed: {state_info.speed:.2f}")
```

## Tips for Discretization (Q-Learning)

Suggested bin boundaries for each dimension:

| Dimension          | Suggested bins                           |
|--------------------|------------------------------------------|
| x_position         | [-1.5, -0.5, -0.1, 0.1, 0.5, 1.5]      |
| y_position         | [0, 0.1, 0.3, 0.6, 1.0, 1.5]           |
| x_velocity         | [-1.0, -0.3, 0.0, 0.3, 1.0]            |
| y_velocity         | [-1.0, -0.5, -0.2, 0.0, 0.5]           |
| angle              | [-0.5, -0.1, 0.0, 0.1, 0.5]            |
| angular_velocity   | [-0.5, -0.1, 0.0, 0.1, 0.5]            |
| left_leg_contact   | {0, 1} (already discrete)                |
| right_leg_contact  | {0, 1} (already discrete)                |
