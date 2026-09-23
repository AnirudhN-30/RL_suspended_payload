# FLARE payload task

This is the GPU-parallel MJLab/RSL-RL implementation of the ACP quadrotor
payload task. The native MuJoCo reference remains in the sibling
`flare_quadrotor_mujoco` repository; use it for controller/model validation and
this task for training.

## Plant

- quadrotor mass: 1.05 kg
- payload mass: 0.1128 kg
- tendon length limit: 0.60 m
- simulation: MuJoCo Warp, `implicitfast`, 0.002 s physics step
- policy period: 0.01 s (five physics steps)
- motors: M1 rear-right, M2 front-right, M3 rear-left, M4 front-left
- four physical site actuators, each limited to 20.5 N

The quadrotor and payload are deliberately separate MJCF entities
(`quadrotor.xml` and `payload.xml`). The scene configuration connects their
attachment sites with the spatial tendon. This separation makes payload swaps
and later domain randomization straightforward.

## FLARE interface

The actor and critic each receive the released 26-value FLARE observation:
current waypoint relative position (3), next waypoint relative position (3),
world linear velocity (3), row-major body-to-world rotation matrix (9), previous
normalized action (4), and cable pitch/roll with finite-difference rates (4).

The policy outputs four normalized values in `[-1, 1]`: collective throttle,
desired body roll rate, pitch rate, and yaw rate. A batched low-level rate
controller and exact motor mixer turn these into the four rotor thrusts. The
policy does not output a wrench and there is no position PID between the policy
and vehicle.

The task keeps the six released FLARE Scenario I reward functions. The default
smooth-v2 profile uses target progress `5`, action smoothness `-0.01`, yaw
`0.01`, angular rate `-0.005`, crash `-20`, and cable safety `1.0`. Rewards are
scaled by the 0.01 s policy period. The completed first tuning profile remains
available through `Mjlab-Flare-Waypoint-Payload-ACP-Tuned-V1`, and the original
FLARE coefficients through `Mjlab-Flare-Waypoint-Payload-Flare-Rewards`.

## Payload waypoint navigation

`Mjlab-Flare-Payload-Targeting` is the payload-navigation variant. It deliberately
retains the existing 26-value observation, four CTBR actions, rate PID, motor
mixer, and ACP-smooth body-rate penalty. The active waypoint is considered
reached when the **payload** enters a 0.2 m sphere, and target progress is
computed from the payload position. Current and next targets are sampled from
the paper's `[-2, 2] x [-2, 2] x [0.5, 1.5] m` volume relative to the quadrotor.
Action smoothness uses the paper's L2 norm of consecutive action differences.
The separate task keeps existing Scenario-I checkpoints compatible.

## Setup and checks

The repository is pinned to Python 3.10, MuJoCo 3.6, Warp 1.12, and the CUDA
12.8 PyTorch extra because the former nightly MuJoCo wheel in the cloned lockfile
no longer exists.

```bash
cd /home/anirudh/.openclaw/workspace/mjlab
uv sync --python /usr/bin/python3.10 --extra cu128
uv run --extra cu128 python scripts/verify_flare_payload.py
uv run --extra cu128 python scripts/verify_flare_payload_targeting.py
uv run --extra cu128 python scripts/benchmark_flare_payload.py --num-envs 4096
```

The verification checks the 26/4 interface, finite observations and rewards,
tendon length, loaded hover motor thrust, and CUDA execution.

## Train

```bash
cd /home/anirudh/.openclaw/workspace/mjlab
uv run --extra cu128 train Mjlab-Flare-Waypoint-Payload \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard
```

Training uses RSL-RL PPO with a `26 -> 128 -> 128 -> 4` tanh actor and a matching
critic. The smooth-v2 profile uses learning rate `1e-4` and entropy coefficient
`0.0005`; the remaining PPO settings match the FLARE baseline. Checkpoints and
TensorBoard logs are written below `logs/rsl_rl/flare_payload_acp_smooth_v2/`. This
first version intentionally has no plant or sensor randomization; add that only
after the nominal policy learns reliably.

Train payload waypoint navigation from scratch with:

```bash
uv run --extra cu128 train Mjlab-Flare-Payload-Targeting \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard
```

Its checkpoints are written below
`logs/rsl_rl/flare_payload_targeting_26d/`. The actor uses a final tanh
projection, while Gaussian exploration and the downstream action clipping stay
unchanged.

Evaluate or visualize a trained payload-targeting checkpoint with:

```bash
uv run --extra cu128 python eval/evaluate_flare.py \
  --task-id Mjlab-Flare-Payload-Targeting \
  --checkpoint logs/rsl_rl/flare_payload_targeting_26d/<run>/model_<iteration>.pt

uv run --extra cu128 python eval/play_flare_trajectory.py \
  --task-id Mjlab-Flare-Payload-Targeting \
  --checkpoint logs/rsl_rl/flare_payload_targeting_26d/<run>/model_<iteration>.pt \
  --trajectory hexagon --viewer native
```

For this task, evaluation distance and waypoint counts are measured from the
payload rather than the quadrotor.

## Visualize

```bash
uv run --extra cu128 play Mjlab-Flare-Waypoint-Payload --agent random
```

For a calm plant visualization and low-level-controller tests, use the viewer in
the sibling native reference repository instead; random policy actions are
aggressive by design.


New changes
Trained once with gains as: 08-11
    "target": 10.0,
    "smooth": -0.05,
    "yaw": 0.01,
    "angular": -0.01,
    "crash": -20.0,
    "cable_angle_safety": 1.0.

Trained once again with gains as: 08-12
    "target": 10.0,
    "smooth": -0.1,
    "yaw": 0.01,
    "angular": -0.015,
    "crash": -20.0,
    "cable_angle_safety": 1.3,    

Trained again for 0.2m arrival distance with gains as: 08-16
    "target": 10.0,
    "smooth": -0.05,
    "yaw": 0.01,
    "angular": -0.01,
    "crash": -20.0,
    "cable_angle_safety": 1,

Trained again for 0.2m arrival distance with gains as: 08-18
    "target": 10.0,
    "smooth": -0.07,
    "yaw": 0.01,
    "angular": -0.012,
    "crash": -20.0,
    "cable_angle_safety": 1,

Trained again for 0.2m arrival distance with gains as: 10-08
    "target": 13.0,
    "smooth": -0.06,
    "yaw": 0.01,
    "angular": -0.012,
    "crash": -20.0,
    "cable_angle_safety": 1,  

Trained again for 0.2m arrival distance with gains as:10-09
    "target": 13.0,
    "smooth": -0.06,
    "yaw": 0.01,
    "angular": -0.015,
    "crash": -20.0,
    "cable_angle_safety": 1,
