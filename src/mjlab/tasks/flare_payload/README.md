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

## Drone-relative waypoint / L2 retraining (v1)

Following paper Section II-C, both initial targets are independently sampled
with XYZ offsets in `[-2, 2] × [-2, 2] × [0.5, 1.5] m` from the drone.
On arrival, waypoint 2 becomes waypoint 1 and a new waypoint 2 is sampled
relative to the drone's current position, not relative to waypoint 1.
Targets remain fixed in world space until promoted/replaced; they do not move
continuously with the drone. Both observation vectors remain drone-relative.
The arrival radius remains 0.5 m. Fixed evaluation tracks are unchanged.

The Z range is interpreted literally as a positive offset from the drone,
not an absolute altitude band. The paper's wording is used for this run;
this allows target altitude to increase across successful transitions.
Other simulator and reward differences mean this is not a full paper reproduction.

Smoothness uses the **unsquared** L2 norm from paper Eq. (8), with the
quad-tuned `acp_smooth` coefficient: `-0.01 * ||action_t - action_(t-1)||_2`
before MJLab's existing 0.01 s reward scaling. Default reward weights are:
target `5`, smoothness `-0.01`, yaw `0.01`, angular rate `-0.005`, crash `-20`,
and cable safety `1.0`. These are tuned coefficients, not the paper coefficients.
The explicit `Mjlab-Flare-Waypoint-Payload-Flare-Rewards` task still uses the
`flare` coefficients. PPO settings remain unchanged (learning rate `3e-4`,
entropy coefficient `0.002`); restoring reward weights does not switch PPO profiles.
The plant, observation layout, controller, drone-relative sampling, and actor
output tanh remain unchanged.

## Setup and checks

The repository is pinned to Python 3.10, MuJoCo 3.6, Warp 1.12, and the CUDA
12.8 PyTorch extra because the former nightly MuJoCo wheel in the cloned lockfile
no longer exists.

```bash
cd /home/anirudh/.openclaw/workspace/RL_suspended_payload
uv sync --python /usr/bin/python3.10 --extra cu128
uv run --extra cu128 python scripts/verify_flare_payload.py
uv run --extra cu128 python scripts/benchmark_flare_payload.py --num-envs 4096
```

The verification checks the 26/4 interface, finite observations and rewards,
tendon length, loaded hover motor thrust, and CUDA execution.

## Train

```bash
cd /home/anirudh/.openclaw/workspace/RL_suspended_payload
uv run --extra cu128 train Mjlab-Flare-Waypoint-Payload \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard --agent.resume False --agent.max-iterations 1000
```

Training uses RSL-RL PPO with a `26 -> 128 -> 128 -> 4` tanh actor and a matching
critic, learning rate `3e-4`, entropy coefficient `0.002`, and 100 rollout steps.
The command above starts from scratch: 1,024 environments × 100 steps × 1,000
iterations = 102.4 million samples. Outputs are isolated under
`logs/rsl_rl/flare_payload_drone_relative_l2_tanh_v2/`. No old checkpoint is resumed.
Nominal plant/sensor settings are retained; no new domain randomization is added.

## Visualize

```bash
uv run --extra cu128 play Mjlab-Flare-Waypoint-Payload --agent random
```

For a calm plant visualization and low-level-controller tests, use the viewer in
the sibling native reference repository instead; random policy actions are
aggressive by design.

## Actor output tanh and actuator mapping

The actor is `26 -> Linear(128) -> tanh -> Linear(128) -> tanh -> Linear(4)
-> tanh`. The critic retains a linear scalar output. The actor class is
`mjlab.tasks.flare_payload.policy:FlareTanhActor`.

Deterministic inference uses the bounded mean. Training samples a Gaussian
around that mean, with ordinary Gaussian PPO log probabilities (not a
squashed Gaussian). Samples can exceed [-1, 1]; the existing controller clips
commands before physical scaling. The smoothness reward still uses the
existing action-manager actions, not newly substituted motor commands.

For clipped normalized actions u:

- Total thrust: `(u[0] + 1) / 2 * 3.5 * 1.1628 * 9.81` N, or 0–39.924738 N.
- Roll rate: `15 * u[1]` rad/s.
- Pitch rate: `15 * u[2]` rad/s.
- Yaw rate: `5 * u[3]` rad/s.

The body-rate PID generates torques, and the existing inverse effectiveness
matrix maps total thrust plus torques to four rotor thrusts. Each rotor is
clamped to [0, 20.5] N. Commands are held over five 0.002 s physics steps.
Level, zero-rate loaded hover requires approximately u[0] = -0.42857,
not zero (zero requests half maximum collective thrust).

Start fresh for this architecture. An actor checkpoint marker prevents strict
loading of older linear-output checkpoints, including in the evaluator/player.
Their old default checkpoint paths are not tanh policies; supply a new v2
checkpoint with `--checkpoint` after training. Old policies remain usable with
the previous code revision. No training is automatically started by this change.
