# RL_suspended_payload

Reinforcement learning for a quadrotor carrying a suspended payload, built on
MJLab (MuJoCo Warp) and RSL-RL PPO. MJLab history and license are preserved;
the Python package remains `mjlab` for compatibility.

## Included implementation

- FLARE payload plant, body-rate controller, rotor mixer, and waypoint task.
- 26-value observations and four normalized policy actions.
- Training reward profiles, verification, and GPU benchmarking.
- Headless evaluation and interactive hexagon/three-point-line visualization.

## Setup and training

```bash
cd RL_suspended_payload
uv sync --python /usr/bin/python3.10 --extra cu128
uv run --extra cu128 python scripts/verify_flare_payload.py
uv run --extra cu128 train Mjlab-Flare-Waypoint-Payload --env.scene.num-envs 4096 --agent.logger tensorboard
```

## Visualization

```bash
uv run --extra cu128 python eval/play_flare_trajectory.py --checkpoint logs/rsl_rl/flare_payload_acp_smooth_v2/2026-09-05_18-55-56/model_1000.pt --trajectory hexagon --trajectory-spacing 1.2 --trajectory-altitude 1.0 --viewer native
```

Use `--trajectory straight-3` for the line or `--viewer viser` for a web viewer.
The existing policy uses a 0.5 m arrival radius. Repository migration does not
change training, controller, or waypoint behavior.

Checkpoints, logs, evaluation results, and virtual environments are excluded
from Git. On the original workstation, ignored symlinks expose existing
`mjlab/logs` and `mjlab/eval/results` without duplicating artifacts. Fresh clones
must install dependencies and supply a checkpoint with `--checkpoint`.

## Documentation and attribution

- [FLARE task and training](src/mjlab/tasks/flare_payload/README.md)
- [Evaluation and visualization](eval/README.md)
- [Original MJLab README](README.mjlab.md)
- [Upstream source](https://github.com/lfrecalde1/mjlab)
- [License](LICENSE)
