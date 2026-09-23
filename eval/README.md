# FLARE checkpoint evaluation

This folder contains deterministic, headless evaluation for the FLARE payload
policy. It records fixed-seed rollouts rather than relying on training-time
episode averages.

Run the current checkpoint from the repository root:

```bash
uv run --extra cu128 python eval/evaluate_flare.py \
  --checkpoint logs/rsl_rl/flare_payload/2026-09-04_15-00-03/model_799.pt
```

Results are written to a timestamped directory under `eval/results/`:

- `summary.json`: aggregate and per-episode metrics
- `trajectories.csv`: step-level state, waypoint, action, and outcome data
- `trajectory_env_000.png`: 3D trajectory and diagnostic plots for one episode

The inference policy returned by RSL-RL is used in evaluation mode, so actions
are the actor means (no exploration sampling). Given the same checkpoint,
seed, environment count, and software versions, the rollouts are reproducible.

## Fixed trajectories

Test a level regular hexagon whose edges are 1.2 m long:

```bash
uv run --extra cu128 python eval/evaluate_flare.py \
  --checkpoint logs/rsl_rl/flare_payload_acp_smooth_v2/2026-09-05_18-55-56/model_1000.pt \
  --trajectory hexagon --trajectory-spacing 1.2 --trajectory-altitude 1.0
```

Test three collinear points spaced 1.2 m apart. The return leg revisits the
center, so every commanded transition remains exactly 1.2 m:

```bash
uv run --extra cu128 python eval/evaluate_flare.py \
  --checkpoint logs/rsl_rl/flare_payload_acp_smooth_v2/2026-09-05_18-55-56/model_1000.pt \
  --trajectory straight-3 --trajectory-spacing 1.2 --trajectory-altitude 1.0
```

## Interactive visualization

Open the native MuJoCo window and watch the trained policy follow either path:

```bash
uv run --extra cu128 python eval/play_flare_trajectory.py \
  --trajectory hexagon --trajectory-spacing 1.2 --trajectory-altitude 1.0 \
  --viewer native

uv run --extra cu128 python eval/play_flare_trajectory.py \
  --trajectory straight-3 --trajectory-spacing 1.2 --trajectory-altitude 1.0 \
  --viewer native
```

The green sphere is the active waypoint and the smaller blue sphere is the
next waypoint. Close the MuJoCo window or press Ctrl+C to stop. On a machine
without a graphical display, use `--viewer viser` and open the URL printed in
the terminal.
