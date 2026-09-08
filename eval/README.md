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

## Classical-controller waypoint test (no RL checkpoint)

```bash
uv run --extra cu128 python eval/play_flare_pid.py --trajectory hexagon --viewer native
uv run --extra cu128 python eval/play_flare_pid.py --trajectory straight-3 --viewer native
```

Use `--viewer viser` for the existing browser viewer, or `--viewer none
--duration 30` for a finite headless test. Interactive playback continues until
the viewer is closed; duration applies only to headless mode.

This standalone test adds a velocity-limited position PD and geometric attitude
P controller upstream of the **unchanged** FLARE rate PID and rotor mixer.
Defaults: 1.2 m spacing, 1.0 m altitude, 0.15 m arrival radius, 0.3 m/s speed
limit, 20 degree tilt limit, and 2 rad/s rate-command limit. These test-specific
limits do not change RL configuration. There is no payload-swing compensation.

Outputs: `tracking.csv`, `tracking.png`, and `summary.json` under `eval/results/pid_*`.
The CSV records pre-step positions, targets, desired and actual body rates,
previous-step rotor thrusts, and cable angle. Interactive crash count is not
collected (JSON null); headless mode counts crash terminations. Waypoint counts
are arrivals, not completed laps. Automatic resets can start another episode.
This tests the full cascade; failures are not uniquely attributable to rate PID.
