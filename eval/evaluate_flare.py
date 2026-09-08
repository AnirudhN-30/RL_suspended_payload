"""Deterministically evaluate a trained FLARE payload checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.flare_payload import mdp
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Flare-Waypoint-Payload"
DEFAULT_CHECKPOINT = Path("logs/rsl_rl/flare_payload/2026-09-04_15-00-03/model_799.pt")


def _trajectory_points(name: str, spacing: float, altitude: float) -> torch.Tensor:
  """Return local-frame waypoints, including points needed for a smooth loop."""
  if spacing <= 0.0:
    raise ValueError("--trajectory-spacing must be positive")
  if name == "random":
    return torch.empty(0, 3)
  if name == "hexagon":
    # For a regular hexagon, circumradius and edge length are equal.
    return torch.tensor(
      [
        (
          spacing * math.cos(index * math.pi / 3.0),
          spacing * math.sin(index * math.pi / 3.0),
          altitude,
        )
        for index in range(6)
      ],
      dtype=torch.float32,
    )
  if name == "straight-3":
    # Revisit the center on the return leg so every transition is `spacing` long.
    return torch.tensor(
      [
        (-spacing, 0.0, altitude),
        (0.0, 0.0, altitude),
        (spacing, 0.0, altitude),
        (0.0, 0.0, altitude),
      ],
      dtype=torch.float32,
    )
  raise ValueError(f"unknown trajectory: {name}")


def _configure_fixed_trajectory(waypoint_term, raw_env, args) -> list[list[float]]:
  points = _trajectory_points(
    args.trajectory, args.trajectory_spacing, args.trajectory_altitude
  ).to(device=raw_env.device)
  if not len(points):
    return []
  cursor = torch.zeros(raw_env.num_envs, dtype=torch.long, device=raw_env.device)

  def sample_fixed(env_ids: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    del anchor  # Fixed tracks already contain absolute local-frame positions.
    indices = cursor[env_ids] % len(points)
    values = points[indices] + raw_env.scene.env_origins[env_ids]
    cursor[env_ids] += 1
    return values

  waypoint_term._sample = sample_fixed
  return points.cpu().tolist()


def _to_numpy(value: torch.Tensor) -> np.ndarray:
  return value.detach().cpu().numpy()


def _write_plot(rows: list[dict[str, float | int | bool]], output: Path) -> None:
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    print("[WARN] matplotlib is unavailable; skipping diagnostic plot")
    return

  selected = [row for row in rows if row["env_id"] == 0]
  if not selected:
    return
  t = np.asarray([row["time_s"] for row in selected])
  pos = np.asarray([[row[f"quad_{axis}"] for axis in "xyz"] for row in selected])
  payload = np.asarray([[row[f"payload_{axis}"] for axis in "xyz"] for row in selected])
  waypoint = np.asarray(
    [[row[f"waypoint_{axis}"] for axis in "xyz"] for row in selected]
  )
  action = np.asarray(
    [[row[f"action_{index}"] for index in range(4)] for row in selected]
  )

  fig = plt.figure(figsize=(13, 9), constrained_layout=True)
  ax3d = fig.add_subplot(2, 2, 1, projection="3d")
  ax3d.plot(*pos.T, label="quadrotor")
  ax3d.plot(*payload.T, label="payload", alpha=0.75)
  ax3d.scatter(*waypoint.T, label="active waypoint", s=8, alpha=0.25)
  ax3d.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="Trajectory")
  ax3d.legend()

  ax = fig.add_subplot(2, 2, 2)
  ax.plot(t, [row["distance_m"] for row in selected])
  ax.axhline(0.5, color="tab:green", linestyle="--", label="arrival threshold")
  ax.set(xlabel="time [s]", ylabel="distance [m]", title="Waypoint distance")
  ax.legend()

  ax = fig.add_subplot(2, 2, 3)
  for index, label in enumerate(("throttle", "roll rate", "pitch rate", "yaw rate")):
    ax.plot(t, action[:, index], label=label)
  ax.set(xlabel="time [s]", ylabel="normalized action", ylim=(-1.05, 1.05))
  ax.legend(fontsize=8)

  ax = fig.add_subplot(2, 2, 4)
  ax.plot(t, [row["cable_angle_deg"] for row in selected], label="cable angle")
  ax.plot(
    t, [row["tendon_length_m"] * 100 for row in selected], label="tendon length [cm]"
  )
  ax.axhline(75.0, color="tab:red", linestyle="--", label="cable safety threshold")
  ax.set(xlabel="time [s]", ylabel="degrees / centimetres", title="Payload safety")
  ax.legend(fontsize=8)

  fig.savefig(output, dpi=160)
  plt.close(fig)


def evaluate(args: argparse.Namespace) -> Path:
  configure_torch_backends()
  checkpoint = args.checkpoint.resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(
      "CUDA is unavailable. Run with `uv run --extra cu128 python "
      "eval/evaluate_flare.py ...`."
    )

  env_cfg = load_env_cfg(TASK_ID)
  agent_cfg = load_rl_cfg(TASK_ID)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  env_cfg.episode_length_s = args.episode_length_s

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device
  )
  policy = runner.get_inference_policy(device=args.device)

  waypoint_term = raw_env.command_manager.get_term("waypoints")
  trajectory_points = _configure_fixed_trajectory(waypoint_term, raw_env, args)

  # Reset after runner construction so the recorded episodes all begin at the
  # requested seed. TensorDict matches the input type used by RSL-RL.
  obs_dict, _ = raw_env.reset(seed=args.seed)
  obs = TensorDict(obs_dict, batch_size=[args.num_envs])
  active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
  returns = torch.zeros(args.num_envs, device=args.device)
  steps = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
  reached_counts = torch.zeros_like(steps)
  min_distance = torch.full((args.num_envs,), float("inf"), device=args.device)
  max_cable_deg = torch.zeros(args.num_envs, device=args.device)
  max_tendon = torch.zeros(args.num_envs, device=args.device)
  end_reason = ["horizon"] * args.num_envs
  rows: list[dict[str, float | int | bool]] = []

  action_term = raw_env.action_manager.get_term("body_rate")
  previous_waypoint = waypoint_term.current.clone()

  with torch.inference_mode():
    for step in range(raw_env.max_episode_length):
      if not active.any():
        break
      action = policy(obs).clamp(-1.0, 1.0)
      quad = raw_env.scene["quadrotor"].data
      payload = raw_env.scene["payload"].data
      distance = torch.linalg.norm(waypoint_term.current - quad.root_link_pos_w, dim=1)
      cable_deg = torch.rad2deg(mdp.cable_body_angle(raw_env))
      tendon = raw_env.sim.data.ten_length[:, 0]
      waypoint_changed = torch.any(
        torch.abs(waypoint_term.current - previous_waypoint) > 1.0e-6, dim=1
      )
      reached_counts += waypoint_changed.long() * active.long()
      previous_waypoint[:] = waypoint_term.current

      arrays = {
        "quad": _to_numpy(quad.root_link_pos_w),
        "payload": _to_numpy(payload.root_link_pos_w),
        "waypoint": _to_numpy(waypoint_term.current),
        "next_waypoint": _to_numpy(waypoint_term.next),
        "velocity": _to_numpy(quad.root_link_lin_vel_w),
        "body_rate": _to_numpy(quad.root_link_ang_vel_b),
        "action": _to_numpy(action),
        "rotor": _to_numpy(action_term.rotor_thrusts),
        "distance": _to_numpy(distance),
        "cable": _to_numpy(cable_deg),
        "tendon": _to_numpy(tendon),
        "active": _to_numpy(active),
      }
      for env_id in np.flatnonzero(arrays["active"]):
        row: dict[str, float | int | bool] = {
          "env_id": int(env_id),
          "step": step,
          "time_s": step * raw_env.step_dt,
          "distance_m": float(arrays["distance"][env_id]),
          "cable_angle_deg": float(arrays["cable"][env_id]),
          "tendon_length_m": float(arrays["tendon"][env_id]),
          "waypoint_changed": bool(waypoint_changed[env_id].item()),
        }
        for name in (
          "quad",
          "payload",
          "waypoint",
          "next_waypoint",
          "velocity",
          "body_rate",
        ):
          for index, axis in enumerate("xyz"):
            row[f"{name}_{axis}"] = float(arrays[name][env_id, index])
        for name in ("action", "rotor"):
          for index in range(4):
            row[f"{name}_{index}"] = float(arrays[name][env_id, index])
        rows.append(row)

      obs, reward, _, _ = env.step(action)
      returns += reward * active
      steps += active.long()
      min_distance = torch.where(
        active, torch.minimum(min_distance, distance), min_distance
      )
      max_cable_deg = torch.where(
        active, torch.maximum(max_cable_deg, cable_deg), max_cable_deg
      )
      max_tendon = torch.where(active, torch.maximum(max_tendon, tendon), max_tendon)
      finished = active & raw_env.reset_buf
      for env_id in finished.nonzero(as_tuple=False).flatten().tolist():
        end_reason[env_id] = (
          "timeout" if raw_env.reset_time_outs[env_id].item() else "crash"
        )
      active &= ~finished

  episode_rows = []
  for env_id in range(args.num_envs):
    episode_rows.append(
      {
        "env_id": env_id,
        "return": float(returns[env_id].item()),
        "steps": int(steps[env_id].item()),
        "duration_s": float(steps[env_id].item() * raw_env.step_dt),
        "end_reason": end_reason[env_id],
        "waypoints_reached": int(reached_counts[env_id].item()),
        "minimum_waypoint_distance_m": float(min_distance[env_id].item()),
        "maximum_cable_angle_deg": float(max_cable_deg[env_id].item()),
        "maximum_tendon_length_m": float(max_tendon[env_id].item()),
      }
    )

  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  output = args.output or Path("eval/results") / f"{timestamp}_{checkpoint.stem}"
  output.mkdir(parents=True, exist_ok=False)
  with (output / "trajectories.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  reached = np.asarray([ep["waypoints_reached"] for ep in episode_rows])
  summary = {
    "checkpoint": str(checkpoint),
    "seed": args.seed,
    "num_episodes": args.num_envs,
    "deterministic_actions": True,
    "episode_length_s": args.episode_length_s,
    "trajectory": args.trajectory,
    "trajectory_spacing_m": args.trajectory_spacing,
    "trajectory_altitude_m": args.trajectory_altitude,
    "trajectory_points_local_m": trajectory_points,
    "success_rate_at_least_one_waypoint": float(np.mean(reached >= 1)),
    "mean_waypoints_reached": float(np.mean(reached)),
    "crash_rate": float(np.mean([ep["end_reason"] == "crash" for ep in episode_rows])),
    "timeout_rate": float(
      np.mean([ep["end_reason"] == "timeout" for ep in episode_rows])
    ),
    "mean_return": float(np.mean([ep["return"] for ep in episode_rows])),
    "mean_minimum_waypoint_distance_m": float(
      np.mean([ep["minimum_waypoint_distance_m"] for ep in episode_rows])
    ),
    "episodes": episode_rows,
  }
  (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  _write_plot(rows, output / "trajectory_env_000.png")
  env.close()
  print(
    json.dumps(
      {key: value for key, value in summary.items() if key != "episodes"}, indent=2
    )
  )
  print(f"Results: {output.resolve()}")
  return output


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--episode-length-s", type=float, default=15.0)
  parser.add_argument(
    "--trajectory",
    choices=("random", "hexagon", "straight-3"),
    default="random",
    help="Waypoint path; fixed paths repeat for the full episode.",
  )
  parser.add_argument(
    "--trajectory-spacing",
    type=float,
    default=1.2,
    help="Hexagon edge length or spacing between straight-line points [m].",
  )
  parser.add_argument(
    "--trajectory-altitude", type=float, default=1.0, help="Fixed path altitude [m]."
  )
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path)
  evaluate(parser.parse_args())


if __name__ == "__main__":
  main()
