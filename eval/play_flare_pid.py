"""Waypoint PD + attitude P feeding the unchanged FLARE body-rate PID (no RL)."""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import torch
from evaluate_flare import _configure_fixed_trajectory

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.flare_payload.mdp import cable_body_angle
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


class WaypointController:
  def __init__(self, env, args, writer):
    self.env, self.args, self.writer = env, args, writer
    self.steps = 0
    self.errors = []
    self.reached = 0

  def __call__(self, obs):
    del obs
    env, args = self.env, self.args
    quad = env.scene["quadrotor"]
    command = env.command_manager.get_term("waypoints")
    motor = env.action_manager.get_term("body_rate")
    pos = quad.data.root_link_pos_w
    vel = quad.data.root_link_lin_vel_w
    rotation = matrix_from_quat(quad.data.root_link_quat_w)
    error = command.current - pos
    # Velocity-limited position P, then velocity P (a bounded position PD).
    desired_vel = args.position_gain * error
    desired_vel *= (
      args.max_speed / desired_vel.norm(dim=1, keepdim=True).clamp_min(1e-6)
    ).clamp(max=1)
    accel = args.velocity_gain * (desired_vel - vel)
    accel[:, 2] = accel[:, 2].clamp(-3, 3)
    # Bound tilt by limiting horizontal acceleration relative to vertical force.
    vertical = accel[:, 2] + motor.cfg.gravity
    horizontal_max = vertical * math.tan(math.radians(args.max_tilt))
    accel[:, :2] *= (
      horizontal_max[:, None] / accel[:, :2].norm(dim=1, keepdim=True).clamp_min(1e-6)
    ).clamp(max=1)
    force = accel.clone()
    force[:, 2] += motor.cfg.gravity
    force *= motor.cfg.loaded_mass
    z_des = torch.nn.functional.normalize(force, dim=1)
    heading = torch.zeros_like(z_des)
    heading[:, 0] = 1  # Fixed world yaw = 0.
    y_des = torch.nn.functional.normalize(torch.cross(z_des, heading, dim=1), dim=1)
    x_des = torch.cross(y_des, z_des, dim=1)
    desired_rotation = torch.stack((x_des, y_des, z_des), dim=2)
    # Geometric attitude error in world frame, transformed into body coordinates.
    attitude_error = 0.5 * torch.cross(rotation, desired_rotation, dim=1).sum(dim=2)
    body_error = torch.bmm(
      rotation.transpose(1, 2), attitude_error.unsqueeze(-1)
    ).squeeze(-1)
    rates = (args.attitude_gain * body_error).clamp(-args.max_rate, args.max_rate)
    collective = (force * rotation[:, :, 2]).sum(dim=1).clamp_min(0)
    max_collective = (
      motor.cfg.loaded_mass * motor.cfg.gravity * motor.cfg.thrust_to_weight_max
    )
    action = torch.cat(
      ((2 * collective / max_collective - 1)[:, None], rates / motor.max_rates), dim=1
    ).clamp(-1, 1)
    measured_rates = quad.data.root_link_ang_vel_b
    distance = error.norm(dim=1).item()
    self.errors.append(distance)
    self.reached += int(command.metrics["reached"][0].item())
    self.writer.writerow(
      [
        self.steps * env.step_dt,
        distance,
        *pos[0].tolist(),
        *command.current[0].tolist(),
        *rates[0].tolist(),
        *measured_rates[0].tolist(),
        *motor.rotor_thrusts[0].tolist(),
        cable_body_angle(env)[0].item(),
      ]
    )
    self.steps += 1
    return action


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--trajectory", choices=["hexagon", "straight-3"], default="hexagon"
  )
  parser.add_argument("--trajectory-spacing", type=float, default=1.2)
  parser.add_argument("--trajectory-altitude", type=float, default=1.0)
  parser.add_argument("--arrival-radius", type=float, default=0.15)
  parser.add_argument("--max-speed", type=float, default=0.3)
  parser.add_argument("--max-tilt", type=float, default=20)
  parser.add_argument("--max-rate", type=float, default=2.0)
  parser.add_argument("--position-gain", type=float, default=0.8)
  parser.add_argument("--velocity-gain", type=float, default=1.5)
  parser.add_argument("--attitude-gain", type=float, default=1.5)
  parser.add_argument("--viewer", choices=["native", "viser", "none"], default="native")
  parser.add_argument("--duration", type=float, default=30)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  for name in [
    "arrival_radius",
    "max_speed",
    "max_rate",
    "position_gain",
    "velocity_gain",
    "attitude_gain",
    "duration",
    "trajectory_spacing",
  ]:
    if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
      parser.error(f"{name} must be finite and positive")
  if not 0 < args.max_tilt < 60:
    parser.error("max-tilt must be between 0 and 60 degrees")
  output = args.output or Path("eval/results") / datetime.now().strftime(
    "pid_%Y-%m-%d_%H-%M-%S"
  )
  output.mkdir(parents=True, exist_ok=False)
  cfg = load_env_cfg("Mjlab-Flare-Waypoint-Payload", play=True)
  cfg.commands["waypoints"].arrival_threshold = args.arrival_radius
  env = ManagerBasedRlEnv(cfg, device=args.device)
  _configure_fixed_trajectory(env.command_manager.get_term("waypoints"), env, args)
  env.reset(seed=42)
  crashes = 0
  try:
    with (output / "tracking.csv").open("w", newline="") as file:
      writer = csv.writer(file)
      writer.writerow(
        [
          "time_s",
          "distance_m",
          "x",
          "y",
          "z",
          "target_x",
          "target_y",
          "target_z",
          "roll_rate_cmd",
          "pitch_rate_cmd",
          "yaw_rate_cmd",
          "roll_rate_actual",
          "pitch_rate_actual",
          "yaw_rate_actual",
          "motor1_previous_N",
          "motor2_previous_N",
          "motor3_previous_N",
          "motor4_previous_N",
          "cable_angle_rad",
        ]
      )
      policy = WaypointController(env, args, writer)
      with torch.inference_mode():
        if args.viewer == "none":
          obs = None
          for _ in range(round(args.duration / env.step_dt)):
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            if not torch.isfinite(action).all() or not torch.isfinite(reward).all():
              raise RuntimeError("Nonfinite control or reward")
            crashes += int(terminated.sum().item())
        else:
          wrapped = RslRlVecEnvWrapper(env)
          viewer = NativeMujocoViewer if args.viewer == "native" else ViserPlayViewer
          viewer(wrapped, policy).run()
  except KeyboardInterrupt:
    print("Stopped; saving collected diagnostics.")
  finally:
    env.close()
  summary = {
    "steps": policy.steps,
    "waypoints_reached": policy.reached,
    "crashes": crashes if args.viewer == "none" else None,
    "mean_distance_m": sum(policy.errors) / max(1, len(policy.errors)),
    "settings": {
      k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    },
  }
  import matplotlib
  import numpy as np

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  data = np.genfromtxt(output / "tracking.csv", delimiter=",", names=True)
  if policy.steps:
    data = np.atleast_1d(data)
    summary["rate_tracking_rmse_rad_s"] = {
      axis: float(
        np.sqrt(np.mean((data[f"{axis}_rate_cmd"] - data[f"{axis}_rate_actual"]) ** 2))
      )
      for axis in ("roll", "pitch", "yaw")
    }
    summary["max_cable_angle_deg"] = float(np.rad2deg(data["cable_angle_rad"].max()))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(data["x"], data["y"], label="quad path")
    axes[0].scatter(data["target_x"], data["target_y"], s=5, label="targets")
    axes[0].set(xlabel="x [m]", ylabel="y [m]", aspect="equal")
    for axis in ("roll", "pitch", "yaw"):
      axes[1].plot(data["time_s"], data[f"{axis}_rate_cmd"], label=f"{axis} command")
      axes[1].plot(
        data["time_s"],
        data[f"{axis}_rate_actual"],
        linestyle="--",
        label=f"{axis} actual",
      )
    axes[1].set(xlabel="time [s]", ylabel="body rate [rad/s]")
    axes[2].plot(data["time_s"], data["distance_m"], label="waypoint distance")
    axes[2].axhline(args.arrival_radius, linestyle="--", label="arrival radius")
    axes[2].set(xlabel="time [s]", ylabel="distance [m]")
    for ax in axes:
      ax.legend()
    fig.savefig(output / "tracking.png", dpi=150)
    plt.close(fig)
  (output / "summary.json").write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary, indent=2))
  print(f"Results: {output}")


if __name__ == "__main__":
  main()
