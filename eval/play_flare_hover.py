"""Interactive FLARE payload-hover disturbance test with mouse-applied forces."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.flare_payload import mdp
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import VerbosityLevel

TASK_ID = "Mjlab-Flare-Payload-Targeting"
LOG_ROOT = Path("logs/rsl_rl/flare_payload_targeting_scenario2_26d")


def _latest_checkpoint() -> Path:
  checkpoints = list(LOG_ROOT.glob("*/model_*.pt"))
  if not checkpoints:
    raise FileNotFoundError(
      f"No payload-targeting checkpoints found below {LOG_ROOT.resolve()}"
    )
  return max(checkpoints, key=lambda path: path.stat().st_mtime)


def _hold_target(waypoint, raw_env, target_local: torch.Tensor) -> None:
  """Replace random waypoint sampling and promotion with one fixed target."""

  def sample_fixed(env_ids: torch.Tensor) -> torch.Tensor:
    return target_local.expand(len(env_ids), -1) + raw_env.scene.env_origins[env_ids]

  def keep_current_target() -> None:
    waypoint.observation_current[:] = waypoint.current
    waypoint.observation_next[:] = waypoint.next

  waypoint._sample = sample_fixed
  waypoint._update_command = keep_current_target


class HoverDiagnosticsPolicy:
  """Wrap inference with low-rate hover diagnostics for the terminal."""

  def __init__(self, policy, raw_env, waypoint, report_every: int = 100):
    self.policy = policy
    self.raw_env = raw_env
    self.waypoint = waypoint
    self.report_every = report_every
    self.steps = 0
    self.max_payload_error = 0.0
    self.max_cable_angle = 0.0

  def __call__(self, observations):
    action = self.policy(observations)
    payload_position = self.raw_env.scene["payload"].data.root_link_pos_w
    error = torch.linalg.vector_norm(
      self.waypoint.current - payload_position, dim=1
    )[0]
    cable_angle = torch.rad2deg(mdp.cable_body_angle(self.raw_env))[0]
    self.max_payload_error = max(self.max_payload_error, float(error.item()))
    self.max_cable_angle = max(self.max_cable_angle, float(cable_angle.item()))
    if self.steps % self.report_every == 0:
      print(
        f"[HOVER] t={self.steps * self.raw_env.step_dt:7.2f}s "
        f"payload_error={error.item():.3f}m "
        f"cable_angle={cable_angle.item():.1f}deg "
        f"max_error={self.max_payload_error:.3f}m "
        f"max_angle={self.max_cable_angle:.1f}deg"
      )
    self.steps += 1
    return action


def play(args: argparse.Namespace) -> None:
  configure_torch_backends()
  checkpoint = (args.checkpoint or _latest_checkpoint()).resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; pass --device cpu or install CUDA support")

  env_cfg = load_env_cfg(TASK_ID, play=True)
  agent_cfg = load_rl_cfg(TASK_ID)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.seed

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  waypoint = raw_env.command_manager.get_term("waypoints")
  target = torch.tensor(
    (args.target_x, args.target_y, args.target_z),
    device=raw_env.device,
    dtype=torch.float32,
  )
  _hold_target(waypoint, raw_env, target)
  raw_env.reset(seed=args.seed)

  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device
  )
  inference_policy = runner.get_inference_policy(device=args.device)
  policy = HoverDiagnosticsPolicy(inference_policy, raw_env, waypoint)

  print(f"[INFO] Checkpoint: {checkpoint}")
  print(f"[INFO] Fixed payload target: {waypoint.current[0].tolist()}")
  if args.headless_steps > 0:
    for _ in range(args.headless_steps):
      observations = env.get_observations()
      env.step(policy(observations))
    env.close()
    print("[INFO] Headless hover smoke complete")
    return

  print("[INFO] Mouse disturbance controls:")
  print("  1. Double-click the quadrotor or payload to select it.")
  print("  2. Hold Ctrl + left-drag to apply a translational force.")
  print("  3. Hold Ctrl + right-drag to apply a rotational torque.")
  print("  4. Release the mouse and observe policy recovery.")
  print("  Space: pause/resume | Enter: reset | P: reward plots")

  try:
    NativeMujocoViewer(
      env,
      policy,
      frame_rate=args.frame_rate,
      enable_perturbations=True,
      verbosity=VerbosityLevel.INFO,
    ).run()
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--target-x", type=float, default=0.0)
  parser.add_argument("--target-y", type=float, default=0.0)
  parser.add_argument(
    "--target-z",
    type=float,
    default=0.29,
    help="Fixed payload hover altitude [m]; 0.29 matches the default reset pose.",
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--frame-rate", type=float, default=60.0)
  parser.add_argument(
    "--headless-steps",
    type=int,
    default=0,
    help="Run a finite no-window smoke test instead of opening the native viewer.",
  )
  play(parser.parse_args())


if __name__ == "__main__":
  main()
