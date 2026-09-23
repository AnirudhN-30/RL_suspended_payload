"""Interactively visualize a trained FLARE policy on a fixed trajectory."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch
from evaluate_flare import PAYLOAD_TARGET_TASK_ID, TASK_ID, _configure_fixed_trajectory

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

DEFAULT_CHECKPOINT = Path(
  "logs/rsl_rl/flare_payload_acp_smooth_v2/2026-09-05_18-55-56/model_1000.pt"
)


def play(args: argparse.Namespace) -> None:
  configure_torch_backends()
  checkpoint = args.checkpoint.resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; pass --device cpu or install CUDA support")

  env_cfg = load_env_cfg(args.task_id, play=True)
  agent_cfg = load_rl_cfg(args.task_id)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  waypoint_term = raw_env.command_manager.get_term("waypoints")
  points = _configure_fixed_trajectory(waypoint_term, raw_env, args)
  raw_env.reset(seed=args.seed)

  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device
  )
  policy = runner.get_inference_policy(device=args.device)

  viewer_name = args.viewer
  if viewer_name == "auto":
    viewer_name = (
      "native"
      if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
      else "viser"
    )
  print(
    f"[INFO] Playing {args.trajectory} with {len(points)} command points "
    f"at z={args.trajectory_altitude:.2f} m using the {viewer_name} viewer"
  )
  try:
    if viewer_name == "native":
      NativeMujocoViewer(env, policy).run()
    else:
      ViserPlayViewer(env, policy).run()
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--task-id",
    choices=(TASK_ID, PAYLOAD_TARGET_TASK_ID),
    default=TASK_ID,
  )
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument(
    "--trajectory", choices=("hexagon", "straight-3"), default="hexagon"
  )
  parser.add_argument("--trajectory-spacing", type=float, default=1.2)
  parser.add_argument("--trajectory-altitude", type=float, default=1.0)
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
  play(parser.parse_args())


if __name__ == "__main__":
  main()
