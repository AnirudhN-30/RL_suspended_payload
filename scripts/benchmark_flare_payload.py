"""Benchmark GPU stepping throughput for the FLARE payload task."""

from __future__ import annotations

import argparse
import time

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--warmup-steps", type=int, default=20)
  parser.add_argument("--steps", type=int, default=200)
  args = parser.parse_args()

  cfg = load_env_cfg("Mjlab-Flare-Waypoint-Payload")
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  env.reset(seed=7)
  actions = torch.empty((env.num_envs, 4), device=env.device).uniform_(-1.0, 1.0)

  for _ in range(args.warmup_steps):
    env.step(actions)
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(args.steps):
    env.step(actions)
  torch.cuda.synchronize()
  elapsed = time.perf_counter() - start

  transitions = args.num_envs * args.steps
  print(f"environments: {args.num_envs}")
  print(f"policy steps: {args.steps}")
  print(f"elapsed: {elapsed:.3f} s")
  print(f"throughput: {transitions / elapsed:,.0f} transitions/s")
  env.close()


if __name__ == "__main__":
  main()
