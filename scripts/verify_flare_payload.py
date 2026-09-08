"""Small GPU smoke check for the FLARE payload task."""

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main() -> None:
  cfg = load_env_cfg("Mjlab-Flare-Waypoint-Payload")
  cfg.scene.num_envs = 16
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  observations, _ = env.reset(seed=7)
  assert observations["actor"].shape == (16, 26)
  assert observations["critic"].shape == (16, 26)
  assert env.action_manager.total_action_dim == 4

  action = torch.zeros((env.num_envs, 4), device=env.device)
  action[:, 0] = 2.0 / 3.5 - 1.0
  for _ in range(10):
    observations, rewards, terminated, truncated, _ = env.step(action)
  assert torch.isfinite(observations["actor"]).all()
  assert torch.isfinite(rewards).all()
  tendon = env.sim.data.ten_length[:, 0]
  action_term = env.action_manager.get_term("body_rate")

  print(f"observations: {tuple(observations['actor'].shape)}")
  print(f"actions: {env.action_manager.total_action_dim}")
  print(f"terminated/timeouts: {terminated.sum().item()}/{truncated.sum().item()}")
  print(f"tendon min/max: {tendon.min().item():.6f}/{tendon.max().item():.6f} m")
  print(f"motor thrusts env 0: {action_term.rotor_thrusts[0].tolist()}")
  print("MJLab FLARE GPU smoke: PASS")
  env.close()


if __name__ == "__main__":
  main()
