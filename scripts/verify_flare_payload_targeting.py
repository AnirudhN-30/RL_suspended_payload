"""Focused CUDA verification for FLARE payload-waypoint navigation."""

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.flare_payload import mdp
from mjlab.tasks.registry import load_env_cfg


def main() -> None:
  cfg = load_env_cfg("Mjlab-Flare-Payload-Targeting")
  cfg.scene.num_envs = 16
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  observations, _ = env.reset(seed=7)

  assert observations["actor"].shape == (16, 26)
  assert observations["critic"].shape == (16, 26)
  assert env.action_manager.total_action_dim == 4

  waypoint = env.command_manager.get_term("waypoints")
  quad_position = env.scene["quadrotor"].data.root_link_pos_w
  payload_position = env.scene["payload"].data.root_link_pos_w
  sampled_offset = waypoint.current - quad_position
  sampled_altitude = waypoint.current[:, 2] - env.scene.env_origins[:, 2]
  assert torch.all(sampled_offset[:, 0].abs() <= 2.0 + 1.0e-5)
  assert torch.all(sampled_offset[:, 1].abs() <= 2.0 + 1.0e-5)
  assert torch.all(sampled_altitude >= 0.5 - 1.0e-5)
  assert torch.all(sampled_altitude <= 1.5 + 1.0e-5)

  # Arrival must follow the payload rather than the quadrotor.
  original_target = waypoint.current.clone()
  waypoint.current[:] = payload_position
  waypoint._update_metrics()
  assert torch.all(waypoint.metrics["reached"] == 1.0)
  waypoint.current[:] = quad_position
  waypoint._update_metrics()
  payload_quad_distance = torch.linalg.vector_norm(payload_position - quad_position, dim=1)
  expected = (payload_quad_distance < waypoint.cfg.arrival_threshold).float()
  assert torch.equal(waypoint.metrics["reached"], expected)
  waypoint.current[:] = original_target

  action = torch.zeros((env.num_envs, 4), device=env.device)
  action[:, 0] = 2.0 / 3.5 - 1.0
  for _ in range(10):
    observations, rewards, terminated, truncated, _ = env.step(action)

  assert torch.isfinite(observations["actor"]).all()
  assert torch.isfinite(rewards).all()
  assert torch.isfinite(mdp.action_smoothness_l2(env)).all()

  print(f"observations: {tuple(observations['actor'].shape)}")
  print(f"actions: {env.action_manager.total_action_dim}")
  print(f"tracking entity: {waypoint.cfg.entity_name}")
  print("target sampling: quadrotor-relative XY, environment-absolute Z")
  print(f"arrival radius: {waypoint.cfg.arrival_threshold:.3f} m")
  print(f"terminated/timeouts: {terminated.sum().item()}/{truncated.sum().item()}")
  print("MJLab FLARE payload-targeting CUDA verification: PASS")
  env.close()


if __name__ == "__main__":
  main()
