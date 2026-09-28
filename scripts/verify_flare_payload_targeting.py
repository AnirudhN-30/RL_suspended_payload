"""Focused CUDA verification for FLARE payload-waypoint navigation."""

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.flare_payload import mdp
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def main() -> None:
  cfg = load_env_cfg("Mjlab-Flare-Payload-Targeting")
  cfg.scene.num_envs = 16
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  observations, _ = env.reset(seed=7)
  baseline_cfg = load_env_cfg("Mjlab-Flare-Waypoint-Payload")
  agent_cfg = load_rl_cfg("Mjlab-Flare-Payload-Targeting")
  baseline_agent_cfg = load_rl_cfg("Mjlab-Flare-Waypoint-Payload")

  assert observations["actor"].shape == (16, 26)
  assert observations["critic"].shape == (16, 26)
  assert env.action_manager.total_action_dim == 4
  assert list(cfg.observations["actor"].terms) == [
    "quad_target",
    "payload_target",
    "linear_velocity",
    "rotation_matrix",
    "previous_action",
    "cable_state",
  ]
  assert (
    cfg.observations["actor"].terms["quad_target"].func
    is mdp.scenario2_quad_target_rel
  )
  assert (
    cfg.observations["actor"].terms["payload_target"].func
    is mdp.payload_target_rel
  )
  assert (
    cfg.observations["actor"].terms["linear_velocity"].func
    is mdp.scenario2_linear_velocity_world
  )
  assert (
    cfg.observations["actor"].terms["cable_state"].func
    is mdp.Scenario2CableAngleObservation
  )
  assert agent_cfg.actor == baseline_agent_cfg.actor
  assert agent_cfg.critic == baseline_agent_cfg.critic
  assert agent_cfg.algorithm == baseline_agent_cfg.algorithm
  assert agent_cfg.num_steps_per_env == baseline_agent_cfg.num_steps_per_env
  assert agent_cfg.max_iterations == baseline_agent_cfg.max_iterations

  for name in ("smooth", "yaw", "angular", "crash", "cable_angle_safety"):
    assert cfg.rewards[name].func is baseline_cfg.rewards[name].func
    assert cfg.rewards[name].weight == baseline_cfg.rewards[name].weight
  assert cfg.rewards["target"].weight == baseline_cfg.rewards["target"].weight
  assert cfg.rewards["target"].func is mdp.PayloadTargetProgressReward

  waypoint = env.command_manager.get_term("waypoints")
  quad_position = env.scene["quadrotor"].data.root_link_pos_w
  payload_position = env.scene["payload"].data.root_link_pos_w
  sampled_position = waypoint.current - env.scene.env_origins
  sampled_altitude = waypoint.current[:, 2] - env.scene.env_origins[:, 2]
  assert torch.all(sampled_position[:, 0].abs() <= 1.5 + 1.0e-5)
  assert torch.all(sampled_position[:, 1].abs() <= 1.5 + 1.0e-5)
  assert torch.all(sampled_altitude >= 0.5 - 1.0e-5)
  assert torch.all(sampled_altitude <= 1.5 + 1.0e-5)

  expected_quad_error = ((waypoint.current - quad_position) * torch.tensor(
    (1.0 / 5.0, 1.0 / 5.0, 1.0), device=env.device
  )).clamp(-1.0, 1.0)
  expected_payload_error = ((waypoint.current - payload_position) * torch.tensor(
    (1.0 / 5.0, 1.0 / 5.0, 1.0), device=env.device
  )).clamp(-1.0, 1.0)
  expected_velocity = (env.scene["quadrotor"].data.root_link_lin_vel_w * torch.tensor(
    (0.1, 0.1, 1.0 / 3.0), device=env.device
  )).clamp(-1.0, 1.0)
  assert torch.allclose(observations["actor"][:, 0:3], expected_quad_error)
  assert torch.allclose(observations["actor"][:, 3:6], expected_payload_error)
  assert torch.allclose(observations["actor"][:, 6:9], expected_velocity)
  assert not torch.allclose(expected_quad_error, expected_payload_error)

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
  assert torch.isfinite(mdp.action_smoothness(env)).all()

  # A reset after moving the payload must initialize progress from the new
  # qpos, not from stale derived kinematics belonging to the previous episode.
  payload_before_reset = env.scene["payload"].data.root_link_pos_w.clone()
  env.reset()
  payload_after_reset = env.scene["payload"].data.root_link_pos_w
  reward_term = env.reward_manager.get_term_cfg("target").func
  assert not torch.allclose(payload_before_reset, payload_after_reset)
  assert torch.allclose(reward_term.previous_position, payload_after_reset)

  print(f"observations: {tuple(observations['actor'].shape)}")
  print(f"actions: {env.action_manager.total_action_dim}")
  print(f"tracking entity: {waypoint.cfg.entity_name}")
  print("observation: FLARE Scenario II plus cable-angle rates (26D)")
  print("network/PPO/reward calculation and weights: MJLab baseline")
  print(f"arrival radius: {waypoint.cfg.arrival_threshold:.3f} m")
  print(f"terminated/timeouts: {terminated.sum().item()}/{truncated.sum().item()}")
  print("MJLab FLARE payload-targeting CUDA verification: PASS")
  env.close()


if __name__ == "__main__":
  main()
