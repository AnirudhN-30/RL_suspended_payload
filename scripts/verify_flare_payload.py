"""Small GPU smoke check for the FLARE payload task."""

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.flare_payload import mdp
from mjlab.tasks.registry import load_env_cfg


def main() -> None:
  cfg = load_env_cfg("Mjlab-Flare-Waypoint-Payload")
  cfg.scene.num_envs = 16
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  observations, _ = env.reset(seed=7)
  assert observations["actor"].shape == (16, 26)
  assert observations["critic"].shape == (16, 26)
  assert env.action_manager.total_action_dim == 4

  command = env.command_manager.get_term("waypoints")
  ids = torch.arange(env.num_envs, device=env.device)
  anchor = env.scene.env_origins.clone()
  anchor += 20.0  # Distinguish drone-relative sampling from world-origin sampling.
  sampled = command._sample(ids, anchor)
  assert ((sampled[:, :2] - anchor[:, :2]).abs() <= 2.0).all()
  altitude = sampled[:, 2] - anchor[:, 2]
  assert ((altitude >= 0.5) & (altitude <= 1.5)).all()
  command.current[:] = anchor
  command.next[:] = anchor + torch.tensor([1.0, -1.0, 0.0], device=env.device)
  old_current, old_next = command.current.clone(), command.next.clone()
  command.metrics["reached"][:] = 0
  command.metrics["reached"][::2] = 1
  command._update_command()
  assert torch.equal(command.current[::2], old_next[::2])
  assert torch.equal(command.current[1::2], old_current[1::2])
  assert torch.equal(command.next[1::2], old_next[1::2])
  delta = command.next[::2] - env.scene["quadrotor"].data.root_link_pos_w[::2]
  assert (delta[:, :2].abs() <= 2).all()
  assert ((delta[:, 2] >= 0.5) & (delta[:, 2] <= 1.5)).all()
  # Distinguish L2 from squared L2 using a 3-4-5 delta.
  env.action_manager.prev_action.zero_()
  env.action_manager.action[:] = torch.tensor([0.3, 0.4, 0, 0], device=env.device)
  assert torch.allclose(
    mdp.action_smoothness(env), torch.full((16,), 0.5, device=env.device)
  )
  assert cfg.rewards["smooth"].weight == -0.0001
  observations, _ = env.reset(seed=7)
  for target in (command.current, command.next):
    delta = target - env.scene["quadrotor"].data.root_link_pos_w
    assert (delta[:, :2].abs() <= 2.0).all()
    assert ((delta[:, 2] >= 0.5) & (delta[:, 2] <= 1.5)).all()
  print("Drone-relative waypoint reset/promotion/subset and unsquared L2 reward: PASS")

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
