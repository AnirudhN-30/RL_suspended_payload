"""Batched FLARE Scenario-I MDP terms for MuJoCo Warp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, matrix_from_quat

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer

QUAD_CFG = SceneEntityCfg("quadrotor")
PAYLOAD_CFG = SceneEntityCfg("payload")
MOTOR_SITES = (
  "motor_1_rear_right",
  "motor_2_front_right",
  "motor_3_rear_left",
  "motor_4_front_left",
)


def _quad(env, cfg: SceneEntityCfg = QUAD_CFG) -> Entity:
  return env.scene[cfg.name]


def _payload(env, cfg: SceneEntityCfg = PAYLOAD_CFG) -> Entity:
  return env.scene[cfg.name]


def _waypoints(env) -> "FlareWaypointCommand":
  return env.command_manager.get_term("waypoints")


@dataclass(kw_only=True)
class FlareRotorRateActionCfg(BaseActionCfg):
  """FLARE normalized throttle/body-rate action driving four physical rotors."""

  loaded_mass: float = 1.1628
  gravity: float = 9.81
  thrust_to_weight_max: float = 3.5
  max_rates: tuple[float, float, float] = (15.0, 15.0, 5.0)
  kp: tuple[float, float, float] = (0.020, 0.020, 0.020)
  ki: tuple[float, float, float] = (0.0, 0.0, 0.0)
  kd: tuple[float, float, float] = (0.0004, 0.0004, 0.0002)
  integral_limit: tuple[float, float, float] = (3.0, 3.0, 2.0)
  rotor_max: float = 20.5

  def __post_init__(self):
    self.transmission_type = TransmissionType.SITE
    self.preserve_order = True

  def build(self, env):
    return FlareRotorRateAction(self, env)


class FlareRotorRateAction(BaseAction):
  cfg: FlareRotorRateActionCfg

  def __init__(self, cfg: FlareRotorRateActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    if self.action_dim != 4:
      raise ValueError(f"expected four motor sites, got {self.action_dim}")
    self.kp = torch.tensor(cfg.kp, device=self.device)
    self.ki = torch.tensor(cfg.ki, device=self.device)
    self.kd = torch.tensor(cfg.kd, device=self.device)
    self.max_rates = torch.tensor(cfg.max_rates, device=self.device)
    self.integral_limit = torch.tensor(cfg.integral_limit, device=self.device)
    self.integral = torch.zeros(self.num_envs, 3, device=self.device)
    self.previous_rate = torch.zeros_like(self.integral)
    self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    # Columns follow M1 RR, M2 FR, M3 RL, M4 FL.
    effectiveness = torch.tensor(
      [
        [1.0, 1.0, 1.0, 1.0],
        [-0.1, -0.1, 0.1, 0.1],
        [0.1, -0.1, 0.1, -0.1],
        [0.012, -0.012, -0.012, 0.012],
      ],
      device=self.device,
    )
    self.inverse_effectiveness = torch.linalg.inv(effectiveness)
    self.rotor_thrusts = torch.zeros(self.num_envs, 4, device=self.device)

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    clipped = actions.clamp(-1.0, 1.0)
    self._processed_actions[:] = clipped
    collective_max = (
      self.cfg.thrust_to_weight_max * self.cfg.loaded_mass * self.cfg.gravity
    )
    collective = 0.5 * (clipped[:, 0] + 1.0) * collective_max
    desired_rate = clipped[:, 1:4] * self.max_rates
    measured_rate = self._entity.data.root_link_ang_vel_b
    error = desired_rate - measured_rate
    candidate_integral = torch.clamp(
      self.integral + error * self._env.step_dt,
      min=-self.integral_limit,
      max=self.integral_limit,
    )
    derivative = torch.where(
      self.initialized[:, None],
      (measured_rate - self.previous_rate) / self._env.step_dt,
      torch.zeros_like(measured_rate),
    )
    moments = self.kp * error + self.ki * candidate_integral - self.kd * derivative
    wrench = torch.cat((collective[:, None], moments), dim=1)
    unconstrained = wrench @ self.inverse_effectiveness.T
    self.rotor_thrusts[:] = unconstrained.clamp(0.0, self.cfg.rotor_max)
    saturated = torch.any(torch.abs(self.rotor_thrusts - unconstrained) > 1e-7, dim=1)
    self.integral[:] = torch.where(
      saturated[:, None], self.integral, candidate_integral
    )
    self.previous_rate[:] = measured_rate
    self.initialized[:] = True

  def apply_actions(self) -> None:
    self._entity.set_site_effort_target(self.rotor_thrusts, site_ids=self._target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self.integral[env_ids] = 0.0
    self.previous_rate[env_ids] = 0.0
    self.initialized[env_ids] = False
    self.rotor_thrusts[env_ids] = 0.0


@dataclass(kw_only=True)
class FlareWaypointCommandCfg(CommandTermCfg):
  entity_name: str = "quadrotor"
  sampling_entity_name: str | None = None
  sample_relative_to_entity: bool = False
  relative_sampling_axes: tuple[bool, bool, bool] = (True, True, True)
  x_range: tuple[float, float] = (-1.5, 1.5)
  y_range: tuple[float, float] = (-1.5, 1.5)
  z_range: tuple[float, float] = (0.5, 1.5)
  arrival_threshold: float = 0.2

  def build(self, env):
    return FlareWaypointCommand(self, env)


class FlareWaypointCommand(CommandTerm):
  cfg: FlareWaypointCommandCfg

  def __init__(self, cfg: FlareWaypointCommandCfg, env):
    super().__init__(cfg, env)
    self.asset = env.scene[cfg.entity_name]
    self.sampling_asset = (
      env.scene[cfg.sampling_entity_name]
      if cfg.sampling_entity_name is not None
      else self.asset
    )
    self.current = torch.zeros(self.num_envs, 3, device=self.device)
    self.next = torch.zeros_like(self.current)
    # Snapshots reproduce FLARE's reached-waypoint transition observation.
    self.observation_current = torch.zeros_like(self.current)
    self.observation_next = torch.zeros_like(self.current)
    self.metrics["distance"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["reached"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.current

  def _sample(self, env_ids: torch.Tensor) -> torch.Tensor:
    values = torch.empty(len(env_ids), 3, device=self.device)
    values[:, 0].uniform_(*self.cfg.x_range)
    values[:, 1].uniform_(*self.cfg.y_range)
    values[:, 2].uniform_(*self.cfg.z_range)
    if self.cfg.sample_relative_to_entity:
      # Read the free-joint qpos directly. During reset, MuJoCo's derived xpos
      # cache has not been forwarded yet, while qpos already contains the new
      # per-environment world position.
      entity_data = self.sampling_asset.data
      free_pos_ids = entity_data.indexing.free_joint_q_adr[:3]
      anchor = entity_data.data.qpos[env_ids][:, free_pos_ids]
      base = self._env.scene.env_origins[env_ids].clone()
      for axis, relative in enumerate(self.cfg.relative_sampling_axes):
        if relative:
          base[:, axis] = anchor[:, axis]
      return values + base
    return values + self._env.scene.env_origins[env_ids]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.current[env_ids] = self._sample(env_ids)
    self.next[env_ids] = self._sample(env_ids)
    self.observation_current[env_ids] = self.current[env_ids]
    self.observation_next[env_ids] = self.next[env_ids]

  def _update_metrics(self) -> None:
    distance = torch.linalg.norm(self.current - self.asset.data.root_link_pos_w, dim=1)
    self.metrics["distance"][:] = distance
    self.metrics["reached"][:] = (distance < self.cfg.arrival_threshold).float()

  def _update_command(self) -> None:
    self.observation_current[:] = self.current
    self.observation_next[:] = self.next
    reached = self.metrics["reached"].bool()
    env_ids = reached.nonzero(as_tuple=False).flatten()
    if len(env_ids):
      self.current[env_ids] = self.next[env_ids]
      self.next[env_ids] = self._sample(env_ids)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for env_idx in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=self.current[env_idx],
        radius=0.05,
        color=(0.1, 0.9, 0.2, 0.9),
        label=f"current_waypoint_{env_idx}",
      )
      visualizer.add_sphere(
        center=self.next[env_idx],
        radius=0.035,
        color=(0.2, 0.5, 1.0, 0.7),
        label=f"next_waypoint_{env_idx}",
      )


def current_waypoint_rel(env, asset_cfg: SceneEntityCfg = QUAD_CFG) -> torch.Tensor:
  asset = _quad(env, asset_cfg)
  relative = _waypoints(env).observation_current - asset.data.root_link_pos_w
  scale = torch.tensor((1.0 / 3.0, 1.0 / 3.0, 1.0), device=env.device)
  return (relative * scale).clamp(-1.0, 1.0)


def next_waypoint_rel(env, asset_cfg: SceneEntityCfg = QUAD_CFG) -> torch.Tensor:
  asset = _quad(env, asset_cfg)
  relative = _waypoints(env).observation_next - asset.data.root_link_pos_w
  scale = torch.tensor((1.0 / 3.0, 1.0 / 3.0, 1.0), device=env.device)
  return (relative * scale).clamp(-1.0, 1.0)


def payload_target_rel(env, asset_cfg: SceneEntityCfg = PAYLOAD_CFG) -> torch.Tensor:
  """Current target relative to the payload for payload-targeting policies."""
  asset = _payload(env, asset_cfg)
  relative = _waypoints(env).observation_current - asset.data.root_link_pos_w
  scale = torch.tensor((1.0 / 3.0, 1.0 / 3.0, 1.0), device=env.device)
  return (relative * scale).clamp(-1.0, 1.0)


def linear_velocity_world(env, asset_cfg: SceneEntityCfg = QUAD_CFG) -> torch.Tensor:
  velocity = _quad(env, asset_cfg).data.root_link_lin_vel_w
  scale = torch.tensor((0.1, 0.1, 0.3), device=env.device)
  return (velocity * scale).clamp(-1.0, 1.0)


def rotation_matrix_flat(env, asset_cfg: SceneEntityCfg = QUAD_CFG) -> torch.Tensor:
  return matrix_from_quat(_quad(env, asset_cfg).data.root_link_quat_w).reshape(
    env.num_envs, 9
  )


def previous_action(env) -> torch.Tensor:
  return env.action_manager.prev_action


class CableAngleObservation(ManagerTermBase):
  """Released FLARE center-vector world angles and finite-difference rates."""

  def __init__(self, cfg, env):
    del cfg
    super().__init__(env)
    self.previous = torch.zeros(self.num_envs, 2, device=self.device)
    self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def _angles(self) -> torch.Tensor:
    vector = (
      _payload(self._env).data.root_link_pos_w - _quad(self._env).data.root_link_pos_w
    )
    return torch.stack(
      (
        torch.atan2(vector[:, 0], -vector[:, 2]),
        torch.atan2(vector[:, 1], -vector[:, 2]),
      ),
      dim=1,
    )

  def __call__(self, env) -> torch.Tensor:
    del env
    angles = self._angles()
    delta = (
      torch.remainder(angles - self.previous + torch.pi, 2.0 * torch.pi) - torch.pi
    )
    rates = torch.where(
      self.initialized[:, None], delta / self._env.step_dt, torch.zeros_like(delta)
    )
    self.previous[:] = angles
    self.initialized[:] = True
    return torch.cat((angles * 0.66666667, rates * 0.1), dim=1).clamp(-1.0, 1.0)

  def reset(self, env_ids) -> None:
    self.previous[env_ids] = 0.0
    self.initialized[env_ids] = False


def cable_body_angle(env) -> torch.Tensor:
  vector = _payload(env).data.root_link_pos_w - _quad(env).data.root_link_pos_w
  direction = vector / torch.linalg.norm(vector, dim=1, keepdim=True).clamp_min(1e-8)
  rot = matrix_from_quat(_quad(env).data.root_link_quat_w)
  downward_body_z_world = -rot[:, :, 2]
  return torch.acos(
    torch.sum(direction * downward_body_z_world, dim=1).clamp(-1.0, 1.0)
  )


class TargetProgressReward(ManagerTermBase):
  def __init__(self, cfg, env):
    del cfg
    super().__init__(env)
    self.previous_position = torch.zeros(self.num_envs, 3, device=self.device)

  def __call__(self, env) -> torch.Tensor:
    position = _quad(env).data.root_link_pos_w
    command = _waypoints(env).current
    before = command - self.previous_position
    after = command - position
    reward = torch.sum(before.square() - after.square(), dim=1)
    self.previous_position[:] = position
    return reward

  def reset(self, env_ids) -> None:
    self.previous_position[env_ids] = _quad(self._env).data.root_link_pos_w[env_ids]


class PayloadTargetProgressReward(ManagerTermBase):
  """FLARE Scenario-II progress of the payload toward the active target."""

  def __init__(self, cfg, env):
    del cfg
    super().__init__(env)
    self.previous_position = torch.zeros(self.num_envs, 3, device=self.device)

  def __call__(self, env) -> torch.Tensor:
    position = _payload(env).data.root_link_pos_w
    command = _waypoints(env).current
    before = command - self.previous_position
    after = command - position
    reward = torch.sum(before.square() - after.square(), dim=1)
    self.previous_position[:] = position
    return reward

  def reset(self, env_ids) -> None:
    self.previous_position[env_ids] = _payload(self._env).data.root_link_pos_w[env_ids]


def action_smoothness(env) -> torch.Tensor:
  return torch.sum(
    (env.action_manager.action - env.action_manager.prev_action).square(), dim=1
  )


def action_smoothness_l2(env) -> torch.Tensor:
  """Paper Eq. (8): Euclidean norm of consecutive action changes."""
  return torch.linalg.vector_norm(
    env.action_manager.action - env.action_manager.prev_action, dim=1
  )


def yaw_alignment(env) -> torch.Tensor:
  _, _, yaw = euler_xyz_from_quat(_quad(env).data.root_link_quat_w)
  return torch.exp(-10.0 * torch.abs(yaw / 180.0 * torch.pi))


def angular_rate_norm(env) -> torch.Tensor:
  return torch.linalg.norm(_quad(env).data.root_link_ang_vel_b / torch.pi, dim=1)


def crash_indicator(env) -> torch.Tensor:
  return env.reset_terminated.float()


def cable_safety_penalty(env) -> torch.Tensor:
  return -(
    cable_body_angle(env) > torch.deg2rad(torch.tensor(75.0, device=env.device))
  ).float()


class FlareCrashTermination(ManagerTermBase):
  def __init__(self, cfg, env):
    del cfg
    super().__init__(env)
    self.safety_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

  def __call__(self, env) -> torch.Tensor:
    asset = _quad(env)
    rel = _waypoints(env).current - asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)
    angle = cable_body_angle(env)
    self.safety_count += (
      angle > torch.deg2rad(torch.tensor(75.0, device=env.device))
    ).long()
    return (
      (torch.abs(torch.rad2deg(roll)) > 180.0)
      | (torch.abs(torch.rad2deg(pitch)) > 180.0)
      | (torch.abs(rel[:, 0]) > 3.0)
      | (torch.abs(rel[:, 1]) > 3.0)
      | (torch.abs(rel[:, 2]) > 2.0)
      | (asset.data.root_link_pos_w[:, 2] < 0.1)
      | (self.safety_count > 150)
      | (angle > 1.571)
    )

  def reset(self, env_ids) -> None:
    self.safety_count[env_ids] = 0


class FlarePayloadTargetCrashTermination(ManagerTermBase):
  """Terminate when either vehicle or payload leaves the target workspace."""

  def __init__(self, cfg, env):
    del cfg
    super().__init__(env)
    self.safety_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

  def __call__(self, env) -> torch.Tensor:
    quad = _quad(env)
    payload = _payload(env)
    target = _waypoints(env).current
    quad_rel = target - quad.data.root_link_pos_w
    payload_rel = target - payload.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(quad.data.root_link_quat_w)
    angle = cable_body_angle(env)
    self.safety_count += (
      angle > torch.deg2rad(torch.tensor(75.0, device=env.device))
    ).long()
    return (
      (torch.abs(torch.rad2deg(roll)) > 180.0)
      | (torch.abs(torch.rad2deg(pitch)) > 180.0)
      | (torch.abs(quad_rel[:, 0]) > 3.0)
      | (torch.abs(quad_rel[:, 1]) > 3.0)
      | (torch.abs(quad_rel[:, 2]) > 2.0)
      | (torch.abs(payload_rel[:, 0]) > 3.0)
      | (torch.abs(payload_rel[:, 1]) > 3.0)
      # The payload begins 0.71 m below the quadrotor, so its valid vertical
      # target error must include the cable length in addition to the 2 m
      # quadrotor bound.
      | (torch.abs(payload_rel[:, 2]) > 2.7)
      | (quad.data.root_link_pos_w[:, 2] < 0.1)
      | (payload.data.root_link_pos_w[:, 2] < 0.05)
      | (self.safety_count > 150)
      | (angle > torch.pi / 2.0)
    )

  def reset(self, env_ids) -> None:
    self.safety_count[env_ids] = 0
