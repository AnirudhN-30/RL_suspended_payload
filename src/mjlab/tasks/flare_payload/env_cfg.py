"""GPU-parallel FLARE Scenario-I task using the validated ACP plant."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import mujoco

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from . import mdp

_DIR = Path(__file__).parent

RewardProfile = Literal["flare", "acp_tuned", "acp_smooth"]
PpoProfile = Literal["flare", "acp_tuned", "acp_smooth"]

_REWARD_WEIGHTS = {
  "flare": {
    "target": 10.0,
    "smooth": -0.0001,
    "yaw": 0.01,
    "angular": -0.0002,
    "crash": -10.0,
    "cable_angle_safety": 0.03,
  },
  "acp_tuned": {
    "target": 5.0,
    "smooth": -0.005,
    "yaw": 0.01,
    "angular": -0.002,
    "crash": -20.0,
    "cable_angle_safety": 0.3,
  },
  "acp_smooth": {
    "target": 5.0,
    "smooth": -0.01,
    "yaw": 0.01,
    "angular": -0.005,
    "crash": -20.0,
    "cable_angle_safety": 1.0,
  },
}


def _quad_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_DIR / "quadrotor.xml"))


def _payload_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_DIR / "payload.xml"))


_QUAD_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(
    XmlMotorActuatorCfg(
      target_names_expr=(site,), transmission_type=TransmissionType.SITE
    )
    for site in mdp.MOTOR_SITES
  )
)


def _quad_entity() -> EntityCfg:
  return EntityCfg(
    spec_fn=_quad_spec,
    articulation=_QUAD_ARTICULATION,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 1.0),
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
    ),
  )


def _payload_entity() -> EntityCfg:
  return EntityCfg(
    spec_fn=_payload_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.29),
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
    ),
  )


def _add_payload_tendon(spec: mujoco.MjSpec) -> None:
  tendon = spec.add_tendon(
    name="payload_tendon", limited=True, range=(0.0, 0.60), width=0.0015
  )
  tendon.wrap_site("quadrotor/payload_attachment")
  tendon.wrap_site("payload/payload_connection")


def flare_payload_env_cfg(
  play: bool = False, reward_profile: RewardProfile = "flare"
) -> ManagerBasedRlEnvCfg:
  reward_weights = _REWARD_WEIGHTS[reward_profile]
  actor_terms = {
    # FLARE scales each axis before clipping. Keep that ordering in the term
    # functions because ObservationTermCfg applies its clip before its scale.
    "current_waypoint": ObservationTermCfg(func=mdp.current_waypoint_rel),
    "next_waypoint": ObservationTermCfg(func=mdp.next_waypoint_rel),
    "linear_velocity": ObservationTermCfg(func=mdp.linear_velocity_world),
    "rotation_matrix": ObservationTermCfg(func=mdp.rotation_matrix_flat),
    "previous_action": ObservationTermCfg(func=mdp.previous_action),
    "cable_state": ObservationTermCfg(func=mdp.CableAngleObservation),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, nan_policy="error"),
    "critic": ObservationGroupCfg({**actor_terms}, nan_policy="error"),
  }

  actions: dict[str, ActionTermCfg] = {
    "body_rate": mdp.FlareRotorRateActionCfg(
      entity_name="quadrotor",
      actuator_names=mdp.MOTOR_SITES,
      preserve_order=True,
    )
  }
  commands = {
    "waypoints": mdp.FlareWaypointCommandCfg(
      entity_name="quadrotor",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
    )
  }
  rewards = {
    "target": RewardTermCfg(
      func=mdp.TargetProgressReward, weight=reward_weights["target"]
    ),
    "smooth": RewardTermCfg(
      func=mdp.action_smoothness, weight=reward_weights["smooth"]
    ),
    "yaw": RewardTermCfg(func=mdp.yaw_alignment, weight=reward_weights["yaw"]),
    "angular": RewardTermCfg(
      func=mdp.angular_rate_norm, weight=reward_weights["angular"]
    ),
    "crash": RewardTermCfg(
      func=mdp.crash_indicator, weight=reward_weights["crash"]
    ),
    "cable_angle_safety": RewardTermCfg(
      func=mdp.cable_safety_penalty,
      weight=reward_weights["cable_angle_safety"],
    ),
  }
  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "crash": TerminationTermCfg(func=mdp.FlareCrashTermination),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"quadrotor": _quad_entity(), "payload": _payload_entity()},
      spec_fn=_add_payload_tendon,
      num_envs=4096,
      env_spacing=4.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(0.0, 0.0, 0.75),
      distance=3.0,
      elevation=-20.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      njmax=128,
      nconmax=16,
      mujoco=MujocoCfg(
        timestep=0.002,
        integrator="implicitfast",
        gravity=(0.0, 0.0, -9.81),
      ),
    ),
    decimation=5,
    episode_length_s=15.0,
    scale_rewards_by_dt=True,
  )
  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1.0e10
  return cfg


def flare_payload_ppo_runner_cfg(
  profile: PpoProfile = "flare",
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="mjlab.tasks.flare_payload.policy:FlareTanhActor",
      hidden_dims=(128, 128),
      activation="tanh",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 128), activation="tanh", obs_normalization=False
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef={"flare": 0.002, "acp_tuned": 0.001, "acp_smooth": 0.0005}[
        profile
      ],
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4 if profile == "flare" else 1.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name={
      "flare": "flare_payload_drone_relative_l2_tanh_v2",
      "acp_tuned": "flare_payload_acp_tuned",
      "acp_smooth": "flare_payload_acp_smooth_v2",
    }[profile],
    save_interval=100,
    num_steps_per_env=100,
    max_iterations=1000,
  )
