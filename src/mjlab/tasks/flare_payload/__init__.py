from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import flare_payload_env_cfg, flare_payload_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Flare-Waypoint-Payload",
  env_cfg=flare_payload_env_cfg(),
  play_env_cfg=flare_payload_env_cfg(play=True),
  rl_cfg=flare_payload_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Flare-Waypoint-Payload-ACP-Tuned-V1",
  env_cfg=flare_payload_env_cfg(reward_profile="acp_tuned"),
  play_env_cfg=flare_payload_env_cfg(play=True, reward_profile="acp_tuned"),
  rl_cfg=flare_payload_ppo_runner_cfg(profile="acp_tuned"),
)

register_mjlab_task(
  task_id="Mjlab-Flare-Waypoint-Payload-Flare-Rewards",
  env_cfg=flare_payload_env_cfg(reward_profile="flare"),
  play_env_cfg=flare_payload_env_cfg(play=True, reward_profile="flare"),
  rl_cfg=flare_payload_ppo_runner_cfg(profile="flare"),
)
