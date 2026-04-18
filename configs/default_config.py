from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass
class ModelConfig:
    xml_path: str = (Path("model") / "fish_pool_scene.xml").as_posix()


@dataclass
class CameraObsConfig:
    camera_name: str = "head_camera"
    width: int = 84
    height: int = 84
    underwater_effect_enabled: bool = False
    visibility_distance: float = 2.50
    transmittance_at_visibility: float = 0.45
    max_depth_distance: float = 4.00
    water_color_rgb: tuple[int, int, int] = (52, 102, 128)
    blur_radius: int = 1
    max_blur_strength: float = 0.15
    far_noise_std: float = 1.5


@dataclass
class ImuObsConfig:
    accel_clip: float = 50.0
    gyro_clip: float = 20.0
    goal_relative_clip: float = 5.0


@dataclass
class MappingConfig:
    # PPO action a in [-1, 1] is mapped to theta_m in [-theta_m_max, theta_m_max].
    theta_m_max: float = 0.60
    # Real servos only ignore a very narrow band of tiny command jitter.
    head_servo_command_deadband: float = 0.01
    # Conservative 35 kg-class HV servo speed assumption, specified in common datasheet units.
    servo_speed_sec_per_60deg: float = 0.16
    servo_accel_deg_per_s2: float = 2200.0
    tail_servo_speed_sec_per_60deg: float = 0.18
    tail_servo_accel_deg_per_s2: float = 2600.0
    # High-level map: theta_m_max corresponds to this maximum head deflection.
    head_angle_max_deg: float = 70.0
    # Legacy tendon geometry values kept for reference.
    servo_output_radius: float = 0.0375
    head_joint_arm_length: float = 0.0725
    head_transmission_efficiency: float = 1.00
    base_tail_freq: float = 0.70
    tail_freq_filter_tau: float = 0.18
    # Symmetric tail-frequency boost: both left and right head deflections increase frequency.
    tail_freq_gain: float = 0.80
    tail_freq_min: float = 0.5
    tail_freq_max: float = 1.0
    back_servo_amplitude: float = 1.00
    # Fixed tail-center offset applied when the head is deflected; use a negative value to flip the turn side.
    back_servo_center_bias: float = 0.45
    back_servo_bias_head_deadband: float = 1e-4


@dataclass
class PathConfig:
    # Legacy path-following config kept only for backwards compatibility with helper utilities.
    kind: str = "straight"
    num_points: int = 241
    start_x: float = -1.90
    end_x: float = 1.90
    amplitude: float = 0.35
    wavelength: float = 2.40
    spawn_offset: float = 0.45
    lookahead_distance: float = 0.35
    success_radius: float = 0.18


@dataclass
class TaskConfig:
    spawn_x_range: tuple[float, float] = (-3.35, -2.95)
    spawn_y_range: tuple[float, float] = (-0.85, 0.85)
    spawn_yaw_range_deg: float = 8.0
    spawn_wall_margin: float = 0.06
    spawn_obstacle_margin: float = 0.05
    spawn_max_attempts: int = 200
    goal_center: tuple[float, float] = (3.45, 0.0)
    goal_half_extents: tuple[float, float] = (0.24, 0.40)


@dataclass
class ObstacleConfig:
    min_count: int = 6
    max_count: int = 8
    radius_min: float = 0.11
    radius_max: float = 0.18
    offset_min: float = 0.00
    offset_max: float = 0.00
    obstacle_spacing: float = 0.18
    start_goal_clearance: float = 0.80
    pair_progress_min: float = 0.46
    pair_progress_max: float = 0.54
    pair_inner_gap_min: float = 0.04
    pair_inner_gap_max: float = 0.10
    obs_detect_range: float = 0.90
    obs_fov_deg: float = 150.0
    safety_margin: float = 0.22
    resample_interval_episodes: int = 1000
    max_sampling_attempts: int = 100


@dataclass
class RewardConfig:
    # Composite dense reward:
    #   R_t = w_target * r_target
    #       + w_obs * r_obs
    #       + w_heading * r_heading
    #       + w_smooth * r_smooth
    #       - step_penalty
    #       - wall_collision_cost * hit_wall
    #       - timeout_penalty * timeout
    target_progress_scale: float = 1.0
    success_reward: float = 60.0
    obstacle_safe_distance: float = 0.70
    obstacle_buffer_distance: float = 0.95
    obstacle_danger_distance: float = 0.08
    obstacle_collision_penalty: float = 1.0
    obstacle_buffer_penalty: float = 0.12
    heading_error_scale: float = 1.0
    smooth_action_l2_scale: float = 0.12
    smooth_action_delta_scale: float = 0.20
    step_penalty: float = 0.01
    wall_collision_cost: float = 12.0
    timeout_penalty: float = 8.0
    w_target: float = 85.0
    w_obs: float = 14.0
    w_heading: float = 2.0
    w_smooth: float = 1.0
    goal_distance_clip: float = 1.0


@dataclass
class FishEnvConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    camera: CameraObsConfig = field(default_factory=CameraObsConfig)
    imu: ImuObsConfig = field(default_factory=ImuObsConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    frame_skip: int = 20
    action_filter_tau: float = 0.04
    max_episode_steps: int = 5000
    persistent_contact_termination_steps: int = 20
    pool_half_length: float = 4.20
    pool_half_width: float = 2.20
    fish_collision_radius: float = 0.18
    head_sensor_offset: float = 0.28
    render_size: tuple[int, int] = (900, 500)
    post_goal_duration_sec: float = 5.0


@dataclass
class TrainConfig:
    total_timesteps: int = 3_000_000
    num_envs: int = 4
    algorithm: str = "ppo"
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_hidden_sizes: tuple[int, ...] = (128, 128)
    seed: int = 7
    log_dir: str = (Path("runs") / "ppo_fish_baseline").as_posix()
    model_name: str = "ppo_fish_baseline"
    checkpoint_interval_timesteps: int = 0
    checkpoint_dirname: str = "checkpoints"
    checkpoint_metrics_filename: str = "checkpoint_metrics.csv"
    save_policy_weights: bool = True
    sac_buffer_size: int = 100_000
    sac_learning_starts: int = 5_000
    sac_train_freq_steps: int = 1
    sac_train_freq_unit: str = "step"
    sac_gradient_steps: int = 1
    sac_tau: float = 0.005
    sac_ent_coef: str = "auto"
    sac_target_update_interval: int = 1
    sac_target_entropy: str = "auto"
    sac_optimize_memory_usage: bool = False
    monitor_filename: str = "monitor.csv"
    episode_metrics_filename: str = "episode_metrics.csv"
    save_episode_videos: bool = False
    video_interval_episodes: int = 5
    video_dirname: str = "videos"
    video_camera_name: str = "top"
    video_width: int = 480
    video_height: int = 272
    video_fps: int = 10
    video_frame_stride: int = 4


@dataclass
class EvalConfig:
    episodes: int = 5
    deterministic: bool = True


@dataclass
class ExperimentConfig:
    env: FishEnvConfig = field(default_factory=FishEnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def make_config() -> ExperimentConfig:
    return ExperimentConfig()


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)
