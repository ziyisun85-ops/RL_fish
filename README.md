# RL Fish PPO Baseline

基于 MuJoCo、Gymnasium 和 Stable-Baselines3 PPO 的机器鱼强化学习项目。

当前代码的核心任务是：机器鱼从水池左侧出生，依靠头部相机图像和 IMU 观测，控制前舵机与尾部摆动，绕开中间障碍并到达右侧目标区域。

## 目录结构

```text
RL_fish/
├── configs/
│   └── default_config.py          # 全局默认参数
├── envs/
│   └── fish_env.py                # Gymnasium 环境、reset/step、观测、奖励、渲染、录像
├── model/
│   ├── fish_pool_scene.xml        # 主场景，include 机器鱼和水池环境
│   ├── fish_robot_model.xml       # 机器鱼本体、舵机、拉索、actuator、sensor
│   ├── pool_environment.xml       # 水池、目标区、障碍物 marker、相机
│   └── rebuild_fish_from_stl.py   # 从 STL 重建 MuJoCo XML 的工具脚本
├── scenarios/
│   └── training_envs/             # 固定训练场景 JSON
├── utils/
│   ├── mappings.py                # 舵机角 -> 头角 -> 尾摆频率映射
│   ├── obstacles.py               # 障碍物采样与局部观测
│   ├── scenario_io.py             # 固定场景读写
│   └── geometry.py                # 坐标变换和角度归一化
├── hydrodynamics.py               # 分布式水动力近似
├── generate_training_envs.py      # 批量导出固定训练场景
├── train.py                       # PPO 训练入口
├── eval.py                        # PPO 评估入口
├── requirements.txt
└── runs/                          # 训练输出、日志、权重、视频
```

## 总体运行链路

```text
MuJoCo XML 场景
  -> FishPathAvoidEnv
  -> 头部相机 RGB 图像 + IMU
  -> PPO MultiInputPolicy
  -> 1 维连续动作 a in [-1, 1]
  -> 动作低通滤波
  -> 前舵机目标角 theta_m_target
  -> 前舵机速度/加速度限幅
  -> 头部角 theta_h
  -> 尾摆频率 tail_freq
  -> 后舵机正弦摆动 + 中心偏置
  -> apply_hydrodynamics()
  -> mujoco.mj_step()
  -> 奖励、终止标志、info、录像/日志
```

## 场景与模型关系

`model/fish_pool_scene.xml` 是默认入口，由 `configs/default_config.py` 的 `ModelConfig.xml_path` 指向。它只负责组合：

```xml
<include file="fish_robot_model.xml" />
<include file="pool_environment.xml" />
```

`fish_robot_model.xml` 定义机器鱼：

- 根部自由度：`root_x`、`root_y`、`root_yaw`。
- 舵机：`front_servo`、`back_servo`。
- actuator：`front_servo_act`、`back_servo_act`，都是 MuJoCo `<position>` actuator，`kp=120`，`ctrlrange="-1.1 1.1"`。
- 头部相机：`head_camera`，挂在 `head` body 上，`fovy=120`。
- 动力学碰撞几何：名称以 `_dyn` 结尾，供碰撞、清距和水动力使用。
- 可视化几何：名称以 `_vis` 结尾，主要用于渲染。

`pool_environment.xml` 定义环境：

- MuJoCo 步长：`timestep=0.002`。
- 全局流体参数：`density=850`，`viscosity=0.0007`。
- 重力：`gravity="0 0 0"`。
- 水池可视体：`water_volume`。
- 目标区 marker：`goal_region_marker`。
- 障碍物 marker：`visual_obstacle_0` 到 `visual_obstacle_7`，由环境 reset 时移动到采样位置。
- 观察相机：`top` 和 `oblique`，训练录像默认用 `top`。

## 观测、动作与策略输入

环境类是 `envs.fish_env.FishPathAvoidEnv`。

动作空间：

```text
Box(low=-1.0, high=1.0, shape=(1,), dtype=float32)
```

观测空间是字典：

| key | 形状 | 类型 | 含义 |
| --- | --- | --- | --- |
| `image` | `(84, 84, 3)` | `uint8` | 头部相机 `head_camera` 的 RGB 图像 |
| `imu` | `(5,)` | `float32` | `[ax_body, ay_body, yaw_rate, goal_dx_body, goal_dy_body]` |

IMU 裁剪范围：

| 字段 | 默认值 |
| --- | --- |
| `accel_clip` | `50.0` |
| `gyro_clip` | `20.0` |
| `goal_relative_clip` | `5.0` |

其中 `goal_dx_body` 和 `goal_dy_body` 是目标区域最近点相对于机器鱼当前位置的鱼体坐标系位移：

- `goal_dx_body > 0` 表示目标在鱼体前方。
- `goal_dx_body < 0` 表示目标在鱼体后方。
- `goal_dy_body > 0` 表示目标在鱼体左侧。
- `goal_dy_body < 0` 表示目标在鱼体右侧。

环境内部先用当前位置投影到目标矩形得到 `goal_target`，再把世界系相对位移 `goal_target - position` 变换到鱼体坐标系。

PPO 使用 `MultiInputPolicy`。训练时用 `VecTransposeImage` 把图像从 HWC 转成 Stable-Baselines3 需要的 CHW 格式。

## 控制映射关系

动作不是直接推力，而是高层舵机命令。当前实现的映射链路如下。

1. PPO 输出原始动作：

```text
a = clip(action[0], -1, 1)
```

2. 头舵机死区：

```text
abs(a) <= head_servo_command_deadband 时锁定头舵机输入
head_servo_command_deadband = 0.01
```

3. 一阶低通滤波：

```text
alpha = 1 - exp(-control_timestep / action_filter_tau)
filtered_action += alpha * (input_action - filtered_action)
action_filter_tau = 0.04
```

4. PPO 动作映射到前舵机目标角：

```text
theta_m_target = filtered_action * theta_m_max
theta_m_max = 0.60 rad
```

5. 前舵机速度和加速度限幅：

```text
servo_speed_sec_per_60deg = 0.16
servo_accel_deg_per_s2 = 2200.0
theta_m 被限制在 [-0.60, 0.60] rad
```

6. 前舵机角映射到头部角：

```text
theta_h = radians(head_angle_max_deg) * clip(theta_m / theta_m_max, -1, 1)
head_angle_max_deg = 70.0
```

对应代码：`utils/mappings.py::servo_angle_to_head_angle()`。

7. 头部角映射到尾摆频率：

```text
tail_freq_target = base_tail_freq * (1 + tail_freq_gain * abs(theta_h))
tail_freq = clip(tail_freq_target, tail_freq_min, tail_freq_max)
base_tail_freq = 0.70
tail_freq_gain = 0.80
tail_freq_min = 0.50
tail_freq_max = 1.00
tail_freq_filter_tau = 0.18
```

对应代码：`utils/mappings.py::head_angle_to_tail_frequency()`。

8. 后舵机摆动：

```text
tail_phase += 2 * pi * tail_freq * sim_timestep

if abs(theta_h) > back_servo_bias_head_deadband:
    back_servo_bias = sign(back_servo_center_bias * theta_h) * min(abs(back_servo_center_bias), back_servo_amplitude)
else:
    back_servo_bias = 0

back_servo_wave_amplitude = back_servo_amplitude - abs(back_servo_bias)
back_servo_target = back_servo_bias + back_servo_wave_amplitude * sin(tail_phase)
```

后舵机默认参数：

| 参数 | 默认值 |
| --- | --- |
| `back_servo_amplitude` | `1.00` |
| `back_servo_center_bias` | `0.45` |
| `back_servo_bias_head_deadband` | `1e-4` |
| `tail_servo_speed_sec_per_60deg` | `0.18` |
| `tail_servo_accel_deg_per_s2` | `2600.0` |

9. 写入 MuJoCo 控制量：

```text
data.ctrl[front_servo_act] = theta_m
data.ctrl[back_servo_act] = back_servo_command
```

10. 每个 MuJoCo 子步中先施加水动力，再推进仿真：

```text
apply_hydrodynamics(model, data)
mujoco.mj_step(model, data)
```

## 环境 reset 逻辑

reset 时环境会：

1. 清空 episode 计数、动作滤波器、舵机状态、IMU、奖励缓存和状态标志。
2. 如果传入 `scenario_path`，读取固定场景 JSON，使用其中的出生点、目标区和障碍物。
3. 如果没有固定场景，则按配置采样障碍物和出生姿态。
4. 将 `goal_region_marker` 和 `visual_obstacle_*` 同步到 MuJoCo 场景。
5. 将机器鱼放到初始位姿，清零速度和舵机控制量。
6. 调用 `mj_forward()`，更新缓存状态。
7. 开始 episode 录像，并采集第一帧。

随机采样时，障碍物不是每个 episode 都重新采样。默认：

```text
resample_interval_episodes = 1000
```

也就是说同一组障碍物会连续使用 1000 个 episode；出生姿态每次 reset 都会重新采样。

## 任务区域与障碍物

出生区域：

| 参数 | 默认值 |
| --- | --- |
| `spawn_x_range` | `(-2.20, -1.85)` |
| `spawn_y_range` | `(-0.55, 0.55)` |
| `spawn_yaw_range_deg` | `8.0` |
| `spawn_wall_margin` | `0.06` |
| `spawn_obstacle_margin` | `0.05` |
| `spawn_max_attempts` | `200` |

目标区域：

| 参数 | 默认值 |
| --- | --- |
| `goal_center` | `(2.55, 0.0)` |
| `goal_half_extents` | `(0.20, 0.35)` |

障碍物默认是一对圆柱：

| 参数 | 默认值 |
| --- | --- |
| `min_count` / `max_count` | `2` / `2` |
| `radius_min` / `radius_max` | `0.12` / `0.18` |
| `pair_progress_min` / `pair_progress_max` | `0.46` / `0.54` |
| `pair_inner_gap_min` / `pair_inner_gap_max` | `0.04` / `0.10` |
| `start_goal_clearance` | `0.70` |
| `obs_detect_range` | `0.90` |
| `obs_fov_deg` | `150.0` |
| `safety_margin` | `0.22` |
| `max_sampling_attempts` | `100` |

障碍物采样逻辑在 `utils/obstacles.py`：

- 以出生区域中心到目标中心的连线作为走廊中心线。
- 在进度 `0.46` 到 `0.54` 附近采样障碍物对的纵向位置。
- 两个障碍物分别放在中心线法向两侧。
- 中间间隙由 `pair_inner_gap_min/max` 控制。
- 采样时检查水池边界、出生区 clearance 和目标区 clearance。
- 如果随机采样失败，回退到固定的中心阻挡障碍物对。

## 固定场景 JSON

固定训练场景保存在：

```text
scenarios/training_envs/
```

已有 manifest：

```text
scenarios/training_envs/manifest.json
```

当前 manifest 记录 `20` 个场景，seed 从 `7` 到 `26`。单个 JSON 的字段结构如下：

```json
{
  "scenario_id": "training_env_01",
  "source_seed": 7,
  "spawn_position": [-2.094941800281071, 0.4109087899358881],
  "spawn_yaw": -0.22639001121962732,
  "goal_center": [2.55, 0.0],
  "goal_half_extents": [0.2, 0.35],
  "obstacles": [
    {"center": [0.1619258315365566, 0.20077629870363584], "radius": 0.15750572799628001},
    {"center": [0.1619258315365566, -0.2171033987655303], "radius": 0.17383282805817452}
  ]
}
```

重新生成固定场景：

```bash
python generate_training_envs.py --count 20 --base-seed 7
```

训练指定场景：

```bash
python train.py --scenario-index 1
python train.py --scenario-path scenarios/training_envs/training_env_01.json
```

指定场景训练时，输出目录会追加场景名，例如：

```text
runs/ppo_fish_baseline/training_env_01/
```

## 视觉与水下成像

默认相机配置：

| 参数 | 默认值 |
| --- | --- |
| `camera_name` | `head_camera` |
| `width` / `height` | `84` / `84` |
| `underwater_effect_enabled` | `False` |
| `visibility_distance` | `2.50` |
| `transmittance_at_visibility` | `0.45` |
| `max_depth_distance` | `4.00` |
| `water_color_rgb` | `(52, 102, 128)` |
| `blur_radius` | `1` |
| `max_blur_strength` | `0.15` |
| `far_noise_std` | `1.5` |

当 `underwater_effect_enabled=False` 时：

- 策略收到原始头部相机 RGB。
- `visual_obstacle_detected` 始终为空状态。

当 `underwater_effect_enabled=True` 时：

- 环境会渲染深度图，按距离混入水体颜色、模糊和噪声。
- 环境会渲染 segmentation 图，判断 `visual_obstacle_*` 是否进入头部相机视野。
- 策略本身仍然只接收 RGB 图像和 IMU，不直接接收 segmentation 或障碍物真值。

## 水动力实现

水动力入口是 `hydrodynamics.py::apply_hydrodynamics()`，在每个 MuJoCo 子步执行一次。

### 建模思路

当前实现采用二维平面内的分布式 Resistive Force Theory, RFT 近似。它不做完整 CFD 求解，也不显式求解水体压力场，而是把鱼体离散成多个随 MuJoCo body 运动的水动力段，对每一段计算相对于水体的局部速度，并把局部速度分解到鱼体自身坐标系的切向和法向：

```text
v_rel,i = v_body,i - v_water
u_i     = dot(v_rel,i, t_i)
w_i     = dot(v_rel,i, n_i)
```

其中：

- `i` 是水动力段编号。
- `v_body,i` 是第 `i` 段 body 的世界系线速度。
- `v_water` 是环境水流速度，默认静水 `(0, 0, 0)`。
- `t_i` 是该段 body x 轴投影到水平面的单位向量，表示鱼体局部切向。
- `n_i` 是该段 body y 轴投影到水平面的单位向量，表示鱼体局部法向。
- `u_i` 是切向相对速度标量。
- `w_i` 是法向相对速度标量。

仿生鱼推进的核心是尾部摆动让各段产生不同的局部法向速度。法向阻力通常显著大于切向阻力，分布式法向力在不同 body 上形成合力和偏航力矩，因此鱼可以自然地产生前进、减速和转弯，而不是直接给根节点施加一个人为 yaw torque。

如果把第 `i` 段 body 的平面姿态记成旋转矩阵 `R_i` 在水平面的投影，那么代码中的平面局部基向量等价于：

```text
t_i = normalize(([R_i e_x]_{xy}, 0))
n_i = normalize(([R_i e_y]_{xy}, 0))
```

其中：

- `e_x = [1, 0, 0]^T` 表示 body 局部前向轴。
- `e_y = [0, 1, 0]^T` 表示 body 局部侧向轴。
- `([·]_{xy}, 0)` 表示只保留水平面内分量，把 z 分量置 0。

于是相对速度可以写成：

```text
v_rel,i = u_i t_i + w_i n_i + v_{z,i} e_z
u_i     = t_i^T v_rel,i
w_i     = n_i^T v_rel,i
```

当前实现只关心平面内推进和转向，因此 `v_{z,i}` 对水动力的贡献被忽略。

从经典流体阻力角度，常见表达是：

```text
F_D = -1/2 * rho * C_D * A * |v| * v
```

当前代码没有把 `rho`、`C_D`、`A` 单独拆开建模，而是把它们吸收到集总系数里。因此：

```text
C_t  ~ 等效切向阻力系数
C_n,i ~ 等效法向阻力系数
```

这也是为什么代码里的系数不能直接按真实流体力学中的无量纲阻力系数理解。

### 分段受力公式

代码会找出所有名字以 `_dyn` 结尾的 MuJoCo geom，并按其所属 body 汇总为水动力段。每段使用三类力：

```text
F_t,i = -C_t * u_i * t_i
F_n,i = -C_n,i * abs(w_i) * w_i * n_i
F_a,i = -m_a,i * a_n,i * n_i

F_i = clip_norm(F_t,i + F_n,i + F_a,i, max_segment_force)
```

其中：

- `F_t,i` 是切向线性阻力。切向阻力较小，主要抑制鱼体沿自身长度方向的相对滑移。
- `F_n,i` 是法向二次阻力。它与 `abs(w_i) * w_i` 成正比，方向始终与法向相对运动相反，是鱼体摆动推进和转弯的主要水动力来源。
- `F_a,i` 是法向附加质量力，用上一子步和当前子步的法向速度差估计非定常加速度。
- `clip_norm` 对单段合力做范数限幅，避免速度突变或初始状态导致不稳定力峰值。

分段法向阻力系数为：

```text
C_n,i = normal_drag * normal_multiplier(body_name_i)

normal_multiplier =
  head_normal_multiplier,  body_name == "head"
  tail_normal_multiplier,  body_name == "joint_tail" or body_name startswith "tail_"
  1.0,                     otherwise
```

如果把各段几何差异写成显式形状倍率，可以把上式理解成：

```text
C_n,i = C_n * k_shape,i
```

其中：

- `C_n = normal_drag`
- `k_shape,i = 1.5` 对头部更大，表示头部侧向迎流面积更大
- `k_shape,i = 1.2` 对尾部和尾关节略大，表示尾段摆动更容易产生有效法向反力
- 其他 body 取 `1.0`

附加质量项为：

```text
a_n,i = (w_i(t) - w_i(t - dt)) / dt
m_a,i = lateral_added_mass_fraction * body_mass_i
F_a,i = -m_a,i * a_n,i * n_i
```

第一步或没有上一状态时，`a_n,i` 取 `0`，避免用无效历史速度生成瞬时冲击力。

从连续时间观点，附加质量项本质上对应：

```text
F_a,i = -m_a,i * d(w_i n_i)/dt
```

当前代码进一步简化为只保留法向标量速度变化：

```text
F_a,i ≈ -m_a,i * (dw_i/dt) * n_i
dw_i/dt ≈ (w_i^k - w_i^{k-1}) / dt
```

这会忽略法向方向本身快速转动带来的高阶项，但在当前二维平面近似下可以换来更稳定的数值表现。

每段 `F_i` 通过 `mujoco.mj_applyFT()` 施加到该段 body 的当前位置。由于力的作用点不在整体质心，MuJoCo 会自动把它转换成等效广义力并产生对应力矩。代码中记录的

```text
segment_yaw_moment = sum(cross(r_i - r_ref, F_i).z)
```

主要用于诊断分布式力产生的偏航趋势，不会再额外重复施加一次。

如果把三维叉乘写成平面标量力矩，就是：

```text
M_{z,segments} = sum_i ((x_i - x_ref) * F_{y,i} - (y_i - y_ref) * F_{x,i})
```

它和 `cross(r_i - r_ref, F_i).z` 完全等价。

从 MuJoCo 广义力角度，`mj_applyFT()` 可以理解成：

```text
Q_hydro = sum_i J_i^T [F_i, tau_i]
```

其中 `J_i` 是作用点的雅可比矩阵，当前分段直接施加的是力 `F_i`，而局部附加力矩 `tau_i` 取 `0`。

### 整体偏航阻尼和附加转动惯量

分布式阻力之外，代码还给整体偏航加入一个简化的阻尼和附加转动惯量力矩：

```text
tau_damp  = -yaw_damping_linear * omega_z
            -yaw_damping_quadratic * abs(omega_z) * omega_z

I_z       = sum(I_body,zz + body_mass_i * ((x_i - x_ref)^2 + (y_i - y_ref)^2))
I_added   = yaw_added_inertia_fraction * I_z
alpha_z   = (omega_z(t) - omega_z(t - dt)) / dt
tau_added = -I_added * alpha_z

tau_z     = clip(tau_damp + tau_added, -max_yaw_torque, max_yaw_torque)
```

其中 `omega_z` 优先从 `root_yaw` 关节速度读取；如果模型没有这个关节，则从中心 body 的世界系角速度 z 分量读取。参考 body 优先使用 `centre_compartment`，不存在时使用第一个水动力段。

这个 yaw 项的作用是模拟鱼体整体转动时的水阻和非定常流体惯性，防止偏航角速度无限增长。它不是用来替代尾部摆动产生的转弯力矩，真正的转弯趋势仍主要来自分段力在不同 body 上的力臂。

这里的 `I_z` 可以看成各 body 偏航惯量与平行轴项之和：

```text
I_z = sum_i (I_{zz,i} + m_i r_i^2)
r_i^2 = (x_i - x_ref)^2 + (y_i - y_ref)^2
```

代码中的

```text
I_added = yaw_added_inertia_fraction * I_z
```

不是严格的流体附加惯量解析式，而是一个经验比例项，用来近似“水里拧身子比真空里更费劲”的效果。

如果从离散时间实现角度看，`apply_hydrodynamics()` 每个 MuJoCo 子步的顺序可以概括为：

```text
1. 读取当前各段位置 r_i^k、速度 v_i^k、整体偏航角速度 omega_z^k
2. 用上一子步缓存估计法向加速度与偏航角加速度
3. 计算各段 F_i^k
4. 计算整体偏航力矩 tau_z^k
5. 把这些力/力矩写入 data.qfrc_applied
6. 调用 mujoco.mj_step() 推进一步，得到 k+1 时刻状态
7. 缓存 w_i^k 和 omega_z^k 供下一子步使用
```

时间步长取：

```text
dt = max(t_k - t_{k-1}, 1e-8)
```

第一次调用没有历史状态时：

- 法向附加质量项取 `0`
- yaw 附加惯量项取 `0`

这样可以避免 reset 后第一步出现假的冲击力。

### 水动力参数

这些参数定义在 `hydrodynamics.py::HydrodynamicsConfig`。当前系数是 MuJoCo 场景尺度下的集总系数，已经把水密度、截面积、形状阻力等因素合并到一个数值里，因此不应直接按真实 SI 阻力系数解释。

| 参数 | 默认值 |
| --- | --- |
| `tangential_drag` | `0.1` |
| `normal_drag` | `4.0` |
| `head_normal_multiplier` | `1.5` |
| `tail_normal_multiplier` | `1.2` |
| `yaw_damping_linear` | `0.05` |
| `yaw_damping_quadratic` | `0.01` |
| `lateral_added_mass_fraction` | `0.5` |
| `yaw_added_inertia_fraction` | `0.2` |
| `max_segment_force` | `8.0` |
| `max_yaw_torque` | `2.0` |
| `water_velocity` | `(0.0, 0.0, 0.0)` |

参数含义：

| 参数 | 作用 | 调大后的典型效果 |
| --- | --- | --- |
| `tangential_drag` | 切向线性阻力系数 `C_t` | 鱼体沿自身轴向滑行更困难，惯性滑行距离变短 |
| `normal_drag` | 基础法向二次阻力系数 `C_n` | 摆尾产生的侧向反作用力更强，推进和转弯更敏感，但过大可能抖动 |
| `head_normal_multiplier` | 头部法向阻力倍率 | 头部横向扫动更受阻，航向变化更稳但阻尼更大 |
| `tail_normal_multiplier` | 尾部和尾关节法向阻力倍率 | 尾部摆动产生更强侧向水动力，推进和转向能力增强 |
| `yaw_damping_linear` | 偏航角速度线性阻尼 | 小角速度下更快衰减偏航运动 |
| `yaw_damping_quadratic` | 偏航角速度二次阻尼 | 大角速度下强烈抑制快速打转 |
| `lateral_added_mass_fraction` | 法向附加质量占 body 质量的比例 | 法向速度变化更有惯性，快速横摆会受到更强反作用 |
| `yaw_added_inertia_fraction` | 偏航附加转动惯量占估计 yaw 惯量的比例 | 偏航角加速度更难突变，转向响应更钝但更稳定 |
| `max_segment_force` | 单个水动力段合力限幅 | 防止局部数值峰值；过小会削弱推进，过大可能不稳定 |
| `max_yaw_torque` | 整体 yaw 阻尼和附加力矩限幅 | 防止全局偏航阻尼过强或数值发散 |
| `water_velocity` | 世界系水流速度 `(vx, vy, vz)` | 非零时水动力基于鱼体相对水流速度计算，可模拟恒定流场 |

实现细节：

- `data.qfrc_applied[:] = 0.0` 会在每个水动力子步开头清空外力槽，然后写入本次水动力广义力。
- 当前模型只使用水平面内的 `t_i`、`n_i`，会把 body 轴的 z 分量置零后归一化。
- `get_last_hydrodynamics_diagnostics()` 可读取上一子步的 `segment_count`、`segment_yaw_moment`、`damping_yaw_torque`、`added_yaw_torque` 和 `max_segment_force`。
- `reset_hydrodynamics_state()` 会清空上一子步速度缓存，环境 reset 时应调用，避免跨 episode 的附加质量项污染。

## 奖励函数

奖励在 `FishPathAvoidEnv._get_reward()` 中组合。

### 总体结构

每个控制步的奖励是稠密奖励，目标是让策略同时学会靠近目标、远离障碍、保持朝向和避免舵机指令剧烈变化：

```text
R_t =
  w_target  * r_target
  + w_obs   * r_obs
  + w_heading * r_heading
  + w_smooth  * r_smooth
  - step_penalty
  - wall_collision_cost * 1_wall_collision
  - timeout_penalty * 1_timeout
  + success_reward * 1_reached_goal
```

代码中的 `reward_terms["target_reward"]`、`reward_terms["obstacle_reward"]`、`reward_terms["heading_reward"]`、`reward_terms["smooth_reward"]` 已经是乘过权重后的值。`step_penalty`、`wall_collision_cost` 和 `timeout_penalty` 在 `info` 里以负值记录。

### 目标推进奖励

```text
d_norm(t) = clip(goal_distance(t) / max(initial_goal_distance, 1e-6),
                 0,
                 goal_distance_clip)

r_target = target_progress_scale * (d_norm(t-1) - d_norm(t))

target_reward = w_target * r_target
```

含义：

- `goal_distance(t)` 是当前鱼体到目标点的距离。
- `initial_goal_distance` 是本 episode 初始目标距离，用来把距离变化归一化。
- 如果当前步比上一控制步更靠近目标，`d_norm(t)` 变小，`r_target > 0`。
- 如果远离目标，`r_target < 0`。
- `goal_distance_clip` 限制归一化距离上界，避免离目标非常远时单步奖励尺度过大。

### 障碍物奖励

障碍物距离先取一个保守值：

```text
obstacle_distance =
  min(min_obstacle_clearance,
      local_obstacle_obs.edge_distance if local obstacle is detected)
```

如果没有有效距离，则按 `inf` 处理，不给避障惩罚。奖励函数允许使用几何真值和视觉检测结果；策略观测仍只包含 RGB 图像和 IMU，不直接拿到这些真值。

```text
if obstacle_distance is not finite or obstacle_distance >= obstacle_safe_distance:
    r_obs = 0
elif obstacle_distance <= obstacle_danger_distance:
    r_obs = -obstacle_collision_penalty
else:
    gap = (obstacle_safe_distance - obstacle_distance)
          / (obstacle_safe_distance - obstacle_danger_distance)
    r_obs = -(gap ** 2)

obstacle_reward = w_obs * r_obs
```

含义：

- 距离大于等于 `obstacle_safe_distance` 时，不惩罚。
- 距离小于等于 `obstacle_danger_distance` 时，直接给最大避障惩罚。
- 两者之间用二次曲线平滑过渡，越靠近障碍，惩罚增长越快。
- 默认 `obstacle_collision_penalty=1.0`，因此危险区边界附近与二次曲线末端尺度一致。

### 朝向奖励

```text
desired_yaw   = atan2(goal_y - fish_y, goal_x - fish_x)
heading_error = wrap_to_pi(desired_yaw - yaw)

r_heading = -heading_error_scale * abs(heading_error) / pi

heading_reward = w_heading * r_heading
```

含义：

- `heading_error` 被限制到 `[-pi, pi]`。
- 完全朝向目标时 `r_heading = 0`。
- 背对目标时 `abs(heading_error) = pi`，`r_heading = -heading_error_scale`。
- 这一项是弱约束，默认权重 `w_heading=2.0`，主要用于减少无意义绕圈，不替代目标推进奖励。

### 动作平滑奖励

奖励函数使用的是动作滤波后的舵机归一化命令：

```text
r_smooth =
  -(smooth_action_l2_scale * current_action^2
    + smooth_action_delta_scale * (current_action - prev_action)^2)

smooth_reward = w_smooth * r_smooth
```

含义：

- `current_action` 是当前控制步滤波后的动作，范围来自环境动作空间。
- `prev_action` 是上一控制步滤波后的动作。
- `current_action^2` 抑制持续的大舵角。
- `(current_action - prev_action)^2` 抑制舵角命令突变，让尾部驱动更平滑。

### 终端和事件奖励

```text
wall_collision_cost term = -wall_collision_cost if wall_collision else 0
timeout_penalty term    = -timeout_penalty if timeout else 0
success term             =  success_reward if reached_goal else 0
```

当前实现中：

- `wall_collision` 来自 `min_wall_clearance <= 0.0`，撞墙会扣 `wall_collision_cost`。
- `timeout` 来自 `elapsed_steps >= max_episode_steps`，超时截断时会扣一次 `timeout_penalty`。
- `collided`、`wall_collision`、`out_of_bounds` 会写入 `info`，但默认不会让 episode 立即终止。
- `reached_goal=True` 的判据是鱼头首次触碰目标区域；达到这一条件后就视为成功。
- 成功后环境还会继续保持 `post_goal_duration_sec=5.0 s`，用于继续录像；这段延迟不会再被 `timeout` 抢先截断。
- 因为奖励项写成 `success_reward if self.reached_goal else 0.0`，所以成功奖励会在触达目标后的每个保持步继续发放，而不是只在触达瞬间发一次。

### 默认奖励参数

| 参数 | 默认值 |
| --- | --- |
| `target_progress_scale` | `1.0` |
| `success_reward` | `60.0` |
| `obstacle_safe_distance` | `0.70` |
| `obstacle_danger_distance` | `0.08` |
| `obstacle_collision_penalty` | `1.0` |
| `heading_error_scale` | `1.0` |
| `smooth_action_l2_scale` | `0.12` |
| `smooth_action_delta_scale` | `0.20` |
| `step_penalty` | `0.01` |
| `wall_collision_cost` | `12.0` |
| `timeout_penalty` | `8.0` |
| `w_target` | `100.0` |
| `w_obs` | `14.0` |
| `w_heading` | `2.0` |
| `w_smooth` | `1.0` |
| `goal_distance_clip` | `1.0` |

参数含义：

| 参数 | 作用 | 调大后的典型效果 |
| --- | --- | --- |
| `target_progress_scale` | 目标距离进度的基础缩放 | 更强调每步靠近目标 |
| `success_reward` | `reached_goal=True` 后的成功奖励 | 更强调触达目标和保持在目标区域 |
| `obstacle_safe_distance` | 避障惩罚开始生效的安全距离 | 策略会更早绕开障碍 |
| `obstacle_danger_distance` | 进入最大避障惩罚的危险距离 | 危险区变宽时，近障动作惩罚更强 |
| `obstacle_collision_penalty` | 危险区内的基础避障惩罚 | 更强惩罚贴近或穿过障碍 |
| `heading_error_scale` | 朝向误差基础缩放 | 更强调鱼头朝向目标 |
| `smooth_action_l2_scale` | 动作幅值惩罚系数 | 更抑制大幅度舵机命令 |
| `smooth_action_delta_scale` | 相邻动作变化惩罚系数 | 更抑制命令突变 |
| `step_penalty` | 每个控制步固定时间成本 | 鼓励更快完成任务 |
| `wall_collision_cost` | 撞墙惩罚 | 更强避免贴墙和越界 |
| `timeout_penalty` | 超时截断时的一次性惩罚 | 更强避免拖延和原地徘徊 |
| `w_target` | 目标推进项权重 | 增强靠近目标的主驱动力 |
| `w_obs` | 避障项权重 | 增强安全距离约束 |
| `w_heading` | 朝向项权重 | 增强航向对准目标 |
| `w_smooth` | 平滑项权重 | 增强动作保守性 |
| `goal_distance_clip` | 归一化目标距离上界 | 限制远距离状态下的奖励尺度 |

注意：当前代码里碰撞、撞墙和出界会写入 `info`，撞墙也会扣 `wall_collision_cost`，但 `_check_terminated()` 只在目标保持完成后返回终止。也就是说，默认实现不是“碰撞即终止”。

## 终止与截断

默认终止逻辑：

- 鱼头第一次碰到目标区域后，立即记为 `reached_goal=True`。
- 继续保持 `post_goal_duration_sec=5.0` 秒，用于继续录像。
- 达到保持时长后，`goal_hold_complete=True`，episode 终止。
- 若尚未成功且连续 `20` 个控制步处于撞墙或撞障碍状态，则判为 `persistent_contact_failure` 并终止。

默认截断逻辑：

```text
max_episode_steps = 5000
```

在默认 `0.04 s` 控制周期下，`5000` 步对应 `200 s` 的环境内部 timeout。若鱼头已经先触碰目标区域，则会优先进入成功后的 5 秒延迟结束流程，不再被 timeout 抢先截断。训练入口另有 `--max-episodes` 回调，默认在完成 `500` 个 episode 后停止训练，但它只统计已经完成的 episode。

仿真时间尺度：

| 参数 | 默认值 |
| --- | --- |
| MuJoCo `timestep` | `0.002 s` |
| `frame_skip` | `20` |
| 控制周期 | `0.04 s` |
| 控制频率 | `25 Hz` |
| `persistent_contact_termination_steps` | `20` |
| `post_goal_duration_sec` | `5.0 s` |
| 触达目标后的保持步数 | `125` 控制步 |

## 训练参数

默认训练参数来自 `configs/default_config.py::TrainConfig`。

| 参数 | 默认值 |
| --- | --- |
| `total_timesteps` | `3_000_000` |
| `num_envs` | `4` |
| `learning_rate` | `3e-4` |
| `n_steps` | `1024` |
| `batch_size` | `256` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_range` | `0.2` |
| `ent_coef` | `0.0` |
| `vf_coef` | `0.5` |
| `max_grad_norm` | `0.5` |
| `policy_hidden_sizes` | `(128, 128)` |
| `seed` | `7` |
| `log_dir` | `runs/ppo_fish_baseline` |
| `model_name` | `ppo_fish_baseline` |
| `checkpoint_interval_timesteps` | `0` |
| `save_policy_weights` | `True` |
| `monitor_filename` | `monitor.csv` |
| `episode_metrics_filename` | `episode_metrics.csv` |
| `save_episode_videos` | `False` |
| `video_interval_episodes` | `5` |
| `video_camera_name` | `top` |
| `video_width` / `video_height` | `480` / `272` |
| `video_fps` | `10` |
| `video_frame_stride` | `4` |

默认训练命令：

```bash
python train.py
```

常用命令：

```bash
python train.py --timesteps 50000
python train.py --num-envs 1
python train.py --device cpu
python train.py --render --render-env-index 0 --render-slowdown 1
python train.py --plot-reward --reward-plot-window 20
python train.py --record-videos
python train.py --no-record-videos
python train.py --video-interval-episodes 10
python train.py --scenario-index 1
python train.py --max-episodes 0
python train.py --resume-from runs/ppo_fish_baseline/ppo_fish_baseline.zip
```

命令行参数说明：

| 参数 | 作用 |
| --- | --- |
| `--timesteps` | 覆盖 `total_timesteps` |
| `--num-envs` | 覆盖并行环境数 |
| `--resume-from` | 从已有 `.zip` PPO 模型继续训练 |
| `--xml-path` | 覆盖 MuJoCo XML 场景路径 |
| `--render` / `--no-render` | 是否打开 MuJoCo 实时 viewer |
| `--render-env-index` | 多环境训练时展示第几个环境 |
| `--render-slowdown` | viewer 慢放倍率，`0` 表示不按实时节奏等待 |
| `--device` | `cuda`、`cuda:0`、`cpu` 或 `auto` |
| `--plot-reward` / `--no-plot-reward` | 是否显示并保存奖励曲线 |
| `--reward-plot-window` | 奖励曲线移动平均窗口 |
| `--record-videos` / `--no-record-videos` | 是否采集 episode 视频帧 |
| `--video-interval-episodes` | 每隔多少个 episode 保存一次视频和 episode checkpoint，设为 `0` 可关闭该回调 |
| `--scenario-path` | 指定固定场景 JSON |
| `--scenario-index` | 使用 `training_env_XX.json` |
| `--scenario-dir` | 固定场景目录 |
| `--max-episodes` | 最多完成多少个 episode 后停止，`0` 或负数表示禁用 |

设备选择逻辑：

- 默认请求 `cuda`。
- 如果当前 PyTorch 没有 CUDA 支持，会自动回退到 `cpu`。
- `--device auto` 会在可用时选 `cuda`，否则选 `cpu`。

## 结果保存与权重落盘

训练开始时会创建：

```text
runs/ppo_fish_baseline/
```

如果指定固定场景，会创建：

```text
runs/ppo_fish_baseline/<scenario_id>/
```

主要输出文件：

| 路径 | 生成时机 | 内容 |
| --- | --- | --- |
| `config.json` | 训练启动时 | 当前完整配置；如果有固定场景或 resume，会额外写入 `selected_scenario_path` / `resume_from` |
| `monitor.csv` | 训练过程中 | Stable-Baselines3 `VecMonitor` 输出 |
| `episode_metrics.csv` | 每个完成的 episode | 自定义 episode 指标 |
| `reward_curve.png` | 使用 `--plot-reward` 时 | step reward 和 episode reward 曲线 |
| `ppo_fish_baseline.zip` | 正常训练结束 | SB3 完整 PPO 模型 |
| `ppo_fish_baseline_policy.pth` | 正常训练结束 | 仅 policy 权重和 `num_timesteps` |
| `ppo_fish_baseline_interrupted.zip` | `Ctrl+C` 中断时 | 中断时 SB3 完整 PPO 模型 |
| `ppo_fish_baseline_interrupted_policy.pth` | `Ctrl+C` 中断时 | 中断时 policy 权重 |

最终模型保存格式：

```text
runs/ppo_fish_baseline/ppo_fish_baseline.zip
runs/ppo_fish_baseline/ppo_fish_baseline_policy.pth
```

中断保存格式：

```text
runs/ppo_fish_baseline/ppo_fish_baseline_interrupted.zip
runs/ppo_fish_baseline/ppo_fish_baseline_interrupted_policy.pth
```

按 timestep 的 checkpoint：

```text
runs/ppo_fish_baseline/checkpoints/ppo_fish_baseline_step_<num_timesteps>.zip
runs/ppo_fish_baseline/checkpoints/ppo_fish_baseline_step_<num_timesteps>_policy.pth
```

但当前默认：

```text
checkpoint_interval_timesteps = 0
```

因此按 timestep 的 checkpoint 默认关闭。只有把该值改成正数才会启用。

按 episode 的 artifact checkpoint 默认开启，因为：

```text
video_interval_episodes = 5
```

每完成 5 个 episode，会保存：

```text
runs/ppo_fish_baseline/checkpoints/ppo_fish_baseline_episode_000005.zip
runs/ppo_fish_baseline/checkpoints/ppo_fish_baseline_episode_000005_policy.pth
```

同时，如果 `save_episode_videos=True` 并且环境采集到了视频帧，会保存：

```text
runs/ppo_fish_baseline/videos/top_view/ppo_fish_baseline_episode_000005.mp4
runs/ppo_fish_baseline/videos/head Cemara/ppo_fish_baseline_episode_000005_head.mp4
```

注意：目录名 `head Cemara` 是当前源码里的实际拼写。

`.zip` 和 `.pth` 的区别：

- `.zip` 是 Stable-Baselines3 的完整 PPO 模型，可用于 `PPO.load()` 或 `eval.py --model-path`。
- `.pth` 只保存一个字典：

```python
{
    "num_timesteps": int(model.num_timesteps),
    "policy_state_dict": model.policy.state_dict()  # 已搬到 CPU
}
```

如果后续只想部署策略网络，可优先看 `.pth`；如果要继续训练或用 SB3 直接评估，用 `.zip`。

## episode_metrics.csv 字段

`episode_metrics.csv` 由 `EpisodeMetricsCallback` 写入，字段包括：

| 字段 | 含义 |
| --- | --- |
| `num_timesteps` | 当前训练总步数 |
| `episode_reward` | SB3 monitor 记录的 episode reward |
| `episode_length` | episode 步数 |
| `episode_time_sec` | episode 用时 |
| `termination_reason` | `goal_reached`、`timeout`、`out_of_bounds` 或 `running` |
| `episode_return` | 环境累计回报 |
| `goal_progress_ratio` | 到目标区的进度比例 |
| `distance_to_goal_region` | 到目标区最近点的距离 |
| `visual_obstacle_detected` | 头部相机是否检测到可视障碍 |
| `visual_obstacle_pixel_fraction` | 障碍物像素占比 |
| `visual_obstacle_center_fraction` | 障碍物在图像中心区域的占比 |
| `visual_obstacle_nearest_depth` | 可视障碍最近深度 |
| `success` | 是否已触达目标 |
| `collision` | 是否与障碍物几何清距小于等于 0 |
| `wall_collision` | 是否与墙体清距小于等于 0 |
| `persistent_contact_steps` | 当前连续碰墙/碰障碍的控制步数 |
| `persistent_contact_failure` | 是否因连续接触达到阈值而失败终止 |
| `out_of_bounds` | 根位置是否越出水池半长/半宽 |
| `timeout` | 是否达到 `max_episode_steps` |

## info 字段

`env.step()` 的 `info` 还会包含控制和诊断字段：

| 字段 | 含义 |
| --- | --- |
| `scenario_id` | 当前固定场景 ID，随机场景时可能为 `None` |
| `raw_action` | PPO 原始动作 |
| `filtered_action` | 低通滤波后的动作 |
| `theta_m_target` | 前舵机目标角 |
| `theta_m` | 前舵机实际命令角 |
| `theta_h` | 映射后的头部角 |
| `head_servo_locked` | 动作处于死区时是否锁定 |
| `tail_freq_target` | 目标尾摆频率 |
| `tail_freq` | 滤波后的尾摆频率 |
| `back_servo_command` | 后舵机命令 |
| `min_obstacle_clearance` | 与最近障碍物的几何清距 |
| `min_wall_clearance` | 与水池墙的最小清距 |
| `target_reward` | 目标推进奖励项 |
| `obstacle_reward` | 避障奖励项 |
| `heading_reward` | 朝向奖励项 |
| `smooth_reward` | 动作平滑奖励项 |
| `step_penalty` | 步进惩罚项 |
| `wall_collision_cost` | 撞墙惩罚项 |
| `timeout_penalty` | 超时惩罚项 |
| `success_reward` | 成功奖励项 |

## 评估

评估命令：

```bash
python eval.py --model-path runs/ppo_fish_baseline/ppo_fish_baseline.zip
```

指定评估回合数：

```bash
python eval.py --model-path runs/ppo_fish_baseline/ppo_fish_baseline.zip --episodes 10
```

开启 Matplotlib 顶视图渲染：

```bash
python eval.py --model-path runs/ppo_fish_baseline/ppo_fish_baseline.zip --render
```

评估会逐 episode 打印：

- reward
- goal_progress
- collision
- success

最后打印：

- Average reward
- Average goal completion
- Collision rate
- Success rate

## 依赖安装

pip：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 当前内容：

```text
gymnasium>=1.0
stable-baselines3>=2.4
mujoco==3.3.7
numpy>=1.26
matplotlib>=3.8
imageio>=2.37
imageio-ffmpeg>=0.6
```

服务器环境可参考：

```bash
conda env create -f environment.server.yml
```

该环境使用 Python 3.9、CUDA 12.4 对应的 PyTorch、MuJoCo 3.3.7，并设置：

```text
MUJOCO_GL=egl
```

## 当前实现注意点

- 当前策略输入只有头部相机 RGB 和 IMU。
- 障碍物、目标距离、墙体清距等真值会用于奖励、终止标志和日志，但不会直接输入策略。
- `underwater_effect_enabled` 默认关闭，所以默认不启用深度/segmentation 驱动的水下退化和可视障碍检测。
- `max_episode_steps` 默认是 `5000`，对应每个 episode 最长 `200 s`。
- 当前 `_check_terminated()` 只在目标保持完成时终止，不会因为碰撞、撞墙或出界直接终止。
- timestep checkpoint 默认关闭；episode checkpoint 默认每 5 个完成的 episode 保存一次。
