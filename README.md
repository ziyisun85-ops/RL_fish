# RL Fish PPO Baseline

基于 MuJoCo、Gymnasium 和 Stable-Baselines3 PPO 的机器鱼强化学习项目。

当前版本的任务不是路径跟踪，而是：

- 机器鱼从水池左侧随机出生
- 目标区域固定在水池右侧
- 水池中间会放置一对立柱障碍物
- 策略需要依靠头部相机图像和 IMU 完成到达与避障

## 当前任务定义

- 出生区域：水池左侧随机区域
- 终点区域：水池右侧固定矩形区域
- 障碍物：默认固定为 2 个圆柱立柱，成对出现在出生区与终点区连线两侧
- 单回合最大时长：60 秒
- 成功条件：机器鱼任一碰撞体接触终点区域
- 失败条件：碰撞障碍物、出界或超时

当前默认设计目标是让机器鱼必须绕开中间障碍，而不是从中间直接穿过去。

## 目录结构

```text
RL_fish/
├─ configs/
│  └─ default_config.py
├─ envs/
│  ├─ __init__.py
│  └─ fish_env.py
├─ model/
│  ├─ fish_pool_scene.xml
│  ├─ fish_robot_model.xml
│  ├─ pool_environment.xml
│  ├─ fish_2d_stl.xml
│  ├─ rebuild_fish_from_stl.py
│  └─ split_fish_pool_mjcf.py
├─ utils/
│  ├─ geometry.py
│  ├─ mappings.py
│  └─ obstacles.py
├─ runs/
├─ eval.py
├─ requirements.txt
└─ train.py
```

## 核心架构

运行时主链路如下：

```text
MuJoCo 场景
-> FishPathAvoidEnv
-> 头部相机图像 + IMU
-> PPO(MultiInputPolicy)
-> 1 维连续动作
-> 头部舵机目标角
-> 头角映射尾频
-> 尾舵机周期摆动
-> MuJoCo 推进
-> reward / terminated / info
```

主要文件职责：

- `configs/default_config.py`
  统一维护环境、相机、舵机、奖励、训练和评估参数
- `envs/fish_env.py`
  Gymnasium 环境封装，负责 reset、step、相机观测、奖励、终止判断和 MuJoCo viewer
- `utils/mappings.py`
  头部舵机角到头角、头角到尾摆频率的映射
- `utils/obstacles.py`
  障碍物采样逻辑
- `train.py`
  PPO 训练入口
- `eval.py`
  评估入口

## 观测、动作与控制链

### 观测

策略当前接收的是字典观测：

- `image`
  头部相机 RGB 图像，默认 `84 x 84 x 3`
- `imu`
  3 维向量 `[ax_body, ay_body, yaw_rate]`

说明：

- 相机图像带有简化的水下成像退化
- 图像不是理想 RGB，相机可视距离会随深度连续衰减
- 当前策略看不到环境内部的障碍物真值坐标

### 动作

- PPO 输出 1 维连续动作 `a in [-1, 1]`
- 该动作不是直接推力，而是头部舵机高层命令

### 控制链

当前控制链是：

1. PPO 输出动作 `a`
2. 动作经轻微滤波后变成头舵机目标角 `theta_m_target`
3. 头舵机按有限速度和有限加速度追踪目标角
4. 根据论文形式近似计算头部偏角 `theta_h`
5. 根据 `theta_h` 调节尾摆频率
6. 尾部按单振荡器正弦摆动，后舵机同样带速度和加速度约束

### 头部与尾部映射

当前实现采用论文形式的近似映射：

- `theta_h = arcsin((r * theta_m) / (eta * l))`
- `f_tail = clip(f0 * (1 + k * sign(theta_h) * |theta_h|), f_min, f_max)`

对应代码：

- `utils/mappings.py`

说明：

- 这部分比最初的线性占位实现更接近论文
- 但尾部目前仍是单振荡器正弦摆动，不是完整多节点 CPG 网络

## 相机与视觉逻辑

### 头部相机

头部挂载一个 MuJoCo 相机：

- 位置固定在鱼头上
- 当前策略只看头部相机图像，不看顶视图

### 水下视觉

头部相机图像带有简化的水下退化效果：

- 距离越远，图像透射率越低
- 图像会混入水体颜色
- 远处会出现轻微模糊和噪声

这部分是为了避免“理想空气中相机”造成过于不真实的视觉输入。

### 障碍物视觉判定

当前项目中，“看见障碍物就要避开”的逻辑已经接入奖励：

- 使用 MuJoCo 相机分割图和深度图
- 只有当障碍物真的出现在头部相机视野里，并且在有效可视距离内时，才触发视觉避障奖励

策略本身仍然只接收 RGB 图像和 IMU，不接收分割图。

## 奖励函数

当前奖励由 5 部分组成：

- `goal_progress`
  鼓励朝终点区域前进
- `visible_obstacle_depth`
  相机可见障碍物如果变远，会得到正反馈
- `visible_obstacle_center_penalty`
  如果障碍物长期挡在视野中央，会被惩罚
- `collision_penalty`
  撞障碍物时强惩罚
- `success_bonus`
  接触终点区域时奖励

说明：

- 避障奖励现在主要由“视觉可见障碍物”触发
- 但终点前进奖励仍然使用环境内部几何距离计算
- 障碍物真正碰撞终止目前仍然主要依赖几何 clearance 判定

这意味着当前实现是：

- 策略输入：相机 + IMU
- 奖励与终止：部分基于环境内部真值

## 障碍物规则

当前障碍物不是全池随机撒点，而是受控采样：

- 默认障碍物数量固定为 2
- 两个障碍物位于出生区和终点区连线两侧
- 它们共享接近的纵向位置
- 中间间隙被压缩，默认不适合直接直穿
- 障碍物每 `1000` 个 episode 才重新采样一次

可视化上，障碍物在 MuJoCo viewer 中显示为红色立柱。

## 出生与终点

### 出生

- 机器鱼在左侧区域随机出生
- 初始朝向大致面向终点
- reset 时会检查出生姿态，避免一出生就贴墙或碰障碍物

### 终点

- 终点不是路径上的一个点，而是右侧矩形区域
- MuJoCo 场景里会显示绿色终点区域
- 成功判定不是“根节点进入区域”，而是“机器鱼任一碰撞体接触终点区域”

## 仿真与物理设置

### 时间尺度

- MuJoCo 基础步长：`0.002 s`
- `frame_skip = 20`
- 控制频率：`25 Hz`
- 单回合最大步数：`1500`
- 单回合最大仿真时长：`60 s`

### 水体参数

当前全局流体参数已下调到较低阻力版本：

- `density = 850`
- `viscosity = 0.0007`

### 舵机逻辑

头部和尾部舵机都不是瞬时打角，而是：

- 有死区
- 有最大角速度
- 有加速度限制
- MuJoCo actuator 仍使用位置伺服

说明：

- 这里的 `kp` 来自 MuJoCo `<position>` actuator
- 不是 Python 里手写的 PID
- 更准确地说，是高层策略加低层位置伺服的组合

## 训练

### 安装依赖

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 当前内容：

- `gymnasium>=1.0`
- `stable-baselines3>=2.4`
- `mujoco==3.3.7`
- `numpy>=1.26`
- `matplotlib>=3.8`

### 默认训练命令

```bash
python train.py
```

当前默认行为：

- 默认开启 MuJoCo viewer
- 默认 `render_slowdown = 1.0`
- 默认设备优先使用 `cuda`
- 如果当前 PyTorch 没有 CUDA 支持，会自动回退到 CPU

常用命令示例：

```bash
python train.py --no-render
python train.py --num-envs 1
python train.py --timesteps 50000
python train.py --device cpu
python train.py --render-env-index 0 --render-slowdown 2
```

### PPO 配置

默认训练参数位于 `configs/default_config.py`：

- 算法：PPO
- 策略：`MultiInputPolicy`
- 输入：`image + imu`
- `total_timesteps = 300000`
- `num_envs = 4`
- `n_steps = 1024`
- `batch_size = 256`

## 评估

评估命令：

```bash
python eval.py --model-path runs/ppo_fish_baseline/ppo_fish_baseline.zip
```

带渲染评估：

```bash
python eval.py --model-path runs/ppo_fish_baseline/ppo_fish_baseline.zip --render
```

说明：

- `train.py --render` 打开的是 MuJoCo 实时 viewer
- `eval.py --render` 当前显示的是环境内部的顶视图轨迹渲染

评估输出会打印每个 episode 的：

- reward
- goal_progress
- collision
- success

## 模型与权重落盘

训练过程中现在会自动落盘：

### 正常训练结束

```text
runs/ppo_fish_baseline/ppo_fish_baseline.zip
runs/ppo_fish_baseline/ppo_fish_baseline_policy.pth
```

### 中途 checkpoint

默认每 `20000` timesteps 保存一次：

```text
runs/ppo_fish_baseline/checkpoints/ppo_fish_baseline_step_20000.zip
runs/ppo_fish_baseline/checkpoints/ppo_fish_baseline_step_20000_policy.pth
```

### 中断保存

如果训练时按 `Ctrl+C`，会额外保存：

```text
runs/ppo_fish_baseline/ppo_fish_baseline_interrupted.zip
runs/ppo_fish_baseline/ppo_fish_baseline_interrupted_policy.pth
```

说明：

- `.zip` 是 Stable-Baselines3 完整模型
- `.pth` 是单独的 policy 权重

## 当前实现的已知简化

1. 尾部还是单振荡器正弦驱动，不是完整多节 CPG 网络
2. 终点前进奖励仍然用了环境内部几何距离，不是纯视觉闭环
3. 障碍物实际碰撞终止主要还是几何 clearance 判定，不是纯 MuJoCo contact
4. 视觉避障奖励使用了 MuJoCo 分割和深度图作为内部裁判信号，但策略本身只接收 RGB 图像和 IMU
5. 水下视觉是工程近似，不是完整水下成像物理模型

## 后续建议

如果继续往真实机器鱼方向推进，建议优先做：

1. 将尾部单振荡器扩展成真正的多节点 CPG
2. 让终点识别更依赖视觉，而不是几何真值
3. 把障碍物碰撞从几何 clearance 进一步收敛到稳定的 MuJoCo 实体接触
4. 标定更接近真实舵机的数据，而不是仅靠速度和加速度近似
