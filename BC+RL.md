你是一个熟悉 Python、PyTorch、MuJoCo、PPO、机器人强化学习、模仿学习、AHTA-CPG 控制、实验消融设计和工程化代码改造的高级工程师。  
请基于我**现有的机器鱼强化学习项目代码**做**增量式修改**，不要推翻重写整个工程，不要主观假设我的项目是标准 PPO 模板，也不要假设我使用的是 stable-baselines3、cleanrl 或其他通用框架，必须先阅读我当前代码结构后再实施修改。

---

## 一、项目背景与目标

我目前已经有一版可以正常训练的 PPO 强化学习代码，环境是机器鱼控制任务。当前控制逻辑中：

- AHTA-CPG 已经能够**自主产生固定频率节律推进**；
- 尾部节律 / body wave **不需要人工控制**；
- 人工示教和策略网络都只控制**头部动作**；
- 动作应为**单维连续动作**：`head_action`，例如头部偏角、舵机转角，或者 PPO 动作空间中的头部控制量；
- 控制链路应始终保持为：

```text
obs -> policy -> head_action -> AHTA-CPG -> rhythmic propulsion -> robot motion
````

我要实现的总目标是：

1. **人工键盘示教采集专家数据**；
2. **使用专家数据做 Behavior Cloning (BC) 预训练**；
3. **把 BC 训练出的 actor 权重加载到 PPO 中继续强化学习训练**；
4. **增加系统化评估与消融实验脚本**，用于严格证明：

   * BC 本身有效；
   * BC 对 RL 有效；
   * RL 微调有效。

---

## 二、最重要的硬约束

请严格遵守以下硬约束：

1. **不要重写整个项目**，优先以“新增脚本 + 最小修改现有训练入口”的方式实现；
2. **不要改变 AHTA-CPG 的底层节律推进逻辑**；
3. **人工与策略网络都只允许控制头部动作**；
4. **不要把尾部节律、尾部频率、body wave 控制重新交给人工或 actor**；
5. **不要破坏当前 PPO 单独训练流程**；
6. **不要改变现有仿真环境的物理逻辑与接口语义**；
7. **不要只给伪代码**，尽量给出完整可运行代码；
8. **关键改动必须适配我真实项目结构**，而不是套一个通用模板；
9. **保留原有日志、权重保存、评估流程，除非必须修改**；
10. **所有路径、参数、开关优先通过 argparse 或现有配置系统管理**；
11. **如果当前 obs 含图像，必须兼容图像 obs，不允许为了实现 BC 而删掉图像输入**；
12. **如果当前 action 有 tanh / clip / scaling / physical mapping，BC 和 eval 必须严格沿用同一套动作定义**；
13. **如果当前项目用到了 obs normalization / running mean std，BC、eval、PPO 微调必须复用同一套规范**；
14. **不要偷懒跳过项目结构分析**，必须先分析再改。

---

## 三、你开始修改前必须先完成的代码结构分析

在给出改动方案之前，你必须先阅读我当前代码，并输出一份**真实代码结构分析**。
不要凭经验假设，必须基于我当前项目中的真实实现回答。

请明确分析以下内容：

### 1. 环境接口

* `env.reset()` 返回什么格式；
* `env.step(action)` 接收什么格式的 `action`；
* `action` 是标量、向量、dict 还是其他结构；
* 当前 action 中哪一维或哪个字段对应头部控制；
* 当前环境是否已经把 AHTA-CPG 与头部控制解耦；
* `info` 中有哪些可用指标，比如：

  * `success`
  * `collision`
  * `tracking_error`
  * `distance_to_goal`
  * `heading_error`
  * `disturbance_level`

### 2. 观测空间

* obs 是 `numpy`、`torch.Tensor`、`dict`、`tuple` 还是混合结构；
* 是否包含图像观测；
* 图像和状态量是如何拼接或分别送入网络的；
* 是否有 frame stack、RNN、history buffer 等机制；
* 现有 obs normalization / preprocessing 在哪里实现。

### 3. Actor / Critic 结构

* actor 的输入输出维度；
* actor 是否输出 deterministic action、Gaussian mean/std、distribution parameters；
* actor 输出是否经过 `tanh`；
* actor 输出如何映射到 env 实际动作；
* critic 结构与 actor 是否共享 backbone；
* 当前模型权重如何保存 / 加载。

### 4. PPO 训练流程

* PPO 训练入口文件和主函数；
* rollout 收集逻辑；
* PPO loss 实现位置；
* entropy loss / value loss / clip loss 的实现方式；
* 当前日志记录位置；
* 当前 checkpoint 保存方式；
* 当前评估逻辑；
* 是否支持 resume。

### 5. 工程化约束

* 哪些文件最适合新增；
* 哪些文件必须修改；
* 哪些现有接口不应改动；
* 是否已有 replay buffer / trajectory buffer / dataset 类可复用。

这一部分必须以“真实项目分析结果”形式输出，而不是泛泛而谈。

---

## 四、总体实现任务

在完成真实代码结构分析后，请你在**尽量少改动现有工程**的前提下，实现以下四部分功能：

1. **人工键盘示教数据采集**
2. **BC 模仿学习预训练**
3. **BC actor 加载到 PPO 继续训练**
4. **实验评估与消融分析**

---

## 五、人工键盘示教数据采集模块

请新增一个人工示教脚本，例如：

```text
collect_demo.py
```

### 功能目标

允许我在环境运行时通过键盘实时控制**头部动作**，完成任务并记录专家数据，用于后续 BC 训练。

### 控制要求

人工只控制一个动作：

```text
head_action
```

例如：

```text
头部偏角 / 舵机角度 / PPO 动作空间中的头部控制量
```

AHTA-CPG 继续自动产生尾部节律推进，人工不得直接控制尾部频率、尾部幅值、body wave 参数。

### 推荐键盘映射

至少支持：

```text
A：头部向左偏
D：头部向右偏
SPACE：头部回中
Q：结束当前 episode
R：重置环境
ESC：退出采集程序
```

可选增强：

```text
W：增大头部偏角步长
S：减小头部偏角步长
P：暂停/继续
```

### 动作一致性要求

这是最关键的地方，必须严格处理：

1. 如果 actor 输出的是 normalized action（例如 `[-1, 1]`），人工输入也必须落在**同一个 normalized space**；
2. 如果环境真正执行的是 physical action（例如弧度 / 角度 / 舵机命令），必须明确给出：

   * normalized action -> physical action 的映射
   * physical action -> normalized action 的逆映射（如果需要）
3. 不允许 BC 训练时用一个动作定义，PPO 训练时用另一个动作定义；
4. 不允许因为动作空间不一致导致 BC 权重加载到 PPO 后失效。

### 每一步需要保存的数据

每个 environment step 至少保存：

```text
obs
action
next_obs
reward
done
episode_id
timestep
```

如果环境里已有以下信息，也请一并保存：

```text
success
collision
tracking_error
distance_to_goal
heading_error
flow_disturbance
```

### 保存格式要求

请保存为 `.pt` 或 `.npz`，例如：

```text
demos/head_demo.pt
```

要求：

* 数据可直接用于 BC 训练；
* 若 obs 是 dict，必须保留 dict 结构；
* 若 obs 含图像，图像数据必须完整保存；
* 保证读取时不需要我手工再次拼接或对齐。

### 数据组织建议

请给出合理实现，支持以下任一种或两种：

1. **按 step 统一存储为一个大文件**
2. **按 episode 分文件，再提供索引文件**

优先选择后续 BC 训练最方便的一种。

### 质量控制建议

请额外实现以下可选功能，若容易实现则一并加入：

* 记录每个 episode 是否成功；
* 支持只保留成功 episode；
* 或至少支持在 BC 训练时可选过滤失败 episode；
* 记录每条 demo 的长度、累计奖励、是否碰撞；
* 打印采集统计摘要。

---

## 六、Behavior Cloning 模仿学习预训练模块

请新增一个 BC 训练脚本，例如：

```text
train_bc.py
```

### 功能目标

利用人工示教数据，对当前 PPO 使用的 actor 网络进行监督学习预训练，让策略先学会“基本合理的头部控制”。

### 核心要求

1. **必须复用当前 PPO 的 actor 结构**

   * 不允许单独定义一套 incompatible actor
   * 训练好的权重必须能直接加载到 PPO actor
2. **输入必须与 PPO 一致**

   * obs 结构一致
   * 图像处理一致
   * normalization 一致
3. **输出必须与 PPO actor 的 head_action 定义一致**

### BC 损失函数

如果 actor 直接输出连续动作：

```python
bc_loss = ((pred_action - expert_action) ** 2).mean()
```

如果当前 actor 输出的是高斯分布参数或策略分布，请基于真实代码结构做最合适的 BC 方案：

优先级如下：

1. 若可以方便取 mean action，则优先用 mean 做 MSE；
2. 若当前结构更适合做 log-prob imitation，也可使用 expert action 的负对数似然；
3. 但必须说明你最终采用哪种方式，以及为什么它最贴合我现有代码结构。

### 数据加载要求

请实现：

* `Dataset`
* `DataLoader`
* `train / val split`
* batch 训练
* shuffle
* 支持大数据量时分批读取

### 日志与 checkpoint

请输出以下训练日志：

```text
epoch
train_loss
val_loss
learning_rate
best_val_loss
```

并保存最优模型为：

```text
checkpoints/bc_actor.pt
```

### normalization 一致性要求

如果 PPO 使用 obs normalization / running mean std / image preprocessing：

1. BC 必须完全复用相同逻辑；
2. 必须保存对应 normalization 状态；
3. 后续 PPO 加载 BC actor 时必须能恢复同一套 normalization；
4. eval 时也必须使用同一套 normalization；
5. 不允许 BC、PPO、eval 各自使用不同的输入预处理。

### BC 输出要求

请明确输出：

* 最终保存了哪些文件；
* 哪个文件给 PPO actor 加载；
* 哪个文件保存 normalization 状态；
* 如果有 optimizer state，是否保存。

---

## 七、PPO 加载 BC 权重继续训练

请修改我现有的 PPO 训练入口，例如：

```text
train_ppo.py
```

新增参数：

```bash
--bc_pretrain_path
```

### 功能逻辑

* 如果提供 `--bc_pretrain_path`：

  * 加载 `bc_actor.pt`
  * 用它初始化 PPO 的 actor
* 如果不提供：

  * 保持原有 PPO from scratch 流程不变

### 注意事项

1. critic 不强制加载 BC 权重；
2. 若 actor/critic 共享 backbone，必须谨慎处理加载逻辑；
3. 若只加载 actor 相关权重，请明确写出过滤规则；
4. 不允许因为加载 BC 而破坏原有 PPO 训练；
5. 不允许因为 state_dict key 不一致导致报错后草草跳过，必须处理干净。

---

## 八、PPO 中加入可选 BC auxiliary loss

请在 PPO 中加入一个**可选**的 BC 辅助损失，用于训练前期防止 PPO 快速破坏 BC 学到的基本行为。

新增参数：

```bash
--use_bc_loss
--bc_data_path
--bc_coef_start
--bc_coef_end
--bc_decay_steps
```

### 总损失形式

```python
loss = ppo_loss + bc_coef * bc_loss
```

其中：

* `bc_coef` 在训练过程中从 `bc_coef_start` 逐步衰减到 `bc_coef_end`；
* 衰减方式默认使用**线性衰减**；
* 需要记录 `bc_coef` 和 `bc_loss` 日志。

### 实现要求

1. 不启用 `--use_bc_loss` 时，PPO 行为必须与原始代码完全一致；
2. 启用时，从 demo dataset 中额外采样 batch；
3. 使用当前 actor 对 expert obs 做前向；
4. 计算 BC loss；
5. 与 PPO loss 合并；
6. 记录日志；
7. 不要让这部分实现影响 rollout buffer 的原有逻辑。

### 工程要求

* 尽量不要把 PPO 代码改得很乱；
* 把 BC auxiliary loss 封装成清晰的函数或模块；
* 让开关逻辑清楚可控。

---

## 九、实验评估与消融设计

请新增或修改评估脚本，例如：

```text
eval.py
evaluate_ablation.py
plot_results.py
```

要求支持以下三类关键实验。

---

## 十、实验 1：证明 BC 本身有效

### 对比组

```text
Random Actor
BC only
```

### Random Actor 的严格定义

Random Actor 指：

* 创建一个与 PPO/BC **完全相同结构**的 actor 网络；
* 仅做默认随机初始化；
* **不加载 BC 权重**；
* **不进行 PPO 训练**；
* **直接评估**。

注意：

* Random Actor **不是** `env.action_space.sample()`
* Random Actor **不是** PPO from scratch
* Random Actor **不经过 PPO**

### BC only 的定义

BC only 指：

* 创建同样结构的 actor；
* 加载 `bc_actor.pt`；
* 不做 PPO；
* 直接评估。

### 实验目的

证明：

> BC 训练是否让策略从“随机输出头部动作”变成“具备基本合理头部控制能力”。

### 输出指标

至少输出：

```text
success_rate
average_reward
collision_rate
tracking_error
completion_time
average_heading_error
```

如果环境已有更多指标，也尽量加入。

---

## 十一、实验 2：证明 BC 对 RL 有效

### 对比组

```text
PPO from scratch
BC + PPO
```

### PPO from scratch 定义

* actor 随机初始化；
* 不加载 BC；
* 直接 PPO 训练。

### BC + PPO 定义

* 先加载 `bc_actor.pt` 初始化 actor；
* 再进行 PPO 训练。

### 控制变量要求

这两组必须保持完全一致：

* 环境一致；
* 奖励函数一致；
* 网络结构一致；
* 总训练步数一致；
* rollout 长度一致；
* PPO 超参数一致；
* 随机种子尽量多组；
* 评估任务一致；
* normalization 一致；
* 日志统计方式一致。

### 实验目的

证明：

> BC 初始化是否提升 PPO 的收敛速度、样本效率、训练稳定性和最终成功率。

### 输出内容

至少输出：

```text
success_rate vs training_steps
average_reward vs training_steps
collision_rate vs training_steps
tracking_error vs training_steps
steps_to_60_percent_success
steps_to_80_percent_success
final_success_rate
final_average_reward
```

---

## 十二、实验 3：证明 RL 微调有效

### 对比组

```text
BC only
BC + PPO
```

### 定义

* `BC only`：只加载 BC 权重，直接测试；
* `BC + PPO`：加载同一份 BC 权重后再 PPO 微调，训练后测试。

### 实验目的

证明：

> RL 是否在 BC 的基础上进一步提升复杂任务、障碍环境和扰流场景下的性能。

### 重点测试场景

请支持至少以下场景：

```text
简单路径跟踪
单障碍绕行
多障碍绕行
横向扰流
随机目标点
未见过的障碍分布
```

如果我当前环境已支持更多场景，请尽量复用。

### 输出指标

至少输出：

```text
success_rate
average_reward
collision_rate
tracking_error
completion_time
generalization_results
disturbance_robustness_results
```

---

## 十三、最终实验分组总表

请确保至少支持以下五组实验：

```text
Group 1: Random Actor
Group 2: BC only
Group 3: PPO from scratch
Group 4: BC + PPO
Group 5: BC + PPO + BC auxiliary loss
```

各组严格定义如下：

### Group 1: Random Actor

* 不训练；
* 不加载 BC；
* 不跑 PPO；
* 直接评估。

### Group 2: BC only

* 只加载 BC 权重；
* 不跑 PPO；
* 直接评估。

### Group 3: PPO from scratch

* 随机初始化 actor；
* 正常 PPO 训练。

### Group 4: BC + PPO

* 加载 BC 权重；
* 然后 PPO 训练。

### Group 5: BC + PPO + BC auxiliary loss

* 加载 BC 权重；
* PPO 训练时启用 BC auxiliary loss。

---

## 十四、命令行接口要求

请尽量实现或适配以下命令风格。

### 1. 采集人工示教数据

```bash
python collect_demo.py \
  --save_path demos/head_demo.pt \
  --num_episodes 100
```

### 2. 训练 BC

```bash
python train_bc.py \
  --demo_path demos/head_demo.pt \
  --save_path checkpoints/bc_actor.pt \
  --epochs 100 \
  --batch_size 256
```

### 3. 评估 Random Actor

```bash
python eval.py \
  --mode random_actor \
  --num_episodes 100
```

### 4. 评估 BC only

```bash
python eval.py \
  --mode bc \
  --bc_path checkpoints/bc_actor.pt \
  --num_episodes 100
```

### 5. PPO from scratch

```bash
python train_ppo.py
```

### 6. BC + PPO

```bash
python train_ppo.py \
  --bc_pretrain_path checkpoints/bc_actor.pt
```

### 7. BC + PPO + BC auxiliary loss

```bash
python train_ppo.py \
  --bc_pretrain_path checkpoints/bc_actor.pt \
  --use_bc_loss \
  --bc_data_path demos/head_demo.pt \
  --bc_coef_start 1.0 \
  --bc_coef_end 0.05 \
  --bc_decay_steps 200000
```

### 8. 绘制实验曲线

```bash
python plot_results.py --log_dir logs/exp_name
```

如果我现有项目已有统一入口，请以最少改动方式适配，而不是强行重构。

---

## 十五、日志与绘图要求

请保存实验日志，至少支持 CSV 或 TensorBoard 中的一种，最好两种都支持。

至少记录以下字段：

```text
training_step
episode
success_rate
average_reward
collision_rate
tracking_error
completion_time
ppo_loss
value_loss
entropy_loss
bc_loss
bc_coef
learning_rate
```

请提供绘图脚本，绘制：

```text
success_rate_curve
reward_curve
collision_rate_curve
tracking_error_curve
```

并计算：

```text
steps_to_60_percent_success
steps_to_80_percent_success
final_success_rate
final_average_reward
```

---

## 十六、额外鲁棒性要求

请在实现中考虑以下工程细节并处理好：

1. **图像 obs**

   * 保存和读取时不要破坏维度顺序；
   * 注意内存占用；
   * 若项目已有图像 encoder，BC 必须复用。

2. **dict obs**

   * Dataset 和 actor 前向时必须兼容 dict 结构；
   * 不要强行扁平化后把 key 信息丢掉。

3. **RNN / history**

   * 如果当前 actor 用了 RNN、GRU、LSTM 或历史帧堆叠，请基于真实结构适配；
   * 不要假设是纯 MLP。

4. **动作裁剪**

   * eval、BC、PPO 都要使用相同的动作裁剪与映射逻辑。

5. **normalization 状态**

   * 训练、评估、继续训练时必须一致；
   * 不要忘记保存 / 恢复 running mean/std。

6. **随机种子**

   * 尽量支持多 seed 运行，便于做更可信的消融。

7. **checkpoint 兼容**

   * 加载 BC actor 时，要兼容设备切换（cpu / cuda）；
   * 处理 state_dict key 前缀差异；
   * 明确报错信息，不要静默失败。

8. **示教质量过滤**

   * 如果实现难度不大，增加：

     * 仅训练成功轨迹
     * 或设置最小 reward 阈值
     * 或按 success 标签筛选

---

## 十七、输出格式要求

请严格按以下结构输出你的结果。

### 1. 当前代码结构分析

基于真实代码说明：

```text
env action 格式
obs 格式
actor 输入输出
critic 结构
PPO 训练入口
PPO loss 位置
模型保存与加载方式
normalization 方式
```

### 2. 修改方案总览

说明：

```text
新增哪些文件
修改哪些文件
每个文件负责什么功能
为什么这样改动最小
```

### 3. 新增/修改后的完整关键代码

要求：

* 给出完整可运行代码；
* 不要只给伪代码；
* 不要只给零散片段；
* 对关键函数和关键接口加注释。

### 4. 运行命令

给出从头到尾的完整流程命令：

```text
采集示教数据
训练 BC
评估 Random Actor
评估 BC only
训练 PPO from scratch
训练 BC + PPO
训练 BC + PPO + BC auxiliary loss
绘图
```

### 5. 实验指标说明

明确说明如何证明：

```text
BC 本身有效：
Random Actor vs BC only

BC 对 RL 有效：
PPO from scratch vs BC + PPO

RL 微调有效：
BC only vs BC + PPO
```

### 6. 注意事项与潜在坑点

重点提醒：

```text
head_action 范围一致性
obs normalization 一致性
图像 obs 保存与读取
Random Actor 不等于 PPO from scratch
BC only 不等于 BC + PPO
PPO from scratch 和 BC + PPO 的控制变量必须一致
BC auxiliary loss 不能一直过大，否则会妨碍 RL 继续优化
```

---

## 十八、最后的执行要求

请务必遵循以下执行顺序：

1. **先阅读我现有项目代码；**
2. **先输出真实结构分析；**
3. **再给出改造方案；**
4. **最后给出完整关键代码。**

不要跳过“代码结构分析”这一步。
不要直接套一个通用 PPO + BC 模板。
不要把我的 AHTA-CPG 逻辑改掉。
不要把头部控制之外的维度重新交给策略网络。
不要只给思路，不给代码。
不要假装工程适配完成，必须真正贴合我现有代码结构。

```

你要是想，我下一条还能直接给你一版**更适合发给 Codex 的“强约束工程版 prompt”**，会更像代码审查任务单。
```
