# 任务：为 MuJoCo 仿生机器鱼添加基于物理依据的分布式水动力模型（支持真实转弯）

你是一个精通：
- 水下机器人动力学
- 仿生鱼推进理论（Resistive Force Theory / Lighthill Elongated Body Theory）
- MuJoCo 仿真系统
- 工程代码安全修改

的高级工程师。

---

# 🎯 任务目标

在现有 MuJoCo 机器鱼项目中：

👉 添加**物理合理的分布式水动力模型**

使系统能够：
- 产生真实偏航（yaw）
- 实现转弯（非直线摆动）
- 保持与 body wave（AHTA-CPG）耦合

---

# ⚠️ 强约束

❌ 不允许修改：
- 强化学习（PPO）
- observation（图像输入）
- action space
- 渲染 / 视频 / 权重保存

✅ 只能：
- 新增模块
- 在 step() 中增加一行调用

---

# 🧠 Step 1：先分析代码结构（禁止直接写代码）

请先分析：
- 环境类（env / simulation）
- step() 函数位置
- body 索引方式
- XML body 命名

输出结构分析后再继续。

---

# 🧠 Step 2：物理建模原则（必须遵守）

本模型基于：

## 1️⃣ Resistive Force Theory（阻力理论）
核心思想：
- 流体力主要来自局部速度
- 分为：
  - 切向阻力（小）
  - 法向阻力（主导）

依据：
- Gray & Hancock (1955)
- Lighthill (1971)

---

## 2️⃣ Elongated Body Theory（细长体理论）

核心思想：
- 推进和转弯来自 body wave
- 力来自局部横向运动

---

## 3️⃣ 关键物理结论

必须体现：

### ✔ 法向力 >> 切向力
### ✔ 分段受力（不能整体 drag）
### ✔ 力臂 → 偏航力矩
### ✔ 体波 → 局部速度 → 水动力

---

# 🔧 Step 3：必须实现的水动力组成

总水动力：

F_total = F_resistive + F_added_mass + F_damping

---

## 3.1 分段阻力（核心）

对每个 body i：

### 相对速度：

v_rel = v_i - v_water

---

### 分解：

v_t = dot(v_rel, t_i)
v_n = dot(v_rel, n_i)

---

### 阻力模型：

F_t = -C_t * v_t * t_i

F_n = -C_n * |v_n| * v_n * n_i

---

## 📌 重要说明：

- 法向阻力是推进和转弯的主要来源
- 切向阻力主要是能量损耗

---

## 3.2 偏航阻尼（必须）

M_yaw = -D_r * r - D_r2 * |r| r

作用：
- 防止不稳定旋转
- 模拟流体阻尼

---

## 3.3 附加质量（Added Mass）

根据流体动力学：

实际惯性 = 本体 + 附加质量

实现：

(m + m_ax) * du/dt  
(m + m_ay) * dv/dt  
(I + J_ar) * dr/dt  

---

## 📌 简化建议：

只实现：
- m_ay（横向）
- J_ar（偏航）

---

# 🔧 Step 4：参数推荐（必须使用）

## 阻力系数（经验值）

| 参数 | 推荐范围 |
|------|--------|
| C_t | 0.05 ~ 0.2 |
| C_n | 1.0 ~ 5.0 |

👉 初始建议：
C_t = 0.1  
C_n = 2.0  

---

## 偏航阻尼

| 参数 | 推荐范围 |
|------|--------|
| D_r | 0.01 ~ 0.1 |
| D_r2 | 0.001 ~ 0.05 |

---

## 附加质量

| 参数 | 推荐 |
|------|------|
| m_ay | 0.5 × 机器人质量 |
| J_ar | 0.2 × 原始惯量 |

---

## 分段调整（非常关键）

C_n_head = 1.5 × C_n  
C_n_tail = 1.2 × C_n  

---

# 🔧 Step 5：代码实现要求

必须：

## ✔ 新建：

hydrodynamics.py

---

## ✔ 实现：

apply_hydrodynamics(model, data)

---

## ✔ 对每个 body：

- 获取位置、速度、姿态
- 计算局部坐标系
- 分解速度
- 计算 F_t / F_n
- 调用 mj_applyFT

---

## ✔ 计算总偏航力矩：

M_z = Σ (r_i × F_i)

---

## ✔ 加入 step：

apply_hydrodynamics()
mj_step()

---

# ⚠️ 严格禁止

❌ 直接施加 yaw torque 代替物理建模  
❌ 使用全局 drag 代替分段模型  
❌ 修改 RL 逻辑  

---

# 📊 Step 6：输出内容

请按顺序输出：

1️⃣ 代码结构分析  
2️⃣ 修改方案  
3️⃣ hydrodynamics.py 完整代码  
4️⃣ step 修改  
5️⃣ 参数说明  
6️⃣ 调参建议  

---

# 🎯 成功标准

实现后：

- 鱼可以形成弧线轨迹
- yaw rate > 0
- 左右转对称
- 与 body wave 强耦合

---

# END