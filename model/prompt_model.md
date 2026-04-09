# 构建机器鱼 MuJoCo XML 模型（基于整体模型离散化的平面 2D 版本）

【角色设定】
你是一名精通机器人动力学、MuJoCo MJCF 建模、拉线驱动（tendon-driven systems）、串联柔性机构离散化建模、以及数值稳定性调试的高级工程师。

【任务目标】
我当前已有的机器人几何模型在外形上是一个整体连续结构，而不是已经切分完成的多刚体装配体。
你的任务不是简单地在现有独立 body 上补 joint / tendon，而是要在 MuJoCo XML 中，基于整体外形和机构逻辑，将该整体模型离散化为可关节连接的多刚体系统，并为该系统补全完整、可运行、物理合理的 MuJoCo XML 组件，包括：

- `<joint>`（关节）
- `<site>`（过线孔 / tendon routing 点）
- `<tendon>`（spatial 拉线系统）
- `<actuator>`（基于 tendon 的驱动器）
- 必要的 `<default>` 配置

【核心建模目标】
当前版本只用于“水平面 2D 运动”仿真，因此请严格按“平面简化版”建模，不要输出 3D 全向版本，不要输出 Pitch 自由度，不要输出上下两侧拉线。
目标是先得到一个可运行、可调试、物理合理、适合后续 RL 接入的 MuJoCo XML 结构。

【强约束要求】
1. 所有输出必须符合 MuJoCo 官方 MJCF 语法，可直接复制粘贴使用，不允许伪代码。
2. 必须保证物理合理性，避免出现无限拉伸、无约束、过约束、冗余自由度、明显数值不稳定等问题。
3. tendon 必须建模为“只能拉不能推”的单向受力系统。
4. 所有单位必须使用 SI 单位（m, kg, N, rad）。
5. 当前版本严格只保留水平 2D 运动：
   - 只允许 X、Y 平移和绕 Z 轴的 Yaw 转动
   - 禁止 Z 平移、Roll、Pitch
6. 当前版本严格只保留平面内弯曲：
   - 所有离散关节只允许 Yaw hinge
   - 不允许生成 Pitch hinge
7. 当前版本严格只保留左右两侧拉线：
   - 不要生成 top / bottom sites
   - 不要生成 top / bottom tendons
   - 不要生成 top / bottom actuators
8. 如果某些几何参数在我提供的信息中未明确给出，请使用清晰的占位符常量名，不要擅自臆造不确定数值。
9. 输出时必须优先保证“可运行、可调试、逻辑一致”，而不是一次性追求完整复刻真实复杂结构。

--------------------------------------------------
## 一、整体模型的 XML 离散化原则（必须严格执行）
--------------------------------------------------

我当前提供的机器人模型在几何上是一个整体连续结构，因此你必须在 XML 层面先将其离散为多个刚体段，而不是把整条鱼继续当成一个单独 body 使用。

请严格遵守以下原则：

1. 不要将整条鱼继续建模为一个单独 body。
2. 必须按机构逻辑将整体模型离散为多个串联刚体段，并在相邻段之间建立关节。
3. 若原始几何是单一整体 mesh，不能假设 XML 可以自动将一个 mesh 切分成多个独立刚体。
4. 因此在 XML 中，请采用以下方式之一：
   - 使用 capsule / box / ellipsoid 等 primitive geom 对各离散段进行动力学近似；
   - 或使用占位符 mesh 名称表示“每个离散段应有独立 mesh”；
   - 但绝对不要继续把整条鱼作为同一个 mesh 绑定到单一 body 上。
5. 若视觉外观需要保持整体鱼体形状，可将“动力学离散体”和“可视外形”分离处理：
   - 动力学上使用离散刚体链；
   - 视觉上允许使用简化外观 geom 或装饰性 mesh；
   - 但不得破坏关节拓扑与动力学独立性。
6. 离散时请优先依据机器人中心线进行分段：
   - 主动驱动链各段长度尽量一致；
   - 柔性尾链可采用更短的离散段长度，以更好近似硅胶尾的连续弯曲。

--------------------------------------------------
## 二、离散后的目标拓扑结构（必须严格按此生成）
--------------------------------------------------

### 1）整体三段式结构
机器人整体采用三段式结构，由：
- 中部主控舱 `centre_compartment`
- 前驱段（Front Body）
- 后驱段（Back Body）

组成。

### 2）根节点定义
- `centre_compartment` 为整个模型的根 body。
- 当前版本若需要补充根节点自由度，请不要保留完整 `freejoint`。
- 请将根部自由度严格限制为水平 2D 平面运动，即：
  - 允许：X 平移、Y 平移、绕 Z 轴的 Yaw 转动
  - 禁止：Z 平移、Roll、Pitch

也就是说，若你需要定义根部自由度，请使用 planar base 的等效实现，而不是完整 6 自由度自由体。

### 3）前驱主动链拓扑
前驱段必须离散为以下串联刚体链：

`centre_compartment -> front_v1 -> front_v2 -> ... -> front_v9 -> head`

请严格按以下语义理解：
1. `front_v1 ~ front_v9` 为前驱段中间刚性椎骨单元。
2. `head` 为前驱段终端刚体。
3. `head` 同时也是前驱 tendon 的锚定末端。
4. 不要再生成 `front_v10`。
5. 不要在 `head` 之前额外插入新的独立关节级。

### 4）后驱主动链拓扑
后驱段必须离散为以下串联刚体链：

`centre_compartment -> back_v1 -> back_v2 -> ... -> back_v9 -> joint_tail`

请严格按以下语义理解：
1. `back_v1 ~ back_v9` 为后驱段中间刚性椎骨单元。
2. `joint_tail` 为后驱段最后一个刚性尾基单元。
3. `joint_tail` 同时也是后驱 tendon 的锚定末端。
4. 不要再生成 `back_v10`。
5. 不要把后驱 tendon 继续延伸进柔性尾部。

### 5）柔性尾部离散拓扑
原始文件中的 `tail` 当前还是一个整体，但在当前 MuJoCo XML 中，不要将其继续保留为单个整体 body。
必须将其离散为多个小尾段，构成被动柔性尾链：

`joint_tail -> tail_seg1 -> tail_seg2 -> ... -> tail_segN`

要求如下：
1. `tail_seg1 ~ tail_segN` 为柔性尾部离散后的多个小刚体段。
2. 每个 `tail_seg` 之间只保留 1 个水平面 Yaw hinge。
3. 所有 `tail_seg` 关节都必须是被动关节。
4. 柔性尾链不配置 tendon 驱动，不配置 actuator。
5. 柔性尾链仅用于近似硅胶尾的被动摆动。
6. 如果尾段数量 `N` 未明确给定，可采用示例性的 `tail_seg1 ~ tail_seg4`，或者使用占位符 `TAIL_SEG_NUM`，但命名模式必须一致。

--------------------------------------------------
## 三、自由度与坐标轴约定（必须严格执行）
--------------------------------------------------

### 1）当前版本只保留水平 2D 运动
虽然真实机械结构可等效为 2 自由度万向节（Yaw + Pitch），但当前 XML 版本只考虑水平平面运动，因此：

1. 所有相邻离散 body 之间只保留 1 个 `<joint type="hinge">`
2. 该 hinge 仅表示 Yaw 自由度
3. 关节轴统一绕 `Z` 轴
4. 不要输出 Pitch hinge
5. 不要输出任何与俯仰相关的 site、tendon、actuator 或约束配置

### 2）坐标轴约定
请统一采用以下坐标约定：
- 鱼体轴向前进方向：`+X`
- 水平侧向摆动方向：`+Y`
- 竖直方向：`+Z`
- 当前版本唯一保留的关节自由度为 Yaw，绕 `Z` 轴转动

--------------------------------------------------
## 四、核心物理参数（需严格代入）
--------------------------------------------------

请在生成代码中严格使用以下参数：

- 线孔偏置半径：`r = 0.0375` m
- 主动链每个 Yaw hinge 的等效刚度：`stiffness = 1.041` N·m/rad
- 主动链每个 Yaw hinge 的等效阻尼：`damping = 0.05` N·m·s/rad
- 主动链每个 Yaw hinge 的运动范围：`range = [-15, 15]` degree
- 每根 tendon actuator 的控制范围：`ctrlrange = [0, 100]` N

说明：
1. 上述刚度、阻尼、限位优先用于前驱/后驱主动驱动链中的 Yaw hinge。
2. 柔性尾链 `tail_seg1 ~ tail_segN` 的被动关节不要直接照搬主动链参数。
3. 对柔性尾链，请使用单独占位符参数，例如：
   - `TAIL_PASSIVE_STIFFNESS`
   - `TAIL_PASSIVE_DAMPING`
   - `TAIL_PASSIVE_RANGE`

--------------------------------------------------
## 五、离散段几何与关节建模规则
--------------------------------------------------

### 1）主动链建模规则
前驱主动链和后驱主动链中的每个离散刚体段都应：
1. 作为一个独立 `<body>` 存在；
2. 包含一个代表该段外形和质量分布的 geom（可使用 primitive geom 或占位 mesh）；
3. 与其父 body 之间只通过一个 Yaw hinge 连接；
4. 使用上文给定的主动关节刚度、阻尼和 range。

### 2）柔性尾链建模规则
每个 `tail_seg` 都应：
1. 作为一个独立 `<body>` 存在；
2. 具有较轻、较细的几何表示；
3. 与父尾段之间通过一个被动 Yaw hinge 连接；
4. 不配置 actuator；
5. 不配置 tendon 连接点，除非我明确要求。

### 3）示例性占位符
如果以下参数我未提供，请使用明确占位符，不要擅自猜数值：
- `SEG_LEN_FRONT`
- `SEG_LEN_BACK`
- `SEG_LEN_TAIL`
- `SEG_MASS_FRONT`
- `SEG_MASS_BACK`
- `SEG_MASS_TAIL`
- `SITE_X_OFFSET`
- `TAIL_SEG_NUM`
- `TAIL_PASSIVE_STIFFNESS`
- `TAIL_PASSIVE_DAMPING`
- `TAIL_PASSIVE_RANGE`

--------------------------------------------------
## 六、空间穿线逻辑（Tendon Routing，仅保留左右两侧）
--------------------------------------------------

当前版本只做水平面 2D 运动，因此只保留左右两侧拉线系统，不要生成 top / bottom 相关内容。

### 1）前驱 tendon 作用对象
前驱 tendon 只作用于离散后的前驱主动链：
- `front_v1 ~ front_v9`
- `head`

前驱 tendon 的终点必须是 `head`。

### 2）后驱 tendon 作用对象
后驱 tendon 只作用于离散后的后驱主动链：
- `back_v1 ~ back_v9`
- `joint_tail`

后驱 tendon 的终点必须是 `joint_tail`。

### 3）柔性尾链不参与 tendon 驱动
离散后的柔性尾链：
- `tail_seg1 ~ tail_segN`

默认不布置 tendon site，不配置 tendon 驱动，也不配置 actuator。
它们仅通过被动关节与惯性/环境耦合产生摆动。

--------------------------------------------------
## 七、Site 布置规则（仅左右两侧）
--------------------------------------------------

### 1）主控舱上的 site
请在 `centre_compartment` 上建立以下 site：
- `centre_front_left`
- `centre_front_right`
- `centre_back_left`
- `centre_back_right`

### 2）前驱主动链上的 site
请在以下 body 上建立左右两个 site：
- `front_v1 ~ front_v9`
- `head`

命名格式必须为：
- `front_v1_left`，`front_v1_right`
- ...
- `front_v9_left`，`front_v9_right`
- `head_left`，`head_right`

### 3）后驱主动链上的 site
请在以下 body 上建立左右两个 site：
- `back_v1 ~ back_v9`
- `joint_tail`

命名格式必须为：
- `back_v1_left`，`back_v1_right`
- ...
- `back_v9_left`，`back_v9_right`
- `joint_tail_left`，`joint_tail_right`

### 4）柔性尾链上的 site
默认不要在 `tail_seg1 ~ tail_segN` 上布置 tendon site，除非我明确要求。

### 5）Site 局部坐标规则
以各 body 的局部坐标系为准，在当前 2D 版本中只放置左右两个 site。
请在全模型中统一采用一套一致的左右坐标方案，不要混用。

可选方案之一：
- 左侧 site：`(0, +r, 0)`
- 右侧 site：`(0, -r, 0)`

若需要沿轴向前后微调 site 位置，请使用清晰占位符，如：
- `SITE_X_OFFSET`
- `SITE_X_OFFSET_FRONT`
- `SITE_X_OFFSET_REAR`

不要擅自猜测不确定偏移量。

--------------------------------------------------
## 八、Tendon 路由规则（必须严格按此生成）
--------------------------------------------------

### 1）前驱 tendon
请生成 2 根独立前驱拉线：
- `front_wire_left`
- `front_wire_right`

其中：
- `front_wire_left` 必须从 `centre_front_left` 出发，依次穿过前驱主动链左侧 site，最终锚定在 `head_left`
- `front_wire_right` 必须从 `centre_front_right` 出发，依次穿过前驱主动链右侧 site，最终锚定在 `head_right`

### 2）后驱 tendon
请生成 2 根独立后驱拉线：
- `back_wire_left`
- `back_wire_right`

其中：
- `back_wire_left` 必须从 `centre_back_left` 出发，依次穿过后驱主动链左侧 site，最终锚定在 `joint_tail_left`
- `back_wire_right` 必须从 `centre_back_right` 出发，依次穿过后驱主动链右侧 site，最终锚定在 `joint_tail_right`

### 3）Tendon 建模要求
1. 必须使用 `<spatial>` tendon
2. tendon 路径必须连续、命名一致、无交叉
3. 当前版本总共只允许 4 根 tendon：
   - `front_wire_left`
   - `front_wire_right`
   - `back_wire_left`
   - `back_wire_right`
4. 不要生成上下方向 tendon
5. 不要输出 3D 全向 OQW 的 8 线版本
6. 不要把 tendon 继续延伸到柔性尾链中

--------------------------------------------------
## 九、Actuator 配置（仅保留平面拮抗控制）
--------------------------------------------------

当前版本只做平面 Yaw 运动，因此请采用“4 个独立 tendon motor”的实现方式，不要输出高层抽象的 2 个 servo 联动写法。

请直接配置以下 4 个 actuator：
- `front_left_motor` -> 作用于 `front_wire_left`
- `front_right_motor` -> 作用于 `front_wire_right`
- `back_left_motor` -> 作用于 `back_wire_left`
- `back_right_motor` -> 作用于 `back_wire_right`

要求如下：
1. 每个 actuator 独立绑定 1 根 tendon
2. 每个 actuator 只允许正向拉力输出，不允许负向推力
3. 左右两根线构成拮抗对，但拮抗分配逻辑由我在 Python / RL 控制器中实现
4. XML 中不要实现复杂的高层舵机联动逻辑
5. 不要输出 top / bottom actuator
6. 不要输出 Pitch 相关 actuator
7. 不要为柔性尾链 `tail_seg1 ~ tail_segN` 配置 actuator

--------------------------------------------------
## 十、数值稳定性要求（必须考虑）
--------------------------------------------------

为了保证该长链 tendon-driven 模型在 MuJoCo 中可稳定运行，请遵守以下原则：

1. 不要引入多余自由度
2. 不要为同一物理功能重复建模
3. 保持 tendon 路径简洁，避免不必要的急转折
4. 所有关节限位、阻尼和刚度必须显式给出
5. 若需要示例 geom / site 大小，请给出合理的小尺寸，而不是过大尺寸
6. 若相邻离散段碰撞会导致数值不稳定，请优先采用适合平面链条调试的接触简化策略
7. 当前版本优先保证：
   - 可运行
   - 可调试
   - 逻辑一致
   - 便于后续 RL 接入
8. 不要输出看似完整但实际上难以运行的复杂 3D 配置

--------------------------------------------------
## 十一、输出格式要求（必须严格遵守）
--------------------------------------------------

请不要给我伪代码，也不要省略 XML 标签。
请严格按以下顺序输出：

1. 你采用的离散化和平面化简化假设（简短列出）
2. `<default>` 代码块
3. 一个示例性的主动链 `<body>` 内部结构代码块（例如 `front_v1`）
4. 一个示例性的 `head` 或 `joint_tail` `<body>` 内部结构代码块
5. 一个示例性的离散柔性尾段 `<body>` 内部结构代码块（例如 `tail_seg1`，展示被动关节写法）
6. 完整的 `<tendon>` 代码块
7. 完整的 `<actuator>` 代码块

【最终一致性要求】
输出内容必须同时满足以下全部条件：
- 体现“整体模型在 XML 中被离散为多刚体链”
- 只保留水平 2D 运动
- 前驱链为 `front_v1 ~ front_v9 -> head`
- 后驱链为 `back_v1 ~ back_v9 -> joint_tail`
- 柔性尾部为 `joint_tail -> tail_seg1 ~ tail_segN`
- tendon 只作用于主动链
- 柔性尾链只保留被动 Yaw hinge
- 不要输出 Pitch 版本
- 不要输出 top / bottom 拉线版本
- 不要重新合并成 2 个 servo