# Assumption Ledger — SkillVLA 仿真机制级复现
<!-- ASK mode: never -->

| ID | Under-determined by the request | Chosen | Class | Source |
|----|--------------------------------|--------|-------|--------|
| A-001 | "复现"的层级：论文是真机 + π0.5 (PaliGemma 3B) 骨干，无代码无数据 | **机制级仿真复现**：自建平面双臂仿真 + 小型从头训练的 flow-matching VLA 式网络（CNN 视觉 token + 词嵌入语言 + transformer action expert）。数值不可与论文直接比较，只比较**方法间趋势**（组合 0% vs >0%、协作持平、去掉 attn 协作崩溃） | semantic | default |
| A-002 | 仿真环境与技能定义 | 自建 2D 俯视桌面（64×64 RGB 渲染），左臂技能 Cup/Box/Mug、右臂技能 Cake/Stir/Smash 按论文命名，但动作语义是本复现的几何类比（见 Notes） | semantic | default |
| A-003 | 高层 VLM（论文先微调 PaliGemma 做子任务生成再冻结）如何实现 | 规则解析全局指令 → `(u_L, u_R)`，等价于"完美高层"。结果只检验低层解耦与门控，**不检验高层对未见组合的语言泛化** | semantic | default |
| A-004 | 训练场景是否出现另一臂的物体 | 单臂训练示范中另一侧桌面**无物体**；idle 臂起始位置有 ±0.02 抖动，示范中执行带噪声的回 home 动作（不是严格静止）。场景写死在 `env.reset()`，**没有** distractor_prob 配置项（sweep 纠正） | semantic | default |
| A-005 | 门控标签中的"BC loss" | 用 flow-matching 速度误差 `‖v̂−(ε−a)‖²` 作代理：on/off 两次前向共享同一 τ、ε、观测 | semantic | default |
| A-006 | 协作先验 α^vlm（论文用 Qwen3-VL-32B 离线标注） | 规则标签：协作任务示范 = 1，单臂技能示范 = 0（按 episode 常数） | semantic | default |
| A-007 | 训练时 action loss 使用何种门 | action BC loss 用**二值化门** `1[ŷ≥0.5]`（ŷ 不接收 BC 梯度）；ŷ 只由 L_disc + λ_prior·L_prior + λ_sticky·L_sticky + λ_sup·L_sup 训练；推理阈值 0.5 | semantic | default |
| A-008 | 损失系数（论文未给） | λ_disc=1, λ_prior=1, λ_sticky=0.1, λ_sup=0.01 | interface | default |
| A-009 | 每臂 action expert 的状态输入 | SkillVLA 每臂只读本臂状态。Mono 在前缀中**直接**拼接双臂状态；TwinVLA-like 每流只接收本臂状态，另一臂状态经跨流注意力获得（sweep 纠正） | semantic | default |
| A-010 | 示范数与评测回合数 | 每技能/任务 50 条示范（同论文）；评测每格 **50** 回合（论文 10/20），报告 `success/n` 原始计数 | semantic | default |
| A-011 | 训练种子数 | 每个 (方法, 实验) 训练 3 个种子，报告均值与逐种子结果；论文未报告种子数 | semantic | default |
| A-012 | 成功判定 | 组合任务：两臂各自技能条件在 horizon 内都满足（latched）。单臂技能：该臂成功，**且 idle 臂在成功时刻离 home 近**（水平 <0.10、竖直 <0.2）；判定的是终点位置，**不追踪接触历史**（sweep 纠正）。Stir 成功 = 末端在碗中心周围的**环带**内（z<0.3、0.015<距中心<0.08）累计的**带符号净转角** ≥ 2π，单步转角 ≥90° 不计入（sweep 第 1 轮发现原实现累加绝对转角、来回晃动也算成功；第 2 轮发现穿过中心的直径往返仍能骗过带符号版本；两次都已修复，并用保存的 checkpoint 重新评测） | semantic | default |
| A-013 | 动作表示 | 每臂 4D：Δx, Δy, Δz（执行时 env 截断到 [-1,1] 再 ×最大步长）+ 夹爪 {-1,1}；chunk H=8，执行前 4 步。专家噪声加在限幅之后，**示范目标可略超 [-1,1]**（Shake 中最大约 1.10），训练目标与实际执行动作有这点差异（sweep 纠正） | interface | default |
| A-014 | TwinVLA 基线的实现 | 两个独立参数流，**VLM 层与 action expert 层均有始终开启的跨流注意力**，两流都接收全局指令（不做分解） | semantic | default |
| A-015 | Mono（π0.5-like）基线 | 共享编码器（图像 + 全局指令 + 双臂状态）→ 单个 action expert 生成 8D 拼接动作 | semantic | default |
| A-016 | 模型容量公平性 | 各方法**低层策略**（前缀编码器 + action expert）参数量对齐到 ±20% 内（Mono d=192 ≈4.0M，Twin/SkillVLA ≈4.2M，w/o Attn ≈3.6M）；SkillVLA 另有 ≈1.1M 的高层编码器 + 协作估计器，不计入，对应论文中额外的高层 VLM | semantic | default |
| A-017 | 协作任务（Shake/Ball/Align）的仿真定义 | 见 Notes。Shake/Ball 的每回合随机节奏或速度**只存在于示范数据中**，是两臂动作必须一致的隐变量；评测时两臂都由策略控制，环境检验的是**两个策略臂之间的同步**，没有外部领臂（sweep 纠正原"跟随领臂"的表述） | semantic | default |
| A-019 | 协作估计器读取的高层表征 z_H（论文：冻结的高层 VLM KV cache） | 本复现没有高层 VLM（A-003），z_H 由一个小型"高层编码器"（图像 + 全局指令）给出，**仅通过门控损失从头训练**；估计器为 1 个 learned query 的 2 层 transformer decoder | semantic | default |
| A-020 | L_sticky 中 ŷ_{t-1} 的来源与梯度 | 同 episode 前一帧（t=0 时取自身）重新前向得到 ŷ_{t-1}，并 **detach**；ℓ 用 Bernoulli CE | interface | default |
| A-021 | 跨臂 cross-attention 插在 action expert 的哪些层、如何初始化 | **每一层**都插入（self-attn → 前缀 cross-attn → 跨臂 cross-attn → MLP）；每臂独立 QKV；输出投影零初始化，所以初始时门开/关等价 | interface | default |
| A-022 | 低层"每臂 VLM"如何实现（论文：共享冻结 PaliGemma + 各自 LoRA） | 两个参数完全独立的小型前缀编码器（从头训练，无共享冻结基座） | interface | default |
| A-023 | 门控训练时 on/off 两个分支的梯度 | 两个分支都带梯度前向；逐样本 BC = g·L_on + (1−g)·L_off（g 为硬门）；门控标签 y=1[L_on<L_off] 用 detach 值（严格小于，相等时为 0） | semantic | default |
| A-024 | 示范采集是否过滤失败回合 | 只保留专家成功的回合，直到每任务凑满 50 条。正式组合实验数据中实际没有被拒绝的回合 | semantic | sweep |
| A-025 | 训练样本权重 | 按**帧**均匀采样，每帧都作为 chunk 起点；末尾不足 H 的 chunk 按有效长度归一化。任务权重因此与回合长度成正比（如 Stir 平均 46 帧，Smash 21 帧） | semantic | sweep |
| A-026 | "种子"改变什么、如何汇总 | 一个种子同时改变示范数据、模型初始化和训练顺序、评测场景和动作采样。报告的 ± 是**数据 + 训练 + 评测**的合并方差，不是单纯的训练随机性。格子等权平均，std 为总体标准差（ddof=0） | semantic | sweep |
| A-027 | 回合时长与终止 | 单臂与组合任务 horizon 100 步，协作任务 120 步；Align 两臂都释放但不满足条件时**立即判失败并终止**（不可补救） | semantic | sweep |
| A-028 | 抓取与放置物理 | 仅在夹爪由开到合的瞬间尝试抓取（半径 0.045、z<0.12）；被抓物体吸附在末端；松开时物体**立即**落到桌面、保持当前水平位置，没有下落或滑动 | semantic | sweep |
| A-029 | 状态向量内容 | 每臂 7 维 = 位置 xyz + 夹爪 + **上一步截断后的移动指令**（含动作历史，不只是本体感受） | semantic | sweep |
| A-030 | 各方法 BC 损失的尺度 | Mono/Twin 在 8 个动作维度上取均值；SkillVLA 系列是两臂各 4 维均值**之和**，同样误差下日志中的 BC 是 Mono 的 2 倍。优化器是 Adam，损失整体缩放基本不影响更新，只经全局梯度裁剪（1.0）产生细微差别；门控损失只作用于估计器参数，与 BC 参数不重叠。**方法间的 BC 日志数值不可直接比较**。未为此重新训练 | semantic | sweep |
| A-031 | 门开启比例的定义 | `gate_on_fraction` = 每回合内 chunk 级门决策的平均，再按回合等权平均。不是"执行步中开门的比例" | semantic | sweep |
| A-032 | 结果检查器的完整性标准 | 已修复：`check_results.py` 强制声明的协议（全部方法、seeds=[0,1,2]、eval_n=50、每任务 50 条示范、20k 步）。修复前只对照文件自身的列表，单种子 pilot 也能通过 | semantic | sweep |
| A-033 | 诊断 / 冒烟运行的初始化种子 | 已修复：`diagnose.py` 和 `--smoke` 在建模型之前设种子。此前诊断模型的初始化不可复现（这些诊断影响了 Cake/Align 的环境修订，但修订依据是定性失败模式，不依赖具体数字）。正式实验路径一直是先设种子 | semantic | sweep |
| A-034 | 场景分布 | home 位置固定并加 ±0.02 抖动；单臂物体在本侧 x∈[0.10,0.40]/[0.60,0.90]、y∈[0.32,0.78]，最小间距 0.16；Box y∈[0.35,0.6]；Shake/Ball 物体中心 x=0.5、y∈[0.4,0.7]，间距固定；Align 见 A-017。报告的成功率只代表这些分布 | semantic | sweep |
| A-035 | 训练与选模配方 | 20k 步、batch 256、AdamW lr 3e-4、weight decay 1e-4、OneCycleLR（warmup 5%）、bf16 autocast、梯度裁剪 1.0；**评测最后一个 checkpoint**，不做选模 | interface | sweep |
| A-036 | `--eval-only` 重新评测的来源信息 | 按文件名 `out/ckpt/<exp>_all_<method>_s<seed>.pt` 加载权重；结果文件中的 steps/demos 取自命令行，train_history 从旧结果文件复制。**检查器无法从数字本身证明 checkpoint 的来源**；本次两次重新评测均使用正式运行保存的 9 个 checkpoint | semantic | sweep |
| A-037 | 评测中的动作噪声分配 | 同一 (方法, 种子, 格子) 的所有回合共享一个噪声生成器，已结束的回合退出后续批次。因此改变任一回合的终止时刻，会改变其余回合拿到的噪声：**重新评测是重跑轨迹，而不是给同一批轨迹重新打分**。例如 SkillVLA seed 2 cup×stir 的左臂成功数在 Stir 修复后从 29 变为 31，而左臂判定并未改动 | semantic | sweep |
| A-018 | 运行环境 | 独立 uv venv `skillvla_repro/.venv`（torch 2.7.1+cu128），不改动已有 conda 环境 | interface | default |

## Notes

- **Phase 4 sweep（第 1 轮，GPT-6-Astra xhigh）** — 结论 gaps：12 条未声明（11 条 semantic），7 条陈旧行。已全部入账（A-024…A-035）或更正（A-002/004/009/012/013/017）。其中 3 处改了代码：Stir 成功判定（改为带符号净转角，组合实验用保存的 checkpoint 重新评测，SkillVLA 未见组合 0.49→0.47，基线仍为 0.00）、结果检查器强制协议、诊断和冒烟运行的种子顺序。原始回复见 `SILENT_ASSUMPTION_SWEEP.json`。
- **Phase 4 sweep（第 2 轮，最后一轮）** — 结论 gaps：第 1 轮 17 项确认已解决；新发现 2 条未声明（A-036、A-037，均 semantic）和 2 条陈旧（A-004 Notes 中不存在的配置项；Stir 修复不完整，穿过中心的直径往返仍算成功）。均已处理：Stir 改为环带 + 单步 <90° 规则，并新增显式轨迹测试（已确认第 1 轮判定无法通过该测试），组合实验第二次用 checkpoint 重新评测，SkillVLA 未见组合为 **0.46±0.10**，基线 0.00。**轮次预算已用完，这些第 2 轮之后的修复只经过执行者和测试验证，没有经过审查者复核。**

- **A-001** — 真机和 π0.5 权重级复现在本机不可行：没有双臂技能数据集，磁盘仅剩 53G，也只有单张 5090。另一种做法是把 SkillVLA 结构接到 lerobot 的 pi05 实现上，并在 RoboTwin 上训练，列为 DEFERRED（下一步"改进/放大"时的首选）。论文的核心论点（skill entanglement 是结构性问题）与骨干规模无关，因此小模型仿真能检验它。代价是：**VLM 预训练带来的泛化不在检验范围内**。
- **A-002** — 左臂：Cup = 抓杯放到盘上；Box = 推盒到目标线（非抓取）；Mug = 抓杯提起并保持在高位。右臂：Cake = 抓蛋糕放进盒；Stir = 移到碗上方、下降、画圈；Smash = 升到物体上方再快速下砸。
- **A-002（修订）** — 物体颜色须与两臂末端颜色明显不同：修复后最小 RGB 距离为 179.5（container 对左臂青色；左臂在单臂数据中从不靠近 container），其余均 ≥180（sweep 纠正了原"全部 ≥180"的说法）。原 cake 颜色与右臂品红过近，造成"已抓/未抓"视觉混淆，SkillVLA 在 Cake 上仅 4/50。已修复，见 BUILD_NOTE。
- **观察（非假设）** — 当跨臂通信对某类数据没有作用时（如全部单臂示范），L_on 与 L_off 几乎相等（0.0730 vs 0.0725），y=1[L_on<L_off] 近似抛硬币（y_frac≈0.3–0.43）。门没有被误打开，靠的是先验项 L_prior。这是论文离散门标签的固有噪声，也是可能的改进点（例如加 margin）。
- **A-003** — 这是有利于 SkillVLA 的简化。论文里 SkillVLA 的组合泛化部分来自高层 VLM；这里高层是完美的，所以 SkillVLA 的数字是"低层上限"。反转方法：把 `planner=rule` 换成可学习 planner，在 DEFERRED 列表中。
- **A-004** — 如果训练场景里另一侧有静止物体，Mono 会学到"看到物体也保持 idle"，组合更难；没有物体则测试时是视觉 OOD。两者都偏向"基线更差"，这里选论文 Fig.3 描述（"collected from single-arm executions while the other arm remains stationary"）最直接的读法。场景写死在 `skillvla/env.py` 的 `reset()` / `_setup_*` 中，要改这个条件需要改代码（没有配置项）。
- **A-005** — 论文公式写的是 `‖â−a‖²`。flow 策略下直接算 â 需要完整去噪采样（10 步），成本高且带采样噪声。已有复现计划（`reports/SkillVLA_2603.03836/reproduction_plan.md` §3.2）也采用同一代理。
- **A-007** — 另一种做法是用 ŷ 做软门并对 BC 反传。论文把 α 写作门控消息的系数，但离散版明确是"binarize at inference"。训练时用硬门让训练与推理一致；冷启动依赖先验 L_prior（A-006）把协作数据上的门打开。
- **A-017** — Shake：左臂持杯、右臂按住盖。左臂的上下摇动幅度和频率每 episode 随机，右臂必须保持 |Δ| < tol。Ball：双臂同时抬球，抬升速度每 episode 随机，要求 |z_L − z_R| < tol。Align（第 2 次修订，最终）：中央两块积木，位置随机（x∈[0.42,0.58]，两块 x 相差 ≥0.05），**左臂取更靠左的那块**，所以分配随试次变化，需要从画面推断（论文 "assignment is randomized across trials, requiring the policy to resolve task allocation on the fly"）。两臂抓同一块即失败。成功条件：两块各放到本侧（x<0.45 / x>0.55），两块彼此 y 差 < 0.05（论文 "align collinearly"），释放时间差 ≤ 3 步。放置高度 = 两块初始 y 的均值（画面可见）。需要低层协作的部分是同步：示范中先到的臂会等待另一臂。修订经过：(1) 最初版本要求对齐到画面中 1px 的外部目标线，且分配由抛硬币决定、与几何无关。pilot 中所有方法都只有 0–2/50，诊断发现放置 y 与目标线无关。(2) 第 1 次修订去掉目标线、改为隐藏的随机共享高度，Mono 仍 0/20：约一半回合只有一臂抓取，另一臂犹豫，原因是抛硬币分配在画面上无从区分。(3) 第 2 次修订如上。**修订的接受标准事先固定为"论文中 Align 最强基线 Mono（π0.5 0.65）能学会"，在看到 SkillVLA 于新定义上的结果之前确定**，以避免按方法结果调 benchmark。
