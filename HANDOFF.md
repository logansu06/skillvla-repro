# HANDOFF — SkillVLA 仿真机制级复现

**截至 2026-09-27 的状态**：复现阶段已完成。5 个必做阶段（F0–F4）的验收检查全部通过：冒烟测试 exit 0，42 项测试通过，两个结果文件都通过协议检查。下一阶段是"改进"，还没有开始，方向待定（见第 11 节）。

论文：Zhai et al., *SkillVLA: Tackling Combinatorial Diversity in Dual-Arm Manipulation via Skill Reuse*，[arXiv:2603.03836v1](https://arxiv.org/abs/2603.03836)。论文没有公开代码、数据和权重。

**目录**

0. 会话概况
1. 这个复现是什么，不是什么
2. 结果
3. 快速开始
4. 代码地图、论文公式对照、`out/` 目录说明
5. 必须知道的假设与已知问题
6. 构建时间线与关键事件
7. 会话中的问答澄清
8. 开发机环境与本地资源
9. GitHub 仓库与账户设置
10. 踩过的坑
11. 未完成的工作、下一步与待决问题
12. 注意事项

---

## 0. 会话概况

- **时间**：2026-09-25 至 2026-09-27，一个 Claude Code 会话（Claude Opus 5.5）。
- **原始请求**：
  > 复现 https://arxiv.org/html/2603.03836v1，没有官方代码，在当前 workspace 中新建一个路径专门用于本次复现项目，在复现完成后再进行下一步改进
- **使用的流程**：ARIS 的 `/research-implement-feature` 技能，参数取默认值：
  - ASK=never：不中途提问，所有自行做出的决定都写进假设账本；
  - EFFORT=balanced：最多 5 个阶段、每阶段最多修 5 次、跨模型审查最多 2 轮；
  - ASSURANCE=draft：审查发现不阻塞报告，但必须如实写出。
- **流程要点**：
  - 先跑通最薄的端到端主干（F0），再逐个阶段加功能，每个阶段只用一条命令验收；
  - 每个论文没写清的决定，在写对应代码之前先记入 `implement-stage/ASSUMPTIONS.md`；
  - 最后请另一个模型家族读代码，找出没声明的决定。
- **开发机上的项目路径**：`/home/mars/Desktop/sunianbing/skillvla_repro/`。
- **会话中写过一份在线文档**：https://claude.ai/code/artifact/e7d60b93-0fd5-44c2-95fc-38b5021e6432 。它是私有文档，只有作者能看，而且可能没有同步到最后的修改。**以本 HANDOFF 为准。**

---

## 1. 这个复现是什么，不是什么

这是**机制级**复现：在自建的简化仿真里，用从头训练的小模型检验论文的**结构性论点**，不是数值复现。

| 部分 | 论文 | 本复现 |
|---|---|---|
| 平台 | 真机双臂 | 自建俯视 2D + 高度仿真，64×64 RGB，物体是色块（`skillvla/env.py`） |
| 骨干 | π0.5 的 PaliGemma 3B | 从头训练的小网络：CNN 图像 token + 词嵌入 + transformer，每个方法约 4M 参数 |
| 基线 π0.5 / TwinVLA | 真实模型 | **自己写的仿结构小模型**，只模仿与论点相关的结构特征（见下）。**没有用任何真实基线的权重或代码** |
| π0-FAST 基线 | 有 | 未做 |
| 高层 VLM（生成每臂子任务提示） | 微调后冻结的 PaliGemma | **规则解析**全局指令（`skillvla/planner.py`），相当于一个永远正确的高层 |
| 协作估计器的输入 z_H | 冻结高层 VLM 的 KV cache | 从头训练的小编码器，只接收门控损失的梯度 |
| 协作先验 α^vlm | Qwen3-VL-32B 离线标注 | 规则标签：协作任务为 1，单臂任务为 0 |
| 门控机制与各项 loss | 见论文 Sec. V | **按公式实现**（见第 4 节），BC 误差用 flow 速度误差代替 |

两个"仿结构"基线：
- **Mono（π0.5-like）**：一个共享编码器（图像 + 全局指令 + 双臂状态），接一个 action expert，输出 8 维拼接动作。
- **TwinVLA-like**：两路编码器，都读全局指令；编码器层和 expert 层都有始终开启的跨流注意力。

**能说明的**：在条件相同的受控环境中，"双臂共享表征 + 联合动作输出"会造成技能纠缠；而"分臂推理 + 门控跨臂通信"能组合技能，同时保留协作能力。

**不能说明的**：
- 真实 π0.5 / TwinVLA 在这些任务上的表现；
- PaliGemma 版 SkillVLA 能否达到论文的 51%；
- 任何依赖 VLM 预训练的效果，比如高层对新组合指令的理解、视觉鲁棒性；
- 真机表现。

绝对数值不能和论文直接比较，只比较方法之间的趋势。

---

## 2. 结果

所有数字：3 个训练种子，每个评测格 50 回合，每任务 50 条示范，训练 20k 步，评测最后一个 checkpoint。± 为各种子平均值的总体标准差（ddof=0）。原始数据见 `results/*.json`，表格由 `scripts/check_results.py` 生成到 `results/*.md`。

### Table I / II 对应：组合实验（`results/recomposition.json`）

用 6 个单臂技能训练（训练时另一臂静止，另一侧桌面无物体），测 9 个从没见过的左右组合。

| 方法 | 已学单臂技能 | 未见组合（零样本） | 论文（已学 / 组合） |
|---|---|---|---|
| Mono（π0.5-like） | 0.80 ± 0.06 | **0.00** ± 0.01 | 0.77 / 0.00 |
| TwinVLA-like | 0.76 ± 0.08 | **0.00** ± 0.00 | 0.67 / 0.04 |
| SkillVLA | 0.84 ± 0.03 | **0.46** ± 0.10 | 0.78 / 0.51 |

各格数值（3 个种子平均）：

| 方法 | cup×cake | cup×stir | cup×smash | box×cake | box×stir | box×smash | mug×cake | mug×stir | mug×smash |
|---|---|---|---|---|---|---|---|---|---|
| Mono | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 0.00 |
| TwinVLA-like | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SkillVLA | 0.24 | 0.41 | 0.31 | 0.41 | 0.85 | 0.29 | 0.38 | 0.71 | 0.57 |

| 方法 | cup | box | mug | cake | stir | smash |
|---|---|---|---|---|---|---|
| Mono | 0.43 | 0.95 | 0.97 | 0.53 | 0.99 | 0.93 |
| TwinVLA-like | 0.38 | 0.96 | 0.95 | 0.37 | 0.97 | 0.93 |
| SkillVLA | 0.49 | 0.94 | 0.94 | 0.69 | 1.00 | 0.96 |

- 组合成绩受单技能上限约束：Cup 和 Cake 单独做时就只有 0.4–0.7，含它们的组合最低；Stir 接近 1.00，含它的组合最高。
- SkillVLA 的协作门在全部 15 个评测格上都关闭（开门比例 0.00），组合能力完全来自"关门时两臂结构上独立"。
- 这些数字是 Stir 判定修正后，用保存的 checkpoint 重新评测得到的。最初训练完的评测结果是 SkillVLA 0.49，第一次修正后是 0.47，基线始终为 0.00（见第 6 节）。

### Table III 对应：协作实验（`results/cooperative.json`）

每个任务单独训练。括号内是 3 个种子各自的成功次数（满分 50）。

| 方法 | Shake | Ball | Align | 平均 | 论文（Shake / Ball / Align → 平均） |
|---|---|---|---|---|---|
| Mono（π0.5-like） | 0.79（38/34/46） | 0.77（40/37/38） | 0.24（9/9/18） | 0.60 ± 0.06 | 0.30 / 0.45 / 0.65 → 0.47 |
| TwinVLA-like | 0.76（35/39/40） | 0.89（47/45/41） | 0.33（12/18/19） | 0.66 ± 0.02 | 0.15 / 0.55 / 0.55 → 0.42 |
| SkillVLA | 0.91（43/49/44） | 0.91（45/44/47） | 0.29（13/17/14） | **0.70** ± 0.02 | 0.25 / 0.50 / 0.70 → 0.48 |
| SkillVLA w/o Attn | 0.19（11/6/12） | 0.09（1/9/3） | 0.25（18/11/9） | **0.18** ± 0.02 | 0.00 / 0.10 / 0.40 → 0.17 |

- SkillVLA 在三个协作任务上的门开启比例都是 1.00。
- 去掉跨臂注意力后，Shake 和 Ball 大幅下降，Align 几乎不变。论文同样观察到 Align 对低层耦合不敏感。
- Align 的绝对水平偏低（0.24–0.33，论文 0.55–0.70），原因见第 5 节。
- SkillVLA 在 Shake 上高于基线，与论文相反，不宜过度解读：这里一个"种子"同时改变了数据、训练和评测场景（A-026），只有 3 个种子。

### 训练中门控的实际数值（来自 `out/*_formal.log` 的第 20000 步）

| 实验 | L_on | L_off | y=1 的比例 | 门实际打开的比例 |
|---|---|---|---|---|
| 协作 Shake（3 个种子） | 0.014–0.020 | 0.79–1.37 | 1.00 | 1.00 |
| 协作 Ball | 0.005–0.010 | 1.09–2.10 | 1.00 | 1.00 |
| 协作 Align | 0.013–0.020 | 1.56–1.97 | 1.00 | 1.00 |
| 组合 seed 0、seed 2 | 0.0214、0.0268 | 与 L_on 完全相等 | 0.00 | 0.00 |
| 组合 seed 1 | 0.0270 | 0.0269 | 0.37 | 0.00 |

- 协作任务上，开通信的误差比关通信低两个数量级，标签很稳定。
- 组合实验（全是单臂数据）有两种情况：
  - 门从未打开时，跨臂注意力的输出层停在零初始化，L_on 与 L_off 完全相等，y 恒为 0；
  - 早期门短暂打开过时（seed 1），L_on 与 L_off 只差千分之一，y 近似噪声，门没被误开靠的是先验项。
- 这说明论文的离散门标签在"通信没用"的数据上是噪声，是一个可改进的点（见第 11 节）。

---

## 3. 快速开始

需要 NVIDIA GPU。代码里设备写死为 `cuda`，不支持 CPU。开发机是单卡 RTX 5090（32 GB），实际显存占用不到 4 GB。

```bash
git clone https://github.com/logansu06/skillvla-repro.git && cd skillvla-repro
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/pytest -q tests/                          # 42 项测试，约 1 分钟
.venv/bin/python scripts/run.py --smoke             # 主干冒烟，约 10 秒，只写 out/smoke.json
.venv/bin/python -u scripts/run.py --exp recomposition # 组合实验：9 个模型，约 2.3 小时
.venv/bin/python -u scripts/run.py --exp cooperative   # 协作实验：36 个模型，约 4.7 小时
.venv/bin/python scripts/check_results.py results/recomposition.json   # 检查协议 + 生成 .md 表格
./monitor.sh                                        # 实时终端面板（另开一个终端）
```

长时间实验建议放后台并写日志，监控面板会自动找到正在运行的进程和日志：

```bash
nohup .venv/bin/python -u scripts/run.py --exp cooperative > out/cooperative_formal.log 2>&1 &
```

`run.py` 常用参数：
- `--seeds 0,1,2`、`--methods mono,skillvla`、`--steps 20000`、`--eval-n 50`、`--demos 50`；
- `--pilot`：结果写到 `out/pilot_*.json`，不写 `results/`；
- `--eval-only`：从 `out/ckpt/` 加载已保存的权重重新评测，不训练。

`monitor.sh` 参数：`-n 2`（刷新间隔秒数）、`--detail`、`--once`、`--log out/x.log`。

诊断某个任务的失败原因（逐回合打印抓取、释放事件和最终位置）：

```bash
.venv/bin/python -u scripts/diagnose.py --method skillvla --task R_cake --ckpt out/ckpt/diag_x.pt --n 20
.venv/bin/python -u scripts/diagnose.py --method mono --task align --train align --ckpt out/ckpt/diag_y.pt --n 50
```

`--ckpt` 指定的文件不存在时会先训练一个模型并保存到这个路径，存在时直接加载。

单个模型训练 20k 步的用时：Mono 约 5 分钟，Twin 约 7–15 分钟，SkillVLA 约 13–24 分钟。SkillVLA 每步要分别前向开、关两个分支，所以最慢。两个实验同时跑时会互相拖慢。

---

## 4. 代码地图、论文公式对照、`out/` 目录说明

```
skillvla/
  env.py        仿真：6 个单臂技能、9 个组合、3 个协作任务、脚本专家、成功判定、渲染
  models.py     Mono / TwinVLA / SkillVLA / SkillVLA w/o Attn、门控、全部 loss
  flow.py       flow matching（openpi 约定）：x_τ = τ·ε + (1−τ)·a，目标 ε−a，τ ~ Beta(1.5,1)，10 步 Euler
  data.py       示范生成、动作块数据集（H=8，整份放在 GPU 上）
  planner.py    规则高层分解 + 规则协作先验
  text.py       封闭词表分词器
  train.py      AdamW 3e-4、OneCycleLR、bf16、梯度裁剪 1.0
  evaluate.py   批量并行闭环评测；每个动作块执行前 4 步
scripts/
  run.py            统一入口（--smoke / --exp / --eval-only / --pilot）
  check_results.py  强制检查声明的实验协议，并生成 results/*.md
  diagnose.py       逐回合打印抓取、释放事件，用来查失败原因
  monitor.py        实时监控面板（rich），monitor.sh 是它的启动器
tests/            test_env.py（32 项），test_models.py（10 项）
implement-stage/  SPEC.md、ASSUMPTIONS.md（假设账本）、BUILD_NOTE.md（构建记录）、SILENT_ASSUMPTION_SWEEP.json
results/          正式结果（只放正式结果）
out/              冒烟、pilot、诊断日志、修正前的结果备份、env_gallery.png（checkpoint 不在仓库里）
```

### 仿真里的任务

| 任务 | 执行臂 | 成功条件 |
|---|---|---|
| Cup | 左 | 抓起杯子放到盘上 |
| Box | 左 | 把盒子推过目标线（非抓取） |
| Mug | 左 | 抓起马克杯，举到 z≥0.6 并保持 5 步 |
| Cake | 右 | 抓起蛋糕放进容器 |
| Stir | 右 | 在碗中心周围的环带内累计带符号净转角 ≥ 2π |
| Smash | 右 | 从 z≥0.6 的高处快速下砸到目标 |
| Shake | 双臂 | 两臂一起抓住杯身和盖子，上下摇 3 个周期，盖子不能脱开 |
| Ball | 双臂 | 两臂同时把球抬到 z≥0.75，高度差 <0.04 |
| Align | 双臂 | 左臂取靠左的积木，两块积木 y 差 <0.05，释放时间差 ≤3 步 |

- 回合上限：单臂与组合任务 100 步，协作任务 120 步。
- 单臂任务还要求空闲臂在成功时刻离初始位置足够近。
- `out/env_gallery.png` 是每个任务第 12 步的画面，可以用来确认渲染：青色圆环是左臂，品红圆环是右臂，圆环越大表示越高，中心黑点表示夹爪闭合。

### 论文公式在 `skillvla/models.py` 中的位置

| 论文 | 代码 |
|---|---|
| 跨臂消息乘以门：a_i ~ p(a_i \| z_i, α·m_i) | `paired_forward()`：每层先算两臂消息，再加 `g * m`。g=0 时两臂完全独立，有测试验证 |
| 协作估计器 α ~ p(α \| z_H) | `CoopEstimator` + `SkillVLA.gate_prob()`（sigmoid 概率 ŷ） |
| L_BC^on、L_BC^off | `SkillVLA.loss()` 中的 `per_sample(ones)` 和 `per_sample(zeros)`，两次共享同一组 τ、噪声和观测 |
| 离散门 L_disc = BCE(y, ŷ)，y = 1[L_on < L_off]，stop-gradient | 同上，y 用 detach 后的值；相等时 y=0 |
| L_prior、L_sticky、L_sup | 同上，Bernoulli CE；前一时刻 ŷ_{t−1} 取同回合前一帧并 detach |
| 推理时二值化 | `SkillVLA.sample()`：ŷ ≥ 0.5 |

论文没写清、由本复现决定的部分：
- loss 系数：disc 1、prior 1、sticky 0.1、sup 0.01；
- 训练时用硬门 g 选择 BC 分支：BC = g·L_on + (1−g)·L_off；
- 跨臂注意力插在 expert 的每一层，每臂独立 QKV，输出投影零初始化；
- 用 flow 单步速度误差代替 ‖â−a‖²，因为后者要做完整 10 步去噪，成本高且带采样噪声。

没有实现：连续门版本 L_coop = λ·(L_on − L_off)_sg·α_t。它在论文中只用于消融。

各方法的低层参数量（前缀编码器 + action expert）对齐在 ±20% 以内：Mono 取 d=192 约 4.0M，Twin 与 SkillVLA 约 4.2M，w/o Attn 约 3.6M。SkillVLA 另有约 1.1M 的高层编码器和估计器，对应论文中额外的高层 VLM。

### `out/` 里各文件是什么

| 文件 | 内容 | 能否当结果用 |
|---|---|---|
| `recomposition_formal.log`、`cooperative_formal.log` | 两个正式实验的完整训练和评测日志 | 是，结果的来源 |
| `recomposition_reeval.log`、`recomposition_reeval2.log` | Stir 两次修正后用 `--eval-only` 重新评测的日志 | 是，第二次对应当前 `results/recomposition.json` |
| `recomposition_before_stir_fix.json/.md`、`recomposition_after_stir_fix_r1.json` | 修正前、第一次修正后的组合结果备份 | 否，只作对比 |
| `pilot_recomposition.*`、`pilot_cooperative.*` | 单种子 pilot，用的是旧配色、旧 Align 定义 | **否，已作废** |
| `diag_cake.log`、`diag_align*.log` | 诊断 Cake 和 Align 失败原因的逐回合记录 | 否，只用于排查 |
| `smoke.json` | 20 步冒烟模型的输出，键名带 `PLACEHOLDER_` | 否 |
| `env_gallery.png` | 任务画面示意 | — |

---

## 5. 必须知道的假设与已知问题

完整清单见 `implement-stage/ASSUMPTIONS.md`，共 37 条：30 条会影响结果含义（semantic），7 条影响接口（interface），其中 14 条来自跨模型审查。下面是最容易出错的几条：

- **A-003 高层是完美规则**：组合实验中 SkillVLA 永远拿到正确的每臂提示，0.46 是**低层解耦能力的上限**，对 SkillVLA 有利。指令本身就是 `left arm : … . right arm : …` 格式，没有需要理解的语言。门控在这里实际上等于给指令分类，没有检验回合内的模式切换。
- **A-004 单臂训练场景**：另一侧桌面没有物体。所以组合测试时画面对所有方法都是分布外的。
- **A-017 Align 修订过两次**：
  - 原设计要对齐到画面中 1 像素的目标线，而且谁抓哪块积木由抛硬币决定，所有方法都接近 0。
  - 现在的定义：左臂取靠左的积木，两块积木彼此 y 差小于 0.05，两臂释放时间差不超过 3 步。
  - 修订的接受标准是事先定好的"Mono 能学会"，Mono 诊断达到 12/50。这个标准比较宽，所以 Align 的绝对水平低，失败主要来自一臂跳过抓取。
- **A-017 Shake / Ball**：示范中每个回合的摇动节奏、抬升速度是随机的，只存在于示范数据里。评测时两臂都由策略控制，环境检验的是两个策略臂之间是否同步，没有外部"领臂"。
- **Stir 成功判定修过两次（A-012）**：
  - 第 1 次：原来累加绝对转角，来回晃动也算成功；
  - 第 2 次：穿过碗中心的直径往返也能骗过判定。
  - 现在的规则：只在碗中心周围的环带（0.015 < 距离 < 0.08）内累计带符号净转角，单步转角达到 90° 及以上不计入。
  - 组合实验的结果是用保存的 checkpoint 重新评测得到的（`--eval-only`），没有重新训练。
- **A-026 "种子"改变的是数据、训练、评测三者**：± 是三者合并的波动，不是单纯的训练随机性。
- **A-030 BC 日志不能跨方法比较**：SkillVLA 的 BC 是两臂损失之和，数值约为 Mono 的 2 倍。
- **A-036 / A-037 重新评测的性质**：
  - `--eval-only` 按文件名加载权重，无法从结果文件本身证明权重来源；
  - 所有回合共享一个噪声生成器，所以重新评测是重跑轨迹，不是给同一批轨迹重新打分。
- **Cake 颜色 bug（已修复）**：蛋糕原来是粉色，和右臂末端的品红圆环太接近，模型分不清"桌上"和"已抓住"，SkillVLA 的 Cake 只有 4/50。现在所有物体与两臂颜色的最小 RGB 距离为 179.5。

**跨模型审查**：请 GPT-6-Astra（xhigh，只读沙箱，Codex MCP，thread `01a0d831-a443-7882-84d0-f7c3aa20d594`）直接读代码，找没有写进账本的决定，共 2 轮。审查只拿到文件路径，不拿我的总结。原始回复见 `SILENT_ASSUMPTION_SWEEP.json`。
- 第 1 轮：找出 12 条未声明的决定（11 条 semantic）和 7 条与代码不符的陈旧条目。
- 第 2 轮：确认其中 17 项已解决；新发现 2 条未声明的决定（A-036、A-037）和 2 条不完整的修复（Stir 仍可被骗过；A-004 的备注提到了一个不存在的配置项）。
- 全部已处理。**审查轮次已用完，第 2 轮之后的修复（最终版 Stir 判定、A-036、A-037）只经过执行者和测试验证，没有经过审查者复核。**
- 审查者还确认：正式协作实验的进程启动于 Stir 修改之前，而且 Stir 不涉及协作任务，协作结果不受影响。

---

## 6. 构建时间线与关键事件

详细记录（每次的命令、退出码、修复次数）见 `implement-stage/BUILD_NOTE.md`，这里是摘要。

| 阶段 | 做了什么 | 结果与修复 |
|---|---|---|
| F0 主干 | 指令 → 示范生成 → 训练 → 闭环评测 → `out/smoke.json`，内部先用占位的 reach 技能和 MLP 策略 | 通过；修了 1 处：OneCycleLR 在 20 步时 warmup 除零 |
| F1 仿真 | 6 个单臂技能、9 个组合、3 个协作任务、脚本专家、成功判定 | 29 项测试通过，专家成功率均 ≥95%；修了 1 处：推盒子时末端会直接穿过盒子，改为沿运动方向拖动 |
| F2 模型 | 四种方法、门控跨臂注意力、协作估计器和全部门控损失 | 10 项测试通过；修了 2 处：w/o Attn 层缺少模块；BCE 在 bf16 autocast 下报错，改用 fp32 |
| 组合 pilot | seed 0，Mono 与 SkillVLA | 未见组合 SkillVLA 250/450、Mono 0/450；但 SkillVLA 的 Cake 只有 4/50 |
| Cake 诊断 | `diagnose.py` 逐回合记录 | 19/20 回合右臂直接飞到容器上方悬停，从未抓取。原因是颜色混淆，修复后 SkillVLA 的 Cake 在正式实验中为 0.69 |
| 协作 pilot | seed 0，四种方法 | Shake 38/39/43/11，Ball 40/45/45/1，**Align 1/1/2/0**（Mono/Twin/SkillVLA/w/o Attn） |
| Align 诊断与修订 | 两次修订 | 原版：放置位置与目标线无关，约一半回合只有一臂去抓。第 1 次修订后 Mono 0/20；第 2 次修订后 Mono 12/50，按事先定的标准接受 |
| F3 正式组合实验 | 3 方法 × 3 种子 × 15 格 × 50 回合，约 2.3 小时 | 未见组合 SkillVLA 0.49、基线 0.00 |
| 审查第 1 轮 | GPT-6-Astra 读代码 | 修了 Stir（第 1 次）、结果检查器改为强制检查协议、诊断和冒烟先设种子、新增 `--eval-only`；重新评测后 SkillVLA 0.47 |
| 审查第 2 轮 | 复核 | 修了 Stir（第 2 次）并补显式轨迹测试；再次重新评测后 SkillVLA 0.46。两次修正都只改变了 9 个含 Stir 的格子 |
| F4 正式协作实验 | 4 方法 × 3 任务 × 3 种子，36 个模型，约 4.7 小时 | SkillVLA 0.70，去掉跨臂注意力后 0.18 |
| 收尾 | 复跑全部检查 | F0 通过、42 项测试通过、F3/F4 结果检查通过 |

每次修测试，我都确认过：新测试在旧代码下会失败。也就是说，测试确实能抓到被修的问题。

---

## 7. 会话中的问答澄清

以下问题在会话中被问到过，接手的人很可能也会问。

**问：训练时加载了 π0.5 或 PaliGemma 吗？怎么微调的？**
答：没有。所有模型都是随机初始化、从头训练的，不存在微调。原因：论文没有公开数据，架构是冻结高层 VLM 加两份带 LoRA 的低层 VLM，全都基于 PaliGemma 3B。每个方法、每个种子都要训练，单卡跑不起；而论文的核心论点是结构性的，小模型也能检验。

**问：那基线是真的 π0.5 / TwinVLA 吗？**
答：不是。基线是模仿它们结构的小模型，只保留与论点相关的结构特征（见第 1 节）。结果与论文一致，但并没有测过真实的 π0.5。

**问：high-level 是怎么复现的？**
答：没有真正复现，用规则代替了（见 A-003）。代码就是一行正则：

```python
m = re.fullmatch(r"left arm : (.+?) \. right arm : (.+?)", instruction)
```

协作任务查一张固定表，例如 Shake 对应 `("shake the cup", "hold the cap")`。更忠实的做法按由易到难有三种：
1. 把指令改写成多种自然语言说法，训练一个可学习的小 planner；
2. 用现成 VLM 做 zero-shot 分解和先验标注；
3. 按论文做法微调 PaliGemma，然后冻结。

**问：α 的计算和各项 loss 复现了吗？**
答：复现了，按论文公式实现，有单元测试检查门关闭时两臂独立、stop-gradient 成立、标签符号正确。替代的只有三处：α 的输入来源、先验来源，以及用 flow 速度误差代替 BC 误差（见第 4 节）。

**问：不上传模型权重有什么影响？**
答：结果和结论都在 `results/*.json` 里，代码完整，可以重新训练（约 7 小时）。受影响的是：
- `--eval-only` 在别的机器上用不了；
- 重训的数字不一定逐位相同：种子固定，但 GPU 上的 bf16 和部分算子不保证确定性。这一点没有实际验证过；
- 报告数字背后的模型只存在开发机上。
决定：暂不上传。

**问：以后接入真实 π0.5，需要这些权重吗？**
答：不需要。小模型的权重和 π0.5 版本的参数名、形状、层数都对不上，没法加载，也没法迁移。真正要用的是 π0.5 自己的预训练权重（见第 8 节）。能带过去的是代码：门控机制、损失、评测流程、结果检查器。

---

## 8. 开发机环境与本地资源

| 项目 | 情况 |
|---|---|
| GPU | NVIDIA RTX 5090，32 GB |
| 内存 | 188 GB |
| 系统盘 | 1.8 TB，已用 97%，剩约 50 GB |
| Python | 系统 PATH 里没有 `python`，只有 `python3`（无 torch）；本项目一律用 `.venv/bin/python` |
| uv | `~/.local/bin/uv` |
| 本项目环境 | `skillvla_repro/.venv`：Python 3.10、torch 2.7.1+cu128，另有 numpy、matplotlib、pytest、pyyaml、rich。**没有改动任何已有的 conda 环境** |

**不在仓库里、只在开发机上的东西**

| 内容 | 位置 | 说明 |
|---|---|---|
| 模型权重（49 个 .pt，约 800 MB） | `skillvla_repro/out/ckpt/` | 正式实验 45 个：组合 9 个（`recomposition_all_<方法>_s<种子>.pt`）+ 协作 36 个（`cooperative_<任务>_<方法>_s<种子>.pt`）；另有 4 个诊断用。`--eval-only` 依赖这些文件 |
| 虚拟环境 `.venv`（6.4 GB） | `skillvla_repro/.venv` | 按第 3 节重建 |
| 论文 HTML | `skillvla_repro/implement-stage/paper.html` | arXiv 许可不允许我们再分发，请从 arXiv 获取 |
| 早期精读笔记与复现路线 | `/home/mars/Desktop/sunianbing/reports/SkillVLA_2603.03836/`（含 `reproduction_plan.md`），以及 `MyPaper/papers/2603.03836/` | 会话开始前就有，本次参考过。本复现采用 flow 速度误差代替 BC 误差，与其中的建议一致 |

**移植 π0.5 时可能用到的本机资源**（只确认了存在，没有实际运行过）：
- π0.5 预训练权重：`~/.cache/huggingface/hub/models--lerobot--pi05_base`，14 GB，含 PaliGemma 骨干；
- lerobot 的 π0.5 实现：conda 环境 `flowmatching`（`/home/mars/miniforge3/envs/flowmatching`）装了 lerobot 0.4.4，editable 安装，源码在 `/home/mars/Desktop/liuyizhou/new/lerobot_rokae/lerobot`，里面有 `src/lerobot/policies/pi05/`；
- PaliGemma tokenizer：`~/.cache/openpi/big_vision/`；
- 别人做过的 π0.5 LoRA 训练脚本：`/home/mars/Desktop/dyz/pi05_training/`；
- RoboTwin 双臂仿真：`/home/mars/lsy/maniflow/third_party/RoboTwin1.0`；
- π0-FAST 的权重本机没有找到，需要下载。

这台机器是多人共用的：上面这些路径大多在别人的目录下，使用前先确认。

---

## 9. GitHub 仓库与账户设置

- **仓库**：https://github.com/logansu06/skillvla-repro ，私有，默认分支 `main`。
- **推送账户**：`gh` 登录的是 logansu06。会话中途从 TianXingJian641 切换到 logansu06：先 `gh auth logout`，再 `gh auth login -h github.com -w`（浏览器授权），最后 `gh auth setup-git`。
- **本机 `gh` 版本很旧（2.4.0，2022 年）**：
  - `gh auth login` 没有 `-p` 参数，协议在交互中选；
  - 没有 `gh auth switch`，不支持多账户，切换账户只能先登出再登录；
  - `gh repo view --json` 没有 `visibility` 字段，要用 `isPrivate`。
  - 想要多账户切换，需要升级到 2.40 以上。
- **提交署名（重要）**：
  - 这台机器的**全局** git 配置是 `bzhou` / `2980882350@qq.com`，这个邮箱绑定在别人的 GitHub 账户 infinit-luffy 上。
  - 用全局配置提交，GitHub 会把提交显示成 infinit-luffy 的，包括仓库首页、提交历史、Contributors、Blame。
  - 本仓库已单独设置署名 `logansu06` / `logansu06@gmail.com`（写在 `.git/config`，不影响全局），首个提交已改正作者并强制推送，GitHub 确认归属 logansu06。
  - **在这台机器上新建的其他仓库，每次都要重新设置**：`git config user.name logansu06 && git config user.email logansu06@gmail.com`。
- **上传内容**：代码、测试、结果、日志、账本和文档，约 540 KB。`.venv`、`out/ckpt/`、论文 HTML 由 `.gitignore` 排除。

---

## 10. 踩过的坑

- **后台运行要加 `-u`**：第一次用 `nohup` 跑 pilot 时没加，输出被缓冲，日志里长时间只有一行表头。
- **两个实验同时跑**：GPU 利用率约 90%，两边都会变慢。监控面板的预计剩余时间在早期只按最快的方法估算，会偏乐观。
- **`OneCycleLR` 在极少步数时报错**：warmup 比例要保证至少 2 步，已在 `train.py` 里处理。
- **BCE 不能在 bf16 autocast 下计算**：门控损失被放在 fp32 里算。
- **渲染颜色**：新加物体时，颜色要和两臂末端颜色（青 `(0,255,255)`、品红 `(255,0,255)`）明显区分，否则模型分不清"桌上"和"已抓住"。
- **成功判定要用对抗性轨迹测一遍**：Stir 的判定两次被简单的晃动轨迹骗过，都是审查者发现的。新增或修改成功判定时，应当同时写"应该失败"的测试。
- **pytest**：会话中用 `-p no:cacheprovider` 运行，避免在项目里生成缓存目录。

---

## 11. 未完成的工作、下一步与待决问题

**推迟（本轮没做）**：
- π0-FAST 基线；
- 长程任务 Tubes / Collect Items，以及连续门与离散门的消融（Sec. VI-C）；
- 持续学习（Sec. VI-D）；
- 可学习的高层 planner；
- π0.5 / PaliGemma 骨干移植。

**建议的改进方向**（按我的优先级，用户尚未选定）：
1. **可学习的高层 planner**：改动小。把指令改写成多种自然语言说法，训练一个小模型从图像和指令生成 (u_L, u_R)，在未见组合上测分解准确率，看高层出错时 SkillVLA 会掉多少。
2. **换成真实骨干**：
   - 用真 π0.5 做基线，直接在任务数据上微调；
   - SkillVLA 的三个 VLM 位置共用一份冻结的 PaliGemma，两个低层各挂 LoRA，action expert 复制两份并加门控跨臂注意力；
   - 需要 224×224 的更真实画面，例如 RoboTwin，或者把现有仿真改成 3D 渲染；
   - bf16 基座约 6 GB，32 GB 显存能放下但 batch 会小很多，单次训练以小时计，整套实验是天级别；
   - 会话中提过的最小可行版：只做组合实验，真 π0.5 基线对 PaliGemma 版 SkillVLA，各跑 1 个种子，先看趋势是否还成立。
3. **改进门控标签**：在"通信没用"的数据上，y = 1[L_on < L_off] 是噪声或恒为 0（见第 2 节末尾）。可以加 margin，或改成相对差。
4. **在把结果写成结论之前**，先跑 `/experiment-audit` 和 `/result-to-claim`。

**待确认的问题**：
- 本复现的成功判定口径（A-012）与论文是否一致？论文没有给出完整的成功条件。
- 论文 Table II 中的"已学技能"是否也要求另一臂保持静止？
- 在线文档里方法表写的是"对应论文 π0.5"，容易让人误以为用的就是 π0.5 本身。会话中提议过改成"π0.5 结构的小模型"之类的说法，尚未执行。

---

## 12. 注意事项

- 所有命令都在仓库根目录运行；脚本通过 `sys.path` 找到 `skillvla` 包。
- 存数据集和权重前先确认磁盘空间，开发机系统盘只剩约 50 GB。
- 开发机的上级目录 `/home/mars/Desktop/sunianbing/` 里有一个损坏的空 `.git`，`git rev-parse` 会报"不是 git 仓库"。本仓库有自己的 `.git`，互不影响。
- `results/` 只放正式结果。`check_results.py` 会拒绝任何不符合声明协议的文件（方法、种子 0–2、eval_n 50、每任务 50 条示范、20k 步），也会拒绝任何带 `PLACEHOLDER` 的值。
- 改了成功判定之后，用 `run.py --exp <实验> --eval-only` 重新评测，再跑 `check_results.py`。旧结果先备份到 `out/`，方便对比哪些格子变了。
