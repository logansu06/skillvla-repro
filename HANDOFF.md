# HANDOFF — SkillVLA 仿真机制级复现

**截至 2026-09-27 的状态**：复现阶段已完成。5 个必做阶段（F0–F4）的验收检查全部通过：冒烟测试 exit 0，42 项测试通过，两个结果文件都通过协议检查。下一阶段是"改进"，还没有开始。

论文：Zhai et al., *SkillVLA: Tackling Combinatorial Diversity in Dual-Arm Manipulation via Skill Reuse*，[arXiv:2603.03836v1](https://arxiv.org/abs/2603.03836)。论文没有公开代码、数据和权重。

---

## 1. 先读这一节：这个复现是什么，不是什么

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

- 分组合看：box×stir 0.85、mug×stir 0.71 最高，cup×cake 0.24 最低。组合成绩受单技能上限约束，Cup 和 Cake 单独做时就只有 0.4–0.7。
- SkillVLA 的协作门在全部 15 个评测格上都关闭（开门比例 0.00），组合能力完全来自"关门时两臂结构上独立"。

### Table III 对应：协作实验（`results/cooperative.json`）

每个任务单独训练。括号内是 3 个种子各自的成功次数（满分 50）。

| 方法 | Shake | Ball | Align | 平均 | 论文平均 |
|---|---|---|---|---|---|
| Mono（π0.5-like） | 0.79（38/34/46） | 0.77（40/37/38） | 0.24（9/9/18） | 0.60 ± 0.06 | 0.47 |
| TwinVLA-like | 0.76（35/39/40） | 0.89（47/45/41） | 0.33（12/18/19） | 0.66 ± 0.02 | 0.42 |
| SkillVLA | 0.91（43/49/44） | 0.91（45/44/47） | 0.29（13/17/14） | **0.70** ± 0.02 | 0.48 |
| SkillVLA w/o Attn | 0.19（11/6/12） | 0.09（1/9/3） | 0.25（18/11/9） | **0.18** ± 0.02 | 0.17 |

- SkillVLA 在三个协作任务上的门开启比例都是 1.00。
- 去掉跨臂注意力后，Shake 和 Ball 大幅下降，Align 几乎不变。论文同样观察到 Align 对低层耦合不敏感。
- Align 的绝对水平偏低（0.24–0.33，论文 0.55–0.70），原因见第 5 节。
- SkillVLA 在 Shake 上高于基线，与论文相反，不宜过度解读：这里一个"种子"同时改变了数据、训练和评测场景（A-026），只有 3 个种子。

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
.venv/bin/python scripts/run.py --exp recomposition # 组合实验：9 个模型，约 2.3 小时
.venv/bin/python scripts/run.py --exp cooperative   # 协作实验：36 个模型，约 4.7 小时
.venv/bin/python scripts/check_results.py results/recomposition.json   # 检查协议 + 生成 .md 表格
./monitor.sh                                        # 实时终端面板（另开一个终端）
```

`run.py` 常用参数：
- `--seeds 0,1,2`、`--methods mono,skillvla`、`--steps 20000`、`--eval-n 50`、`--demos 50`；
- `--pilot`：结果写到 `out/pilot_*.json`，不写 `results/`；
- `--eval-only`：从 `out/ckpt/` 加载已保存的权重重新评测，不训练。

`monitor.sh` 参数：`-n 2`（刷新间隔秒数）、`--detail`、`--once`、`--log out/x.log`。

单个模型训练 20k 步的用时：Mono 约 5 分钟，Twin 约 7–15 分钟，SkillVLA 约 13–24 分钟。两个实验同时跑时会互相拖慢。

---

## 4. 代码地图与论文公式对照

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

论文公式在 `skillvla/models.py` 中的位置：

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
- 跨臂注意力插在 expert 的每一层，输出投影零初始化；
- 用 flow 单步速度误差代替 ‖â−a‖²，因为后者要做完整 10 步去噪，成本高且带采样噪声。

没有实现：连续门版本 L_coop = λ·(L_on − L_off)_sg·α_t。它在论文中只用于消融。

---

## 5. 必须知道的假设与已知问题

完整清单见 `implement-stage/ASSUMPTIONS.md`，共 37 条：30 条会影响结果含义（semantic），7 条影响接口（interface），其中 14 条来自跨模型审查。下面是最容易出错的几条：

- **A-003 高层是完美规则**：组合实验中 SkillVLA 永远拿到正确的每臂提示，0.46 是**低层解耦能力的上限**，对 SkillVLA 有利。门控在这里实际上等于给指令分类，没有检验回合内的模式切换。
- **A-017 Align 修订过两次**：
  - 原设计要对齐到画面中 1 像素的目标线，而且谁抓哪块积木由抛硬币决定，所有方法都接近 0。
  - 现在的定义：左臂取靠左的积木，两块积木彼此 y 差小于 0.05，两臂释放时间差不超过 3 步。
  - 修订的接受标准是事先定好的"Mono 能学会"，Mono 诊断达到 12/50。这个标准比较宽，所以 Align 的绝对水平低，失败主要来自一臂跳过抓取。
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
- **Cake 颜色 bug（已修复）**：蛋糕原来是粉色，和右臂末端的品红圆环太接近，模型分不清"桌上"和"已抓住"，SkillVLA 的 Cake 只有 4/50。现在所有物体与两臂颜色的最小 RGB 距离为 179.5。`out/pilot_*` 里的 pilot 结果用的是旧配色和旧 Align，**已作废**，只作参考。

**跨模型审查**：请 GPT-6-Astra（xhigh，只读）直接读代码，找没有写进账本的决定，共 2 轮，原始回复见 `SILENT_ASSUMPTION_SWEEP.json`。
- 第 1 轮：找出 12 条未声明的决定和 7 条陈旧条目。
- 第 2 轮：确认其中 17 项已解决，又发现 4 个问题。
- 全部已处理。**审查轮次已用完，第 2 轮之后的修复（最终版 Stir 判定、A-036、A-037）只经过执行者和测试验证，没有经过审查者复核。**

---

## 6. 不在仓库里的东西

| 内容 | 位置 | 说明 |
|---|---|---|
| 模型权重（49 个 .pt，约 800 MB） | 开发机 `skillvla_repro/out/ckpt/` | 正式实验 45 个：组合 9 个 + 协作 36 个；另有 4 个诊断用。`--eval-only` 依赖这些文件，新机器上要么重新训练，要么手动拷过去 |
| 虚拟环境 `.venv`（6.4 GB） | 开发机 | 按第 3 节重建 |
| 论文 HTML | 开发机 `implement-stage/paper.html` | arXiv 许可不允许我们再分发，请从 arXiv 获取 |
| 早期精读笔记与复现路线 | 开发机 `/home/mars/Desktop/sunianbing/reports/SkillVLA_2603.03836/` | 不属于本仓库 |

---

## 7. 未完成的工作与下一步

**推迟（本轮没做）**：
- π0-FAST 基线；
- 长程任务 Tubes / Collect Items，以及连续门与离散门的消融（Sec. VI-C）；
- 持续学习（Sec. VI-D）；
- 可学习的高层 planner；
- π0.5 / PaliGemma 骨干移植。

**建议的改进方向**（按我的优先级）：
1. **可学习的高层 planner**：改动小。把指令改写成多种自然语言说法，训练一个小模型从图像和指令生成 (u_L, u_R)，在未见组合上测分解准确率，看高层出错时 SkillVLA 会掉多少。
2. **换成真实骨干**：开发机上已有 lerobot 的 `pi05` 实现（conda 环境 `flowmatching`，lerobot 0.4.4）和缓存的 π0.5 权重（`~/.cache/huggingface/hub/models--lerobot--pi05_base`，14 GB）。
   - 用真 π0.5 做基线；
   - SkillVLA 共用一份冻结的 PaliGemma，两个低层各挂 LoRA，action expert 复制两份并加门控跨臂注意力。
   - 需要 224×224 的更真实画面，例如 RoboTwin。
   - 代价：单卡每次训练以小时计，整套实验是天级别。
3. **改进门控标签**：在单臂数据上 L_on ≈ L_off（例如 0.0730 对 0.0725），y = 1[L_on < L_off] 接近掷硬币，目前靠先验项压住才没有误开门。可以加一个 margin，或改成相对差。
4. 在把结果写成结论之前，先跑 `/experiment-audit` 和 `/result-to-claim`。

---

## 8. 注意事项

- 所有命令都在仓库根目录运行；脚本通过 `sys.path` 找到 `skillvla` 包。
- 开发机的系统盘只剩约 50 GB，存数据集和权重前先确认空间。
- 开发机的上级目录 `/home/mars/Desktop/sunianbing/` 里有一个损坏的空 `.git`。本仓库有自己的 `.git`，互不影响。
- `results/` 只放正式结果。`check_results.py` 会拒绝任何不符合声明协议的文件（方法、种子 0–2、eval_n 50、每任务 50 条示范、20k 步），也会拒绝任何带 `PLACEHOLDER` 的值。
- 构建过程的完整时间线（每个阶段的命令、退出码、修复次数、诊断结论）在 `implement-stage/BUILD_NOTE.md`。
