# SPEC — SkillVLA (arXiv:2603.03836v1) 复现

**Request（原文）**：复现 https://arxiv.org/html/2603.03836v1，没有官方代码，在当前 workspace 中新建一个路径专门用于本次复现项目，在复现完成后再进行下一步改进。

**Target**：`skillvla_repro/` 下一个自包含的**机制级仿真复现**。内容包括：平面双臂仿真器（每臂 3 个单臂技能，另有 3 个协作任务）、脚本专家示范、4 种策略（Mono/π0.5-like、TwinVLA-like、SkillVLA、SkillVLA w/o Attn）、训练与闭环评测脚本。评测输出论文 Table I/II/III 的仿真对应表。

**Inputs**：无外部数据。所有示范由 `skillvla/env` 中的脚本专家生成，随机种子固定。

**Outputs**：
- `results/recomposition.json`：Table I（9 个未见组合）+ Table II（6 个已见技能），逐格给出 `{success, n}`
- `results/cooperative.json`：Table III（Shake/Ball/Align × 方法）
- `results/*.md`：人可读表格
- 冒烟产物只写 `out/*_smoke.json`，不进入 `results/`

**Success command**（spine）：`.venv/bin/python scripts/run.py --smoke && test -f out/smoke.json`

**Base commit**：none (not a git repo)。`/home/mars/Desktop/sunianbing/.git` 已损坏，`git rev-parse` 报 not a git repository。按技能规则不初始化 git。

**Scope cuts**：
- 不用真机；不用 π0.5/PaliGemma 3B 骨干（见 A-001）
- 高层 VLM 用规则分解代替（A-003）
- Qwen3-VL 协作先验用规则标签代替（A-006）
- 不做 π0-FAST 基线（DEFERRED）
- 不做长程任务 VI-C、持续学习 VI-D（DEFERRED）
- 完成复现后再做"改进"，不在本次运行内
