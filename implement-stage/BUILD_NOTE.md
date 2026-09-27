# Build Note — SkillVLA 仿真机制级复现

| Rung | Feature | Acceptance check (ONE command) | Tier | Status |
|------|---------|-------------------------------|------|--------|
| F0 | spine: 指令 → 示范生成 → 训练 → 闭环评测 → `out/smoke.json`（内部 stub） | `.venv/bin/python scripts/run.py --smoke && test -f out/smoke.json` | MUST | ✅ |
| F1 | 真实双臂仿真：6 个单臂技能 + 9 组合 + 3 协作任务 + 脚本专家 + 成功判定 | `.venv/bin/pytest -q tests/test_env.py` | MUST | ✅ |
| F2 | 真实模型：Mono / TwinVLA-like / SkillVLA（分臂 VLM、门控跨臂注意力、协作估计器及 L_disc/L_prior/L_sticky/L_sup）/ w/o Attn | `.venv/bin/pytest -q tests/test_models.py` | MUST | ✅ |
| F3 | 组合实验（Table I / II 对应） | `.venv/bin/python scripts/run.py --exp recomposition && .venv/bin/python scripts/check_results.py results/recomposition.json` | MUST | ✅ |
| F4 | 协作实验（Table III 对应，含 w/o Attn 消融） | `.venv/bin/python scripts/run.py --exp cooperative && .venv/bin/python scripts/check_results.py results/cooperative.json` | MUST | ✅ |

## Run record
<!-- one line per rung attempt: command, exit code, artifact, fix attempts used -->
- F0 `.venv/bin/python scripts/run.py --smoke && test -f out/smoke.json` → exit 0, `out/smoke.json`（PLACEHOLDER_ 前缀），修复 1 次（OneCycleLR 在 20 步时 warmup 除零）。live stubs：env reach 技能（F1 退役）、StubPolicy（F2 退役）
- F1 `.venv/bin/pytest -q tests/test_env.py` → exit 0（29 passed）；修复 1 次（box 推动模型：接触半径离散化导致 EE 穿过箱子，改为沿运动方向拖动）。F0 复跑 exit 0。env reach stub 已退役
- F2 `.venv/bin/pytest -q tests/test_models.py` → exit 0（10 passed）；修复 2 次（w/o-Attn 层无 xattn 模块；BCE 在 bf16 autocast 下不可用 → 门控损失在 fp32 计算）。F0 改用 20 步 SkillVLA（产物仍为 PLACEHOLDER_ 前缀），F0/F1 复跑 exit 0（全部 39 passed）。StubPolicy 已删除
- F3 pilot（seed 0，Mono+SkillVLA）：未见组合 SkillVLA 250/450、Mono 0/450。异常：SkillVLA R_cake 4/50。`scripts/diagnose.py` 诊断：19/20 回合右臂直接飞到容器上方悬停，从未抓取。原因判断为渲染混淆：cake 颜色 (255,150,200) 与右臂末端品红环 (255,0,255) 过近，被抓物体又画在末端环下方，导致"桌上的蛋糕"和"已抓住"难以区分。修复：cake→白、nut→琥珀、blockA→红，使所有物体与两臂颜色的 RGB 距离 ≥180。此修复对所有方法相同；pilot 中涉及 Cake 的格子作废，以正式实验为准
- F4 pilot（seed 0，旧配色）：Shake Mono 38 / Twin 39 / SkillVLA 43（门 1.00）/ w/o Attn 11；Ball 40 / 45 / 45（门 1.00）/ 1；**Align 1 / 1 / 2 / 0**。诊断（Mono、新配色，3/20）：放置 y 与画面中 1px 目标线基本无关（如目标 0.39 放在 0.57），另有约一半回合只有一臂抓取。修订 Align 为论文定义"两块彼此共线"（见 A-017 Notes），去掉外部目标线；分配随机性保留。协作 pilot 作废，以正式实验为准
- Align 第 1 次修订（去目标线、隐藏随机高度）：Mono 诊断 0/20（一臂犹豫）。第 2 次修订（几何分配：左臂取靠左块；高度 = 两块均值；容差 0.05）：Mono 诊断 12/50。按事先固定的标准（Mono 可学会）接受，不再调整。env 测试 30 passed
- F3 正式 `run.py --exp recomposition && check_results.py results/recomposition.json` → exit 0（3 方法 × 3 种子 × 15 格 × 50 回合，约 2h20m）。未见组合 SkillVLA 0.49±0.10、Mono 0.00±0.01、Twin 0.00±0.00；已学 0.84 / 0.80 / 0.76。修复 1 次（表头标签把 box 里的 x 替换成 ×）
- Phase 4 sweep 第 1 轮后：修复 Stir 成功判定（绝对转角 → 带符号净转角）并新增测试（来回晃动不算搅拌；已确认旧判定下该测试会失败）；`check_results.py` 强制声明的协议；诊断和冒烟先设种子；`run.py --eval-only`。组合实验用 9 个保存的 checkpoint 重新评测，只有 9 个 Stir 相关格子变化，SkillVLA 未见组合 0.47±0.10，基线 0.00。`check_results.py` exit 0；测试 41 passed。F0 复跑 exit 0
- Phase 4 sweep 第 2 轮后：Stir 改为环带 + 单步 <90°，新增显式轨迹测试（旧判定会失败）；第二次 `--eval-only` 重新评测，与原始结果相比仅 9 个 Stir 格子变化；SkillVLA 未见组合 0.46±0.10，Mono 0.00±0.01，Twin 0.00±0.00；`check_results.py` exit 0

- F4 正式 `run.py --exp cooperative && check_results.py results/cooperative.json` → exit 0（4 方法 × 3 任务 × 3 种子 = 36 个模型，每格 50 回合）。Table III 对应：Mono 0.60±0.06、Twin 0.66±0.02、SkillVLA 0.70±0.02（门全开）、w/o Attn 0.18±0.02。审查者确认该进程启动于 Stir 修改之前，且 Stir 不涉及协作任务。最终复跑：F0 exit 0、测试 42 passed、F3/F4 检查 exit 0

## Deferred
- **π0-FAST 基线**：自回归离散 token VLA；本次 rung 预算 5，Table I 中它与 π0.5 同为 0%，优先级最低。
- **VI-C 长程任务（Tubes / Collect Items）与连续/离散门消融**：需要多阶段环境与高层刷新逻辑，超出 5 rung 预算。
- **VI-D 持续学习**：需要"单臂预训练 → 少样本双臂微调"两阶段协议，放到下一轮。
- **π0.5 / PaliGemma 骨干移植（lerobot pi05 + RoboTwin）**：见 A-001，是"下一步改进/放大"的首选方向。
- **可学习的高层 planner**：见 A-003。

## Blockers
