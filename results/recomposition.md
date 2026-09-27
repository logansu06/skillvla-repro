Experiment `recomposition` — 3 training seeds × 50 episodes per cell; 50 demos per task; 20000 training steps. Cell = mean success rate over seeds; Avg = mean ± std over seeds of the per-seed average.

### Table I analogue — unseen skill recompositions (zero-shot)

| Method | cup×cake | cup×stir | cup×smash | box×cake | box×stir | box×smash | mug×cake | mug×stir | mug×smash | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|
| Mono (π0.5-like) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 0.00 | **0.00** ± 0.01 |
| TwinVLA-like | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** ± 0.00 |
| SkillVLA | 0.24 | 0.41 | 0.31 | 0.41 | 0.85 | 0.29 | 0.38 | 0.71 | 0.57 | **0.46** ± 0.10 |

### Table II analogue — trained single-arm skills

| Method | cup | box | mug | cake | stir | smash | Avg. |
|---|---|---|---|---|---|---|---|
| Mono (π0.5-like) | 0.43 | 0.95 | 0.97 | 0.53 | 0.99 | 0.93 | **0.80** ± 0.06 |
| TwinVLA-like | 0.38 | 0.96 | 0.95 | 0.37 | 0.97 | 0.93 | **0.76** ± 0.08 |
| SkillVLA | 0.49 | 0.94 | 0.94 | 0.69 | 1.00 | 0.96 | **0.84** ± 0.03 |

SkillVLA mean gate-on fraction during rollouts — L_cup: 0.00, L_box: 0.00, L_mug: 0.00, R_cake: 0.00, R_stir: 0.00, R_smash: 0.00, cupxcake: 0.00, cupxstir: 0.00, cupxsmash: 0.00, boxxcake: 0.00, boxxstir: 0.00, boxxsmash: 0.00, mugxcake: 0.00, mugxstir: 0.00, mugxsmash: 0.00
