Experiment `cooperative` — 3 training seeds × 50 episodes per cell; 50 demos per task; 20000 training steps. Cell = mean success rate over seeds; Avg = mean ± std over seeds of the per-seed average.

### Table III analogue — cooperative tasks

| Method | shake | ball | align | Avg. |
|---|---|---|---|---|
| Mono (π0.5-like) | 0.79 | 0.77 | 0.24 | **0.60** ± 0.06 |
| TwinVLA-like | 0.76 | 0.89 | 0.33 | **0.66** ± 0.02 |
| SkillVLA | 0.91 | 0.91 | 0.29 | **0.70** ± 0.02 |
| SkillVLA w/o Attn. | 0.19 | 0.09 | 0.25 | **0.18** ± 0.02 |

SkillVLA mean gate-on fraction during rollouts — shake: 1.00, ball: 1.00, align: 1.00
