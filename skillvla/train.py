import time

import torch


def train(policy, dataset, steps, batch_size, lr, seed, log_every=500, log=print):
    gen = torch.Generator(device=dataset.t["image"].device).manual_seed(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=max(0.05, 2 / steps))
    policy.train()
    t0, hist = time.time(), []
    for step in range(1, steps + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = policy.loss(dataset.sample(batch_size, gen), gen)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            rec = {k: float(v) for k, v in out.items()}
            rec["step"] = step
            hist.append(rec)
            log(f"  step {step}/{steps} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "step")
                + f" ({time.time() - t0:.0f}s)")
    policy.eval()
    return hist
