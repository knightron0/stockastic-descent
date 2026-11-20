"""One job of training for one model: 60 AdamW steps (15 minutes of bars) on d8, one learning rate per step."""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vendor.model_config import MODEL_CONFIGS
from vendor.modeling import AutoregressiveLM, initialize_model
from vendor.optimizers import build_optimizer

SEQ_LEN = 1024
BATCH_SIZE = 64
NUM_TRAIN_SEQS = 600_000


class Tokens:
    def __init__(self, data_dir):
        d = Path(data_dir)
        self.train = np.memmap(d / "train_600k.bin", dtype=np.uint16, mode="r").reshape(-1, SEQ_LEN)
        self.val = np.fromfile(d / "val.bin", dtype=np.uint16).reshape(-1, SEQ_LEN)

    def train_batch(self, step, device):
        # every model reads the same rows in the same order, so only the LR differs
        start = (step * BATCH_SIZE) % len(self.train)
        rows = np.take(self.train, range(start, start + BATCH_SIZE), axis=0, mode="wrap")
        return torch.from_numpy(rows.astype(np.int64)).to(device, non_blocking=True)


def lm_loss(model, ids):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids)
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))


@torch.no_grad()
def val_loss(model, tokens, device, bs=100):
    model.eval()
    losses = [lm_loss(model, torch.from_numpy(tokens.val[i:i + bs].astype(np.int64)).to(device)).item()
              for i in range(0, len(tokens.val), bs)]
    model.train()
    return float(np.mean(losses))


def new_model(device):
    torch.manual_seed(42)  # identical init for every run
    model = AutoregressiveLM(MODEL_CONFIGS["d8"])
    initialize_model(model)
    model = model.to(device)
    opt = build_optimizer(model, "adamw", 3e-3, weight_decay=0.1, beta1=0.9, beta2=0.95)
    return model, opt


def eval_points(start_step, n_steps, every=None):
    """Global steps in (start_step, start_step + n_steps] to run validation at: every multiple of `every`
    (None: no mid-job evals), plus always the job's last step."""
    end = start_step + n_steps
    mid = set(range((start_step // every + 1) * every, end, every)) if every else set()
    return sorted(mid | {end}) if n_steps > 0 else []


def train_window(name, mults, schedule, tokens, ckpt_dir, run_eval, eval_every=None, device="cuda"):
    """Load `name`'s checkpoint, run len(mults) steps with lr = schedule(step, mult, state), and save to
    <name>.pt.tmp (call commit_checkpoints after the DB write succeeds). The schedule's state (e.g. the
    envelope's rise cap) lives in the checkpoint, so it carries across jobs. Returns per-step records.
    With run_eval, validation runs at eval_points(): mid-job on the weights at that step, and at the end."""
    from stocklr.lr import TOTAL_STEPS
    t0 = time.perf_counter()
    path = Path(ckpt_dir) / f"{name}.pt"
    model, opt = new_model(device)
    state = {"step": 0, "hour": 0}                    # "hour" counts jobs (old name, kept for old checkpoints)
    if path.exists():
        ck = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); state = ck["state"]
    load_s = time.perf_counter() - t0
    start_step = state["step"]
    if start_step >= TOTAL_STEPS:
        return {"name": name, "done": True, "start_step": start_step, "steps": [], "evals": [], "val_loss": None}

    mults = [float(m) for m in mults][: TOTAL_STEPS - start_step]
    sched_state = dict(state.get("sched", {}))
    torch.cuda.synchronize(); t1 = time.perf_counter()
    losses = torch.empty(len(mults), device=device)
    lrs, logged = [], []
    mid_evals = set(eval_points(start_step, len(mults), eval_every)[:-1]) if run_eval else set()
    evals, eval_s = [], 0.0                            # (global step, val loss); eval time kept out of train_s
    for i, m in enumerate(mults):
        step = start_step + i + 1                      # 1-based global step
        lr, shown = schedule(step, m, sched_state)
        if not (np.isfinite(lr) and lr >= 0):
            raise ValueError(f"{name}: schedule produced lr={lr} at step {step}")
        lrs.append(lr); logged.append(shown)
        for g in opt.param_groups:
            g["lr"] = lr
        loss = lm_loss(model, tokens.train_batch(step - 1, device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses[i] = loss.detach()
        if step in mid_evals:                          # weights as of this step; training resumes after
            te = time.perf_counter()
            evals.append((step, val_loss(model, tokens, device)))
            eval_s += time.perf_counter() - te
    torch.cuda.synchronize(); train_s = time.perf_counter() - t1 - eval_s
    losses = losses.tolist()
    if not all(map(np.isfinite, losses)):
        raise FloatingPointError(f"{name}: non-finite training loss; checkpoint not saved")

    vl = val_loss(model, tokens, device) if run_eval else None
    if vl is not None:
        evals.append((start_step + len(mults), vl))
    state["step"] = start_step + len(mults)
    state["sched"] = sched_state
    state["hour"] = state.get("hour", 0) + 1
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state}, f"{path}.tmp")
    return {"name": name, "done": False, "start_step": start_step, "end_step": state["step"],
            "steps": [(start_step + i + 1, lrs[i], logged[i], losses[i]) for i in range(len(mults))],
            "evals": evals, "val_loss": vl, "load_s": load_s, "train_s": train_s, "ms_per_step": 1000 * train_s / max(1, len(mults)),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}


def commit_checkpoints(ckpt_dir, names):
    import os
    for n in names:
        tmp = Path(ckpt_dir) / f"{n}.pt.tmp"
        if tmp.exists():
            os.replace(tmp, Path(ckpt_dir) / f"{n}.pt")
