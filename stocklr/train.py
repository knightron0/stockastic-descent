"""One hour of training for one model: 240 AdamW steps on d8, one learning rate per step."""
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


def train_hour(name, mults, tokens, ckpt_dir, run_eval, device="cuda"):
    """Load `name`'s checkpoint, run len(mults) steps with lr = ETA0 * wsd(step) * mult, and save to
    <name>.pt.tmp (call commit_checkpoints after the DB write succeeds). Returns per-step records."""
    from stocklr.lr import ETA0, TOTAL_STEPS, wsd
    t0 = time.perf_counter()
    path = Path(ckpt_dir) / f"{name}.pt"
    model, opt = new_model(device)
    state = {"step": 0, "hour": 0}
    if path.exists():
        ck = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); state = ck["state"]
    load_s = time.perf_counter() - t0
    start_step = state["step"]
    if start_step >= TOTAL_STEPS:
        return {"name": name, "done": True, "start_step": start_step, "steps": [], "val_loss": None}

    mults = [float(m) for m in mults][: TOTAL_STEPS - start_step]
    torch.cuda.synchronize(); t1 = time.perf_counter()
    losses = torch.empty(len(mults), device=device)
    lrs = []
    for i, m in enumerate(mults):
        step = start_step + i + 1                      # 1-based global step
        lr = ETA0 * wsd(step) * m
        lrs.append(lr)
        for g in opt.param_groups:
            g["lr"] = lr
        loss = lm_loss(model, tokens.train_batch(step - 1, device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses[i] = loss.detach()
    torch.cuda.synchronize(); train_s = time.perf_counter() - t1
    losses = losses.tolist()
    if not all(map(np.isfinite, losses)):
        raise FloatingPointError(f"{name}: non-finite training loss; checkpoint not saved")

    vl = val_loss(model, tokens, device) if run_eval else None
    state["step"] = start_step + len(mults)
    state["hour"] = state.get("hour", 0) + 1
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state}, f"{path}.tmp")
    return {"name": name, "done": False, "start_step": start_step, "end_step": state["step"],
            "steps": [(start_step + i + 1, lrs[i], mults[i], losses[i]) for i in range(len(mults))],
            "val_loss": vl, "load_s": load_s, "train_s": train_s, "ms_per_step": 1000 * train_s / max(1, len(mults)),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}


def commit_checkpoints(ckpt_dir, names):
    import os
    for n in names:
        tmp = Path(ckpt_dir) / f"{n}.pt.tmp"
        if tmp.exists():
            os.replace(tmp, Path(ckpt_dir) / f"{n}.pt")
