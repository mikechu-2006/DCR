"""From-scratch audio-style data2vec pretraining.

Usage (from data2vec_main/):
    python -m d2v_main.train --config configs/audio_style_base.yaml \
        --out-dir runs/base --ids-file splits/all_ids.txt

The code is single-process/single-GPU by design for the first run; DDP is a
straightforward extension (see README, "Not done yet").
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .charts import ChartStore
from .config import dump_config, load_config
from .data import ChartWindowDataset, collate_windows
from .masking import mask_ratio, span_mask
from .model import Data2VecModel, EMAState, cosine_lr
from .utils import git_rev, human_time, jsonl_append, repo_root, resolve_device, seed_all


def read_ids(path: str | Path) -> list[int]:
    ids = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.append(int(line))
    return ids


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--osu-dir", default=None, help="override data.osu_dir")
    ap.add_argument("--ids-file", default=None, help="one beatmap_id per line; default all .osu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (smoke/debug)")
    ap.add_argument("--limit-charts", type=int, default=0, help="use only the first N charts")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--index-cache", default=None, help="path to chart_index JSON")
    ap.add_argument("--no-index-cache", action="store_true",
                    help="scan .osu files instead of using the persistent index")
    return ap.parse_args(argv)


def build_loader(store, cfg, train: bool, batch_size: int, num_workers: int,
                 index_cache=None, use_index_cache: bool = True):
    ds = ChartWindowDataset(store, cfg, train=train, index_cache=index_cache,
                            use_index_cache=use_index_cache)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
        collate_fn=collate_windows, drop_last=train, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    ), ds


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.osu_dir:
        cfg["data"]["osu_dir"] = args.osu_dir
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.log_every is not None:
        cfg["train"]["log_every"] = args.log_every
    t = cfg["train"]
    seed_all(int(cfg["seed"]))
    device = resolve_device(args.device or str(t.get("device", "auto")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as fh:
        fh.write(dump_config(cfg))
    print(f"[run] out={out_dir} device={device} git={git_rev(repo_root())}", flush=True)

    ids = read_ids(args.ids_file) if args.ids_file else None
    if args.limit_charts:
        ids = (ids or ChartStore(cfg["data"]["osu_dir"]).ids)[: args.limit_charts]
    store = ChartStore(cfg["data"]["osu_dir"], ids=ids)
    print(f"[data] charts={len(store)}", flush=True)
    loader, ds = build_loader(store, cfg, True, int(t["batch_size"]), int(t["num_workers"]),
                              args.index_cache, not args.no_index_cache)
    print(f"[data] train windows/epoch={len(ds)} steps/epoch={len(loader)}", flush=True)

    model = Data2VecModel(cfg).to(device)
    ema = EMAState(model.teacher, enabled=bool(t.get("ema_fp32", True)))
    print(f"[model] {model.parameter_count()} ema_fp32={ema.enabled}", flush=True)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(t["lr"]), weight_decay=float(t["weight_decay"]), betas=(0.9, 0.98),
    )
    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * int(t["epochs"])
    warmup = max(1, int(total_steps * float(t["warmup_ratio"])))
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt.get("ema"))
        start_step = int(ckpt.get("step", 0))
        print(f"[run] resumed {args.resume} at step {start_step}", flush=True)
    ema_start = float(t["ema_start"])
    ema_end = float(t["ema_end"])
    ema_anneal = max(1, int(t["ema_anneal_steps"]))
    log_every = int(t["log_every"])
    save_every = int(t["save_every"])
    amp = bool(t.get("amp", True)) and device.type == "cuda"

    model.train()
    step = start_step
    t0 = time.time()
    running = 0.0
    running_n = 0
    for epoch in range(int(t["epochs"])):
        if args.max_steps and step >= args.max_steps:
            break
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            valid_in = batch["valid_len"].to(device, non_blocking=True)
            b, _, t_in = x.shape
            t_out = (t_in + 1) // 2
            valid_out = (valid_in + 1) // 2
            valid_mask = torch.arange(t_out, device=device)[None, :] < valid_out[:, None]
            gen = torch.Generator().manual_seed(int(cfg["seed"]) * 1000003 + step)
            mask = span_mask(valid_out.cpu(), t_out, float(cfg["model"]["mask_prob"]),
                             int(cfg["model"]["mask_length"]), generator=gen).to(device)
            mask = mask & valid_mask
            if not bool(mask.any()):
                continue
            lr = cosine_lr(step, total_steps, float(t["lr"]), warmup)
            for group in opt.param_groups:
                group["lr"] = lr
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                out, layers = model.forward_student(x, valid_mask, mask)
                with torch.no_grad():
                    _, t_layers = model.forward_teacher(x, valid_mask)
                    target = model.target(t_layers, mask)
                pred = model.predict(out, mask)
                loss = F.smooth_l1_loss(pred, target, beta=model.loss_beta, reduction="mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], float(t["grad_clip"]))
            opt.step()
            decay = ema_start + (ema_end - ema_start) * min(1.0, step / ema_anneal)
            ema.update(model.teacher, model.student, decay)
            running += float(loss.detach())
            running_n += 1
            if step % log_every == 0:
                with torch.no_grad():
                    tv = float(target.float().std(dim=0).mean())
                    pv = float(pred.float().std(dim=0).mean())
                    mr = float(mask_ratio(mask, valid_out))
                rec = {
                    "step": step, "epoch": epoch, "loss": running / max(1, running_n),
                    "lr": lr, "ema": decay, "mask_ratio": mr,
                    "target_var": tv, "pred_var": pv,
                    "elapsed": round(time.time() - t0, 1),
                }
                print(json.dumps(rec), flush=True)
                jsonl_append(out_dir / "metrics.jsonl", rec)
                running = 0.0
                running_n = 0
            if save_every > 0 and step > 0 and step % save_every == 0:
                save_ckpt(out_dir / "checkpoints" / f"step_{step:07d}.pt", model, opt, ema, step, cfg)
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
    save_ckpt(out_dir / "checkpoints" / "last.pt", model, opt, ema, step, cfg)
    elapsed = time.time() - t0
    print(f"[done] steps={step} elapsed={human_time(elapsed)} "
          f"steps_per_sec={step / max(1e-6, elapsed):.3f}", flush=True)
    if device.type == "cuda":
        print(f"[mem] max_allocated_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
    return 0


def save_ckpt(path: Path, model, opt, ema, step: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "ema": ema.state_dict(), "step": step, "cfg": cfg}, path)


if __name__ == "__main__":
    sys.exit(main())
