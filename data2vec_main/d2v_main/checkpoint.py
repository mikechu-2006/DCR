"""Checkpoint helpers for expanding the conv1 input channels (8 -> C).

The only tensor shape that changes when the front-end gains attribute channels
is conv.conv1.weight: [hidden, in_channels, k].  The projection is
function-preserving if the new input-column weights are zero, so a continued
pretraining run from an 8-channel checkpoint starts bit-identical to the old
model (data2vec_main/README.md, route-B notes; ControlNet zero-conv style).

EMA needs no special handling: EMAState holds a master copy of the
TransformerStack only, and the conv is a single module shared by the student
and the EMA teacher (model.py::_features).

Also provides infer_in_channels(), which lets extract / E1 / score-only paths
recover the channel count from a checkpoint instead of guessing from a YAML.
"""
from __future__ import annotations

import copy

import torch


def infer_in_channels(ckpt_or_state: dict) -> int:
    """Return the conv1 input-channel count recorded by a checkpoint or state dict."""
    sd = ckpt_or_state.get("model", ckpt_or_state) if isinstance(ckpt_or_state, dict) else ckpt_or_state
    key = "conv.conv1.weight"
    if key in sd:
        return int(sd[key].shape[1])
    cfg = ckpt_or_state.get("cfg") if isinstance(ckpt_or_state, dict) else None
    if cfg:
        return int(cfg["model"]["in_channels"])
    raise KeyError("cannot infer in_channels: no conv.conv1.weight and no ckpt cfg")


def expand_model_state(sd: dict, new_in: int) -> dict:
    """Zero-init new input columns of conv1 (function-preserving at step 0)."""
    sd = dict(sd)
    key = "conv.conv1.weight"
    if key not in sd:
        raise KeyError(key)
    w = sd[key]
    old_in = int(w.shape[1])
    if new_in == old_in:
        return sd
    if new_in < old_in:
        raise ValueError("cannot shrink conv1: %d -> %d" % (old_in, new_in))
    nw = torch.zeros((w.shape[0], int(new_in), w.shape[2]), dtype=w.dtype, device=w.device)
    nw[:, :old_in] = w
    sd[key] = nw
    return sd


def expand_train_ckpt(ckpt: dict, new_in: int) -> dict:
    """Expand model state and AdamW moments of conv1.weight; keep step/other state."""
    ckpt = copy.copy(ckpt)
    sd = ckpt["model"]
    key = "conv.conv1.weight"
    if key not in sd:
        raise KeyError(key)
    old_shape = tuple(sd[key].shape)
    old_in = int(old_shape[1])
    if new_in == old_in:
        return ckpt
    if new_in < old_in:
        raise ValueError("cannot shrink conv1: %d -> %d" % (old_in, new_in))
    ckpt["model"] = expand_model_state(sd, new_in)
    new_shape = (old_shape[0], int(new_in), old_shape[2])
    opt = ckpt.get("optimizer")
    if isinstance(opt, dict) and "state" in opt:
        for pid, st in list(opt["state"].items()):
            if not isinstance(st, dict):
                continue
            for k in ("exp_avg", "exp_avg_sq"):
                v = st.get(k)
                if isinstance(v, torch.Tensor) and tuple(v.shape) == old_shape:
                    nz = torch.zeros(new_shape, dtype=v.dtype, device=v.device)
                    nz[:, :old_in] = v
                    st[k] = nz
    cfg = ckpt.get("cfg")
    if isinstance(cfg, dict):
        cfg.setdefault("model", {})["in_channels"] = int(new_in)
    return ckpt


def load_model_state_expanded(model, ckpt: dict, expand: bool = True) -> dict:
    """Load ckpt['model'] into model, optionally expanding conv1 channels.

    Returns a small info dict: old_in, new_in, expanded.
    """
    sd = ckpt.get("model", ckpt)
    new_in = int(model.conv.conv1.in_channels)
    old_in = int(sd["conv.conv1.weight"].shape[1])
    if old_in != new_in:
        if not expand:
            raise RuntimeError(
                "checkpoint in_channels=%d but model expects %d; pass --expand-channels"
                % (old_in, new_in))
        sd = expand_model_state(sd, new_in)
    model.load_state_dict(sd, strict=True)
    return {"old_in": old_in, "new_in": new_in, "expanded": old_in != new_in}
