#!/usr/bin/env python3
"""Export HomieRL_mjlab HIM ``model_*.pt`` → single-input ONNX for sim2sim / deploy.

ONNX contract (matches HomieDeploy + kuavo RL controllers):
  input:  float32 [B, hist_len * one_step]   e.g. phase1 [1, 480]
  output: float32 [B, num_actions]           e.g. phase1 [1, 12]

Usage:
  python scripts/export_him_onnx.py \\
    --checkpoint .../model_11000.pt \\
    --output .../g1_phase1.onnx \\
    --num-actions 12 --one-step 80 --hist-len 6
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_src = os.path.join(here, "..", "src")
    mjlab_src = os.path.expanduser("~/OpenHomie/HomieRL_mjlab/src")
    for p in (pkg_src, mjlab_src):
        p = os.path.abspath(p)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


class HimOnnxExporter(nn.Module):
    """Estimator encoder + actor, history → action mean (Homie PolicyExporterHIM)."""

    def __init__(self, actor_critic: nn.Module, one_step: int):
        super().__init__()
        self.actor = actor_critic.actor
        self.estimator = actor_critic.estimator.encoder
        self.one_step = int(one_step)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        parts = self.estimator(obs_history)
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        return self.actor(torch.cat((obs_history[:, -self.one_step :], vel, z), dim=-1))


def main() -> None:
    _ensure_paths()
    from homie_ros_sim2sim.him_loader import load_him_policy

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-actions", type=int, default=12)
    ap.add_argument("--one-step", type=int, default=80)
    ap.add_argument("--hist-len", type=int, default=6)
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    ac, one_step = load_him_policy(
        args.checkpoint,
        num_actions=args.num_actions,
        num_one_step_obs=args.one_step,
        actor_history_length=args.hist_len,
        device="cpu",
    )
    assert one_step == args.one_step

    exporter = HimOnnxExporter(ac, one_step=args.one_step).eval()
    hist_dim = args.hist_len * args.one_step
    dummy = torch.zeros(1, hist_dim, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    torch.onnx.export(
        exporter,
        dummy,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    )

    # Sanity: ORT vs torch
    with torch.no_grad():
        ref = exporter(dummy).numpy()
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        out = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]
        err = float(abs(ref - out).max())
        print(f"ORT max abs err vs torch: {err:.6g}")
    except ImportError:
        print("onnxruntime not installed; skipped ORT check")

    print(f"Exported ONNX: {args.output}")
    print(f"  input  obs      [B, {hist_dim}]")
    print(f"  output actions [B, {args.num_actions}]")


if __name__ == "__main__":
    main()
