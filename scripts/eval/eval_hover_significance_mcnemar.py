#!/usr/bin/env python3
"""Per-GT hit@0.5: video_native vs V1 → hover Q20 → McNemar + bootstrap CI."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_spec_ev = importlib.util.spec_from_file_location(
    "eval_yolomg_val_photo_video_metrics", _SCRIPTS / "eval_yolomg_val_photo_video_metrics.py"
)
_evpv = importlib.util.module_from_spec(_spec_ev)
assert _spec_ev.loader is not None
_spec_ev.loader.exec_module(_evpv)

load_val_pairs = _evpv.load_val_pairs
run_inference_batch = _evpv.run_inference_batch
box_iou_xyxy = _evpv.box_iou_xyxy
label_path_from_rgb = _evpv.label_path_from_rgb
yolo_txt_to_xyxy = _evpv.yolo_txt_to_xyxy

_orig_torch_load = torch.load


def _torch_load_weights(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights


def fd_energy_bbox(fd_bgr: np.ndarray, box_xyxy: np.ndarray) -> float:
    h, w = fd_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.mean(np.abs(fd_bgr[y1:y2, x1:x2].astype(np.float32))))


def collect_video_gt_energy(pairs, photo_marker: str, max_samples: int = 0):
    records, all_e = [], []
    n_seen = 0
    for ipath, ipath2 in pairs:
        if photo_marker in str(ipath).replace("\\", "/"):
            continue
        if max_samples > 0 and n_seen >= max_samples:
            break
        im0 = cv2.imread(str(ipath))
        im02 = cv2.imread(str(ipath2))
        if im0 is None or im02 is None:
            continue
        n_seen += 1
        h0, w0 = im0.shape[:2]
        gt = yolo_txt_to_xyxy(label_path_from_rgb(ipath), w0, h0)
        if len(gt) == 0:
            continue
        for gi, box in enumerate(gt):
            e_b = fd_energy_bbox(im02, box)
            all_e.append(e_b)
            records.append(
                {
                    "path": str(ipath),
                    "path2": str(ipath2),
                    "hw": [h0, w0],
                    "gt_index": gi,
                    "box_xyxy": box.tolist(),
                    "e_bbox": e_b,
                }
            )
    return records, all_e


def best_iou_per_gt(gt, pred_xyxy, pred_conf):
    n_gt = len(gt)
    if n_gt == 0:
        return [], []
    if pred_xyxy.shape[0] == 0:
        return [0.0] * n_gt, [0.0] * n_gt
    ious = box_iou_xyxy(gt, pred_xyxy)
    best_iou, best_cf = [], []
    for gi in range(n_gt):
        row = ious[gi]
        j = int(np.argmax(row))
        mx = float(row[j])
        js = np.where(row >= mx - 1e-6)[0]
        j_best = int(js[np.argmax(pred_conf[js])])
        best_iou.append(float(row[j_best]))
        best_cf.append(float(pred_conf[j_best]))
    return best_iou, best_cf


@dataclass(frozen=True)
class GateConfig:
    name: str
    mode: str


def set_fd_gate_off(model) -> None:
    from models.common import Concat3, Concat3Adaptive

    for m in model.modules():
        if isinstance(m, (Concat3, Concat3Adaptive)):
            if hasattr(m, "_ensure_concat3_ckpt_attrs"):
                m._ensure_concat3_ckpt_attrs()
            m.fd_gate_mode = "off"


def run_model_hits(records, weights: Path, label: str, imgsz, conf_nms, iou_nms, device: str):
    from models.experimental import attempt_load
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_coords

    os.environ["YOLOMG_CONCAT3_FD_GATE"] = "off"
    dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    wpath = weights if weights.is_absolute() else _ROOT / weights
    model = attempt_load(str(wpath), map_location=dev)
    model.eval()
    set_fd_gate_off(model)
    stride = int(model.stride.max())

    unique_frames = {}
    for rec in records:
        if rec["path"] not in unique_frames:
            unique_frames[rec["path"]] = {"path2": rec["path2"], "hw": tuple(rec["hw"])}

    cache = {}
    t0 = time.perf_counter()
    for fi, (p, meta) in enumerate(unique_frames.items()):
        im0 = cv2.imread(p)
        im02 = cv2.imread(meta["path2"])
        if im0 is None or im02 is None:
            cache[p] = (np.zeros((0, 4)), np.zeros(0))
            continue
        h0, w0 = meta["hw"]
        im_lb, _, _ = letterbox(im0, (imgsz, imgsz), auto=True, stride=stride)
        im_lb2, _, _ = letterbox(im02, (imgsz, imgsz), auto=True, stride=stride)
        pred_xyxy, pred_conf = run_inference_batch(
            model, im_lb, im_lb2, (h0, w0), dev, conf_nms, iou_nms,
            non_max_suppression, scale_coords,
        )
        cache[p] = (pred_xyxy, pred_conf)
        if (fi + 1) % 400 == 0:
            print(f"  [{label}] {fi + 1}/{len(unique_frames)} {time.perf_counter() - t0:.0f}s", flush=True)

    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    iou_by_key = {}
    for rec in records:
        key = f"{rec['path']}#{rec['gt_index']}"
        pred_xyxy, pred_conf = cache[rec["path"]]
        gt_one = np.array([rec["box_xyxy"]], dtype=np.float32)
        biou, _ = best_iou_per_gt(gt_one, pred_xyxy, pred_conf)
        iou_by_key[key] = biou[0]
    print(f"Done {label}: {len(unique_frames)} frames, {len(iou_by_key)} GT, {time.perf_counter() - t0:.0f}s", flush=True)
    return iou_by_key


def mcnemar_test(h0: np.ndarray, h1: np.ndarray) -> dict:
    b01 = int(np.sum((~h0) & h1))
    b10 = int(np.sum(h0 & (~h1)))
    discordant = b01 + b10
    out = {
        "n_pairs": len(h0),
        "b01_miss_native_hit_v1": b01,
        "b10_hit_native_miss_v1": b10,
        "discordant": discordant,
    }
    if discordant == 0:
        out.update({"p_value": 1.0, "method": "mcnemar_no_discordant"})
        return out
    try:
        from scipy.stats import binomtest

        out["p_value"] = float(binomtest(min(b01, b10), discordant, 0.5, alternative="two-sided").pvalue)
        out["method"] = "mcnemar_exact_binomial"
    except ImportError:
        chi2 = (abs(b01 - b10) - 1) ** 2 / discordant
        from math import erfc, sqrt

        out["p_value"] = erfc(sqrt(chi2 / 2))
        out["method"] = "mcnemar_chi2_approx"
    return out


def bootstrap_delta(h0, h1, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(h0)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = h1[idx].mean() - h0[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_boot": n_boot,
        "seed": seed,
        "delta_mean": float(h1.mean() - h0.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_boot_le0": float(np.mean(deltas <= 0)),
    }


def summarize(subset, name: str, n_boot: int, seed: int) -> dict:
    if not subset:
        return {"name": name, "n_gt": 0}
    h0 = np.array([r["hit_native"] for r in subset], dtype=bool)
    h1 = np.array([r["hit_v1"] for r in subset], dtype=bool)
    return {
        "name": name,
        "n_gt": len(subset),
        "hit_native": float(h0.mean()),
        "hit_v1": float(h1.mean()),
        "delta_hit_v1_minus_native": float(h1.mean() - h0.mean()),
        "mcnemar": mcnemar_test(h0, h1),
        "bootstrap": bootstrap_delta(h0, h1, n_boot, seed),
    }


def write_md(path: Path, q20: float, h: dict) -> None:
    m, b = h.get("mcnemar", {}), h.get("bootstrap", {})
    pval = m.get("p_value", 1.0)
    sig = pval < 0.05 and h.get("delta_hit_v1_minus_native", 0) > 0
    interp = (
        "Различие по hover **статистически значимо** (McNemar p<0.05); 95% CI для Δ не включает 0."
        if sig and b.get("ci95_low", 0) > 0
        else "См. p-value и CI ниже."
    )
    path.write_text(
        "\n".join(
            [
                "# Hover significance: video_native vs V1 (per-GT)",
                "",
                f"- **n_gt hover Q20:** {h.get('n_gt', 0)} (e_bbox ≤ {q20:.4g})",
                f"- **hit@0.5 video_native:** {h.get('hit_native', 0):.4f}",
                f"- **hit@0.5 V1:** {h.get('hit_v1', 0):.4f}",
                f"- **Δ (V1 − native):** {h.get('delta_hit_v1_minus_native', 0)*100:+.2f} п.п.",
                "",
                "## McNemar",
                f"- discordant: {m.get('discordant', 0)} (b01={m.get('b01_miss_native_hit_v1')}, b10={m.get('b10_hit_native_miss_v1')})",
                f"- **p-value** ({m.get('method', '')}): **{pval:.4g}**",
                "",
                "## Bootstrap Δ (10k)",
                f"- mean Δ: **{b.get('delta_mean', 0)*100:+.2f} п.п.**",
                f"- **95% CI:** [{b.get('ci95_low', 0)*100:+.2f}, {b.get('ci95_high', 0)*100:+.2f}] п.п.",
                "",
                f"## Интерпретация\n\n{interp}\n",
            ]
        ),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-txt", type=Path, default=Path("/home/tanyadiplom/data/ard100/val.txt"))
    ap.add_argument("--val2-txt", type=Path, default=Path("/home/tanyadiplom/data/ard100/val2.txt"))
    ap.add_argument("--weights-native", type=Path,
                    default=_ROOT / "runs/train/ard100_640_b64_e30_p7_video_native/weights/best.pt")
    ap.add_argument("--weights-v1", type=Path,
                    default=_ROOT / "runs/train/ard100_videoOnly_adaptiveFusion_b56_e30_pat7_from_videoNative_physgpu0/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-nms", type=float, default=0.001)
    ap.add_argument("--iou-nms", type=float, default=0.4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path,
                    default=_ROOT / "experiments/results/block_I__val__video_only")
    args = ap.parse_args()

    pairs = load_val_pairs(args.val_txt, args.val2_txt)
    print(f"Val pairs: {len(pairs)}", flush=True)
    records, all_e = collect_video_gt_energy(pairs, "/mixed_photos/")
    q20 = float(np.quantile(np.array(all_e, dtype=np.float64), 0.20))
    print(f"GT: {len(records)}; hover Q20 e_bbox≤{q20:.4g}", flush=True)

    iou_native = run_model_hits(records, args.weights_native, "video_native",
                                args.imgsz, args.conf_nms, args.iou_nms, args.device)
    iou_v1 = run_model_hits(records, args.weights_v1, "V1",
                            args.imgsz, args.conf_nms, args.iou_nms, args.device)

    per_gt = []
    for rec in records:
        key = f"{rec['path']}#{rec['gt_index']}"
        ia, ib = iou_native[key], iou_v1[key]
        per_gt.append({
            "gt_key": key,
            "e_bbox": rec["e_bbox"],
            "hover_q20": rec["e_bbox"] <= q20,
            "best_iou_native": ia,
            "best_iou_v1": ib,
            "hit_native": ia >= 0.5,
            "hit_v1": ib >= 0.5,
        })

    hover_sub = [r for r in per_gt if r["hover_q20"]]
    summaries = {
        "all_gt": summarize(per_gt, "all_gt", args.n_boot, args.seed),
        "hover_q20": summarize(hover_sub, "hover_q20", args.n_boot, args.seed),
    }
    payload = {
        "weights_native": str(args.weights_native),
        "weights_v1": str(args.weights_v1),
        "e_bbox_q20_threshold": q20,
        "summaries": summaries,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "per_gt_hit_native_vs_v1.json").write_text(
        json.dumps({"per_gt": per_gt, "meta": payload}, indent=2), encoding="utf-8")
    (args.out_dir / "hover_significance_mcnemar.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(args.out_dir / "hover_significance_mcnemar.md", q20, summaries["hover_q20"])
    print(json.dumps(summaries, indent=2, ensure_ascii=False), flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
