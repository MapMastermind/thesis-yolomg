#!/usr/bin/env python3
"""
Eval B (гл. 3, блок I): бины e_bbox по энергии FD внутри GT-bbox + gate-off / FD-Gate на одних кадрах.
Определения e_bbox и hover-proxy — гл. 2 (experiments/analytics/ard100_chapter2_analytics.md).

Энергия в bbox (на полном разрешении FD-кадра):
  e_bbox = mean(|FD_BGR|) по пикселям внутри GT xyxy (клип к границам кадра).

Энергия кадра (для справки, как calibrate_video_fd_energy_tau):
  e_frame = mean(|letterbox(FD, energy_imgsz, auto=False)|).

Бины: квантили e_bbox по всем GT на val-видео (без /mixed_photos/), по умолчанию 5 бинов.

Условия инференса (одни и те же кадры, нативный FD во 2-м потоке):
  gate_off, gate_energy (τ), опционально gate_energy_mix; опционально второй чекпойнт (fdgate train).

FD-Gate переключается патчем Concat3 после load (без пересборки весов).
"""
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
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "_evpv", _SCRIPTS / "eval_yolomg_val_photo_video_metrics.py"
)
_evpv = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_evpv)

box_iou_xyxy = _evpv.box_iou_xyxy
label_path_from_rgb = _evpv.label_path_from_rgb
load_val_pairs = _evpv.load_val_pairs
run_inference_batch = _evpv.run_inference_batch
yolo_txt_to_xyxy = _evpv.yolo_txt_to_xyxy

_orig_torch_load = torch.load


def _torch_load_weights(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights


@dataclass(frozen=True)
class GateConfig:
    name: str
    mode: str  # off | energy | energy_mix
    tau: float = 0.015
    k: float = 120.0
    min_alpha: float = 0.05


@dataclass(frozen=True)
class WeightRun:
    label: str
    weights: Path


@dataclass(frozen=True)
class Stream2Policy:
    name: str
    mode: str  # native | zero_if_low_e_bbox
    e_bbox_tau: float = 0.0


def fd_energy_bbox(fd_bgr: np.ndarray, box_xyxy: np.ndarray) -> float:
    h, w = fd_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = fd_bgr[y1:y2, x1:x2]
    return float(np.mean(np.abs(crop.astype(np.float32))))


def fd_energy_frame_letterbox(fd_bgr: np.ndarray, energy_imgsz: int) -> float:
    from utils.augmentations import letterbox

    lb, _, _ = letterbox(fd_bgr, (energy_imgsz, energy_imgsz), auto=False, scaleup=False)
    return float(np.mean(np.abs(lb.astype(np.float32))))


def set_fd_gate_on_model(model: torch.nn.Module, cfg: GateConfig) -> int:
    from models.common import Concat3, Concat3Adaptive

    n = 0
    for m in model.modules():
        if isinstance(m, (Concat3, Concat3Adaptive)):
            if hasattr(m, "_ensure_concat3_ckpt_attrs"):
                m._ensure_concat3_ckpt_attrs()
            m.fd_gate_mode = cfg.mode
            m.fd_gate_tau = float(cfg.tau)
            m.fd_gate_k = float(cfg.k)
            m.fd_gate_min_alpha = float(cfg.min_alpha)
            n += 1
    return n


# Фиксированные бины rel_area (доля кадра) — как compare_ard100_model_errors.py
REL_AREA_BIN_EDGES_FIXED = np.array([0.0, 1e-5, 5e-5, 2e-4, 1e-3, 1.0], dtype=np.float64)


def rel_area_bin_label(i: int, edges: np.ndarray) -> str:
    lo, hi = edges[i], edges[i + 1]
    if hi >= 1.0:
        return f"A{i} rel∈[{lo:.0e},{hi:.3f}]"
    return f"A{i} rel∈[{lo:.0e},{hi:.0e})"


def motion_trust_alpha_from_model(model: torch.nn.Module) -> float | None:
    """Средний α MotionTrustHead после forward (только Concat3Adaptive, eval mode)."""
    from models.common import Concat3Adaptive

    vals: list[float] = []
    for m in model.modules():
        if isinstance(m, Concat3Adaptive) and m.last_alpha_mean is not None:
            vals.append(float(m.last_alpha_mean))
    return float(np.mean(vals)) if vals else None


def stem_gate_alpha_stats(
    model: torch.nn.Module,
    im_lb: np.ndarray,
    im_lb2: np.ndarray,
    dev: torch.device,
) -> dict | None:
    """α FD-Gate по e_stem на входе Concat3 (pre-hook), как calibrate_stem_energy_tau.py."""
    from models.common import Concat3, Concat3Adaptive

    concat = None
    for m in model.modules():
        if isinstance(m, (Concat3, Concat3Adaptive)):
            concat = m
            break
    if concat is None or concat.fd_gate_mode == "off":
        return None
    captured: list[float] = []

    def pre_hook(_mod, inputs):
        captured.append(float(inputs[0].detach().abs().mean().cpu()))

    handle = concat.register_forward_pre_hook(pre_hook)
    im_t = torch.from_numpy(np.ascontiguousarray(im_lb.transpose((2, 0, 1))[::-1])).to(dev)
    im_t = im_t.float().unsqueeze(0) / 255.0
    im_t2 = torch.from_numpy(np.ascontiguousarray(im_lb2.transpose((2, 0, 1))[::-1])).to(dev)
    im_t2 = im_t2.float().unsqueeze(0) / 255.0
    try:
        with torch.no_grad():
            model(im_t, im_t2)
    finally:
        handle.remove()
    if not captured:
        return None
    e = captured[0]
    alpha = float(
        torch.sigmoid(
            torch.tensor(concat.fd_gate_k * (e - concat.fd_gate_tau))
        ).clamp(min=concat.fd_gate_min_alpha, max=1.0)
    )
    return {
        "stem_energy_mean": e,
        "alpha_mean": alpha,
        "tau": float(concat.fd_gate_tau),
    }


def best_iou_per_gt(gt: np.ndarray, pred_xyxy: np.ndarray, pred_conf: np.ndarray) -> tuple[list[float], list[float]]:
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


def aggregate_bin_metrics(rows: list[dict]) -> dict:
    n_gt = len(rows)
    if n_gt == 0:
        return {"n_gt": 0}
    hits50 = sum(1 for r in rows if r["best_iou"] >= 0.5)
    hits75 = sum(1 for r in rows if r["best_iou"] >= 0.75)
    ious = np.array([r["best_iou"] for r in rows], dtype=np.float64)
    return {
        "n_gt": n_gt,
        "hit_rate_iou_0.5": hits50 / n_gt,
        "hit_rate_iou_0.75": hits75 / n_gt,
        "mean_best_iou": float(ious.mean()),
        "median_best_iou": float(np.median(ious)),
    }


def bin_label(i: int, edges: np.ndarray, n_bins: int) -> str:
    lo, hi = edges[i], edges[i + 1]
    q_lo = int(round(100 * i / n_bins))
    q_hi = int(round(100 * (i + 1) / n_bins))
    lo_s = "-inf" if not np.isfinite(lo) or lo < -1e30 else f"{lo:.4g}"
    hi_s = "inf" if not np.isfinite(hi) or hi > 1e30 else f"{hi:.4g}"
    return f"Q{q_lo}–Q{q_hi} e∈[{lo_s},{hi_s})"


def collect_video_gt_energy(
    pairs: list[tuple[Path, Path]],
    photo_marker: str,
    max_samples: int,
) -> tuple[list[dict], list[float]]:
    """Первый проход: только GT и e_bbox / e_frame (без нейросети)."""
    records: list[dict] = []
    all_e_bbox: list[float] = []
    n_seen = 0
    for ipath, ipath2 in pairs:
        pnorm = str(ipath).replace("\\", "/")
        if photo_marker in pnorm:
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
        e_frame = fd_energy_frame_letterbox(im02, 640)
        for gi, box in enumerate(gt):
            e_b = fd_energy_bbox(im02, box)
            all_e_bbox.append(e_b)
            records.append(
                {
                    "path": str(ipath),
                    "path2": str(ipath2),
                    "hw": [h0, w0],
                    "gt_index": gi,
                    "box_xyxy": box.tolist(),
                    "e_bbox": e_b,
                    "e_frame": e_frame,
                    "rel_area": float(((box[2] - box[0]) * (box[3] - box[1])) / max(1.0, h0 * w0)),
                }
            )
    return records, all_e_bbox


def frame_min_e_bbox_map(records: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in records:
        p = r["path"]
        e = float(r["e_bbox"])
        out[p] = min(out.get(p, e), e)
    return out


def apply_stream2_policy(
    im_lb2: np.ndarray,
    path: str,
    policy: Stream2Policy,
    frame_min_e: dict[str, float],
) -> np.ndarray:
    if policy.mode == "native":
        return im_lb2
    if policy.mode == "zero_if_low_e_bbox":
        if frame_min_e.get(path, float("inf")) < policy.e_bbox_tau:
            return np.zeros_like(im_lb2)
        return im_lb2
    raise ValueError(f"Unknown stream2 mode: {policy.mode}")


def assign_bins(records: list[dict], bin_edges: np.ndarray) -> None:
    n_bins = len(bin_edges) - 1
    for r in records:
        e = r["e_bbox"]
        bi = int(np.searchsorted(bin_edges, e, side="right") - 1)
        bi = min(max(bi, 0), n_bins - 1)
        r["bin_id"] = bi
        r["bin_label"] = bin_label(bi, bin_edges, n_bins)


def assign_area_bins(records: list[dict], area_edges: np.ndarray) -> None:
    n_bins = len(area_edges) - 1
    for r in records:
        ra = float(r["rel_area"])
        bi = int(np.searchsorted(area_edges, ra, side="right") - 1)
        bi = min(max(bi, 0), n_bins - 1)
        r["area_bin_id"] = bi
        r["area_bin_label"] = rel_area_bin_label(bi, area_edges)


def run_conditions(
    records: list[dict],
    weight_runs: list[WeightRun],
    gate_configs: list[GateConfig],
    stream2_policies: list[Stream2Policy],
    imgsz: int,
    conf_nms: float,
    iou_nms: float,
    device: str,
) -> dict:
    import os

    from models.experimental import attempt_load
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_coords

    dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    cache: dict[tuple[str, str, str], dict[str, tuple[np.ndarray, np.ndarray, float | None]]] = {}
    frame_min_e = frame_min_e_bbox_map(records)

    unique_frames: dict[str, dict] = {}
    for rec in records:
        if rec["path"] not in unique_frames:
            unique_frames[rec["path"]] = {"path2": rec["path2"], "hw": tuple(rec["hw"])}

    for wr in weight_runs:
        wpath = wr.weights if wr.weights.is_absolute() else _ROOT / wr.weights
        os.environ["YOLOMG_CONCAT3_FD_GATE"] = "off"
        model = attempt_load(str(wpath), map_location=dev)
        model.eval()
        stride = int(model.stride.max())
        n_concat = 0
        for s2 in stream2_policies:
            for gc in gate_configs:
                n_concat = set_fd_gate_on_model(model, gc)
                key = (wr.label, gc.name, s2.name)
                cache[key] = {}
                t0 = time.perf_counter()
                n_zero = 0
                for fi, (p, meta) in enumerate(unique_frames.items()):
                    im0 = cv2.imread(p)
                    im02 = cv2.imread(meta["path2"])
                    if im0 is None or im02 is None:
                        cache[key][p] = (np.zeros((0, 4)), np.zeros(0), None)
                        continue
                    h0, w0 = meta["hw"]
                    im_lb, _, _ = letterbox(im0, (imgsz, imgsz), auto=True, stride=stride)
                    im_lb2, _, _ = letterbox(im02, (imgsz, imgsz), auto=True, stride=stride)
                    im_lb2 = apply_stream2_policy(im_lb2, p, s2, frame_min_e)
                    if s2.mode == "zero_if_low_e_bbox" and frame_min_e.get(p, float("inf")) < s2.e_bbox_tau:
                        n_zero += 1
                    _area_conf = os.environ.get("YOLOMG_INFER_AREA_CONF", "0").strip().lower() in (
                        "1", "true", "yes", "on",
                    )
                    _conf_base = float(os.environ.get("YOLOMG_INFER_AREA_CONF_BASE", "0.25"))
                    pred_xyxy, pred_conf = run_inference_batch(
                        model,
                        im_lb,
                        im_lb2,
                        (h0, w0),
                        dev,
                        conf_nms,
                        iou_nms,
                        non_max_suppression,
                        scale_coords,
                        area_dependent_conf=_area_conf,
                        conf_base=_conf_base,
                    )
                    alpha_m = motion_trust_alpha_from_model(model)
                    cache[key][p] = (pred_xyxy, pred_conf, alpha_m)
                    if (fi + 1) % 400 == 0:
                        print(
                            f"  [{wr.label}|{gc.name}|{s2.name}] frames {fi + 1}/{len(unique_frames)} "
                            f"elapsed={time.perf_counter() - t0:.0f}s",
                            flush=True,
                        )
                print(
                    f"Done {wr.label}|{gc.name}|{s2.name}: {len(unique_frames)} frames "
                    f"(stream2_zero={n_zero}), Concat3={n_concat}, {time.perf_counter() - t0:.0f}s",
                    flush=True,
                )
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # Собрать per-GT результаты и бины
    out_bins: dict = {}
    out_bins_2d: dict = {}
    hover_rows: dict = {}
    motion_trust_alpha: dict = {}
    n_e_bins = len(bin_edges_global) - 1 if len(bin_edges_global) else 0
    n_a_bins = len(area_edges_global) - 1 if len(area_edges_global) else 0
    for wr in weight_runs:
        for s2 in stream2_policies:
            for gc in gate_configs:
                cond = f"{wr.label}__{gc.name}__{s2.name}"
                gt_rows: list[dict] = []
                for rec in records:
                    p = rec["path"]
                    pred_xyxy, pred_conf, alpha_m = cache[(wr.label, gc.name, s2.name)][p]
                    gt_one = np.array([rec["box_xyxy"]], dtype=np.float32)
                    biou, bcf = best_iou_per_gt(gt_one, pred_xyxy, pred_conf)
                    gt_rows.append(
                        {
                            "path": p,
                            "bin_id": rec["bin_id"],
                            "area_bin_id": rec.get("area_bin_id", 0),
                            "e_bbox": rec["e_bbox"],
                            "e_frame": rec["e_frame"],
                            "rel_area": rec["rel_area"],
                            "motion_alpha": alpha_m,
                            "best_iou": biou[0],
                            "best_conf": bcf[0],
                        }
                    )
                n_bins = max(r["bin_id"] for r in records) + 1 if records else 0
                per_bin = {}
                for bi in range(n_bins):
                    sub = [r for r in gt_rows if r["bin_id"] == bi]
                    n_b = len(bin_edges_global) - 1
                    per_bin[str(bi)] = {
                        "bin_label": bin_label(bi, bin_edges_global, n_b),
                        **aggregate_bin_metrics(sub),
                    }
                out_bins[cond] = per_bin

                per_2d: dict = {}
                for ei in range(n_e_bins):
                    for ai in range(n_a_bins):
                        sub = [
                            r
                            for r in gt_rows
                            if r["bin_id"] == ei and r["area_bin_id"] == ai
                        ]
                        key2 = f"e{ei}_a{ai}"
                        per_2d[key2] = {
                            "e_bin": ei,
                            "e_bin_label": bin_label(ei, bin_edges_global, n_e_bins),
                            "area_bin": ai,
                            "area_bin_label": rel_area_bin_label(ai, area_edges_global),
                            **aggregate_bin_metrics(sub),
                        }
                out_bins_2d[cond] = per_2d

                alphas_by_e: dict[str, list[float]] = {str(i): [] for i in range(n_e_bins)}
                for r in gt_rows:
                    if r["motion_alpha"] is not None:
                        alphas_by_e[str(r["bin_id"])].append(float(r["motion_alpha"]))
                mt_summary: dict = {}
                for ei in range(n_e_bins):
                    vals = alphas_by_e.get(str(ei), [])
                    mt_summary[str(ei)] = {
                        "e_bin_label": bin_label(ei, bin_edges_global, n_e_bins),
                        "n_gt_with_alpha": len(vals),
                        "alpha_mean": float(np.mean(vals)) if vals else None,
                        "alpha_median": float(np.median(vals)) if vals else None,
                        "alpha_std": float(np.std(vals)) if len(vals) > 1 else None,
                    }
                motion_trust_alpha[cond] = mt_summary

                q20 = float(np.quantile([r["e_bbox"] for r in records], 0.20))
                hover_sub = [r for r in gt_rows if r["e_bbox"] <= q20]
                hover_rows[cond] = {
                    "e_bbox_q20_threshold": q20,
                    "definition": "GT with e_bbox <= global Q20 on val-video (proxy low-FD / hover)",
                    **aggregate_bin_metrics(hover_sub),
                }

    return {
        "by_bin": out_bins,
        "by_bin_2d": out_bins_2d,
        "hover_q20": hover_rows,
        "motion_trust_alpha_by_e_bbox_bin": motion_trust_alpha,
    }


# module-level for bin_label in run_conditions
bin_edges_global: np.ndarray = np.array([])
area_edges_global: np.ndarray = np.array([])


def main() -> None:
    import os

    ap = argparse.ArgumentParser(description="Eval B: e_bbox bins in GT bbox + FD-Gate ablation (гл. 3)")
    ap.add_argument("--weights-baseline", type=Path, default=Path(
        "runs/train/ard100_mixed640_zeroP_vidDrop03_from_vidbest_gpu0/weights/best.pt"
    ))
    ap.add_argument(
        "--weights-label",
        default="baseline_mixed640_zeroP",
        help="Метка чекпойнта в JSON/таблице (напр. lowfdZero_h1)",
    )
    ap.add_argument(
        "--weights-fdgate-train",
        type=Path,
        default=Path("runs/train/ard100_mixed640_fdgate_energy_tau0015_e12_gpu0/weights/best.pt"),
        help="Опционально: чекпойнт после train с FD-Gate; пусто — не грузить",
    )
    ap.add_argument("--skip-fdgate-weights", action="store_true")
    ap.add_argument("--val-txt", type=Path, default=Path("/home/tanyadiplom/data/ard100_mixed/val.txt"))
    ap.add_argument("--val2-txt", type=Path, default=Path("/home/tanyadiplom/data/ard100_mixed/val2.txt"))
    ap.add_argument("--photo-marker", default="/mixed_photos/")
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-nms", type=float, default=0.001)
    ap.add_argument("--iou-nms", type=float, default=0.4)
    ap.add_argument("--pred-rate-conf", type=float, default=0.25)
    ap.add_argument("--gate-tau", type=float, default=None, help="τ FD-Gate (stem); если не задан — см. --auto-gate-tau")
    ap.add_argument(
        "--auto-gate-tau",
        choices=("p50", "p75", "none"),
        default="none",
        help="Взять τ из experiments/stem_energy_val_video.json",
    )
    ap.add_argument("--stem-calib-json", type=Path, default=Path("experiments/stem_energy_val_video.json"))
    ap.add_argument("--gate-k", type=float, default=120.0)
    ap.add_argument(
        "--gate-modes",
        default="gate_off,gate_energy",
        help="Список: gate_off,gate_energy,gate_energy_mix",
    )
    ap.add_argument(
        "--stream2-policies",
        default="native",
        help="native,zero_if_low_e_bbox или оба через запятую",
    )
    ap.add_argument(
        "--e-bbox-tau",
        type=float,
        default=None,
        help="Порог e_bbox для zero_if_low_e_bbox (min e_bbox на кадре)",
    )
    ap.add_argument(
        "--e-bbox-tau-quantile",
        type=float,
        default=0.20,
        help="Если --e-bbox-tau не задан: квантиль по GT e_bbox (default Q20)",
    )
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-samples", type=int, default=0, help="Лимит видео-кадров (0 = все)")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument(
        "--gate-alpha-samples",
        type=int,
        default=64,
        help="Сколько val-кадров зондировать для stem α (0 = выкл)",
    )
    ap.add_argument(
        "--weights-compare",
        type=Path,
        default=None,
        help="Второй чекпойнт в том же прогоне (напр. adaptiveFusion vs zeroP)",
    )
    ap.add_argument("--weights-compare-label", default="compare")
    ap.add_argument(
        "--rel-area-bin-mode",
        choices=("fixed", "quantile"),
        default="fixed",
        help="Бины rel_area: fixed (как compare_ard100) или квантили по val-video GT",
    )
    ap.add_argument("--rel-area-n-bins", type=int, default=5, help="Число бинов при quantile mode")
    args = ap.parse_args()

    pairs = load_val_pairs(args.val_txt, args.val2_txt)
    print(f"Val pairs: {len(pairs)}; collecting GT FD energy in bbox…", flush=True)
    records, all_e = collect_video_gt_energy(pairs, args.photo_marker, args.max_samples)
    if not records:
        raise SystemExit("No GT records on video val.")

    global bin_edges_global, area_edges_global
    qs = np.linspace(0.0, 1.0, args.n_bins + 1)
    bin_edges_global = np.quantile(np.array(all_e, dtype=np.float64), qs)
    bin_edges_global[0] = -np.inf
    bin_edges_global[-1] = np.inf
    assign_bins(records, bin_edges_global)

    all_ra = np.array([r["rel_area"] for r in records], dtype=np.float64)
    if args.rel_area_bin_mode == "fixed":
        area_edges_global = REL_AREA_BIN_EDGES_FIXED.copy()
    else:
        qs_a = np.linspace(0.0, 1.0, args.rel_area_n_bins + 1)
        area_edges_global = np.quantile(all_ra, qs_a)
        area_edges_global[0] = 0.0
        area_edges_global[-1] = 1.0
    assign_area_bins(records, area_edges_global)

    print(
        f"GT instances: {len(records)}; e_bbox "
        f"min={min(all_e):.4g} med={np.median(all_e):.4g} max={max(all_e):.4g}",
        flush=True,
    )
    print(f"Bin edges (e_bbox quantiles): {bin_edges_global}", flush=True)

    e_bbox_tau = args.e_bbox_tau
    if e_bbox_tau is None:
        e_bbox_tau = float(np.quantile(np.array(all_e, dtype=np.float64), args.e_bbox_tau_quantile))
    print(f"e_bbox_tau for zero_if_low_e_bbox (Q{int(args.e_bbox_tau_quantile*100)}): {e_bbox_tau:.4g}", flush=True)

    gate_tau = args.gate_tau
    stem_calib = {}
    if args.auto_gate_tau != "none":
        cal_path = args.stem_calib_json if args.stem_calib_json.is_absolute() else _ROOT / args.stem_calib_json
        if not cal_path.is_file():
            raise SystemExit(f"Run calibrate_stem_energy_tau.py first; missing {cal_path}")
        stem_calib = json.loads(cal_path.read_text(encoding="utf-8"))
        gate_tau = float(stem_calib["quantiles"][args.auto_gate_tau])
    if gate_tau is None:
        gate_tau = 0.015
    print(f"FD-Gate stem τ={gate_tau:.6f}", flush=True)

    mode_names = [x.strip() for x in args.gate_modes.split(",") if x.strip()]
    gate_configs: list[GateConfig] = []
    for mn in mode_names:
        if mn == "gate_off":
            gate_configs.append(GateConfig("gate_off", "off"))
        elif mn == "gate_energy":
            gate_configs.append(GateConfig("gate_energy", "energy", tau=gate_tau, k=args.gate_k))
        elif mn == "gate_energy_mix":
            gate_configs.append(GateConfig("gate_energy_mix", "energy_mix", tau=gate_tau, k=args.gate_k))
        else:
            raise SystemExit(f"Unknown gate mode: {mn}")

    s2_names = [x.strip() for x in args.stream2_policies.split(",") if x.strip()]
    stream2_policies: list[Stream2Policy] = []
    for sn in s2_names:
        if sn == "native":
            stream2_policies.append(Stream2Policy("stream2_native", "native"))
        elif sn == "zero_if_low_e_bbox":
            stream2_policies.append(
                Stream2Policy("stream2_zero_if_low_e_bbox", "zero_if_low_e_bbox", e_bbox_tau=e_bbox_tau)
            )
        else:
            raise SystemExit(f"Unknown stream2 policy: {sn}")
    weight_runs = [WeightRun(args.weights_label, args.weights_baseline)]
    if args.weights_compare:
        wcp = args.weights_compare
        wcl = args.weights_compare_label.strip() or "compare"
        weight_runs.append(WeightRun(wcl, wcp))
    if not args.skip_fdgate_weights and args.weights_fdgate_train:
        wp = args.weights_fdgate_train
        if wp.exists() or (_ROOT / wp).exists():
            weight_runs.append(WeightRun("fdgate_trained_tau0015", wp))

    print(f"Weight runs: {[w.label for w in weight_runs]}", flush=True)
    print(f"Gate configs: {[g.name for g in gate_configs]}", flush=True)
    print(f"Stream2 policies: {[(s.name, s.mode, s.e_bbox_tau) for s in stream2_policies]}", flush=True)

    gate_alpha_diag: dict = {}
    if args.gate_alpha_samples > 0:
        import os
        from models.experimental import attempt_load
        from utils.augmentations import letterbox

        dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        os.environ["YOLOMG_CONCAT3_FD_GATE"] = "off"
        wpath = weight_runs[0].weights if weight_runs[0].weights.is_absolute() else _ROOT / weight_runs[0].weights
        model = attempt_load(str(wpath), map_location=dev)
        model.eval()
        stride = int(model.stride.max())
        for gc in [g for g in gate_configs if g.mode != "off"]:
            set_fd_gate_on_model(model, gc)
            alphas, stem_e = [], []
            n = 0
            for p, meta in list({r["path"]: r for r in records}.items())[: args.gate_alpha_samples]:
                im0 = cv2.imread(p)
                im02 = cv2.imread(meta["path2"])
                if im0 is None or im02 is None:
                    continue
                im_lb, _, _ = letterbox(im0, (args.imgsz, args.imgsz), auto=True, stride=stride)
                im_lb2, _, _ = letterbox(im02, (args.imgsz, args.imgsz), auto=True, stride=stride)
                st = stem_gate_alpha_stats(model, im_lb, im_lb2, dev)
                if st:
                    alphas.append(st["alpha_mean"])
                    stem_e.append(st["stem_energy_mean"])
                    n += 1
            gate_alpha_diag[gc.name] = {
                "n_samples": n,
                "alpha_mean_avg": float(np.mean(alphas)) if alphas else None,
                "stem_energy_mean_avg": float(np.mean(stem_e)) if stem_e else None,
                "note": "τ относится к energy motion-стема после Conv, не к e_bbox в пикселях FD",
            }
            print(f"Gate diag {gc.name}: {gate_alpha_diag[gc.name]}", flush=True)
        del model

    results = run_conditions(
        records,
        weight_runs,
        gate_configs,
        stream2_policies,
        args.imgsz,
        args.conf_nms,
        args.iou_nms,
        args.device,
    )

    # Дельты gate vs off на baseline (те же кадры, тот же stream2)
    deltas = {}
    base_label = args.weights_label
    for s2 in stream2_policies:
        ref_key = f"{base_label}__gate_off__{s2.name}"
        if ref_key not in results["by_bin"]:
            continue
        ref = results["by_bin"][ref_key]
        for gc in gate_configs:
            if gc.name == "gate_off":
                continue
            key = f"{base_label}__{gc.name}__{s2.name}"
            if key not in results["by_bin"]:
                continue
            d = {}
            for bi, ref_m in ref.items():
                cur = results["by_bin"][key].get(bi, {})
                if ref_m.get("n_gt", 0) and cur.get("n_gt", 0):
                    d[bi] = {
                        "bin_label": ref_m.get("bin_label", bi),
                        "delta_hit_0.5": cur["hit_rate_iou_0.5"] - ref_m["hit_rate_iou_0.5"],
                        "delta_mean_iou": cur["mean_best_iou"] - ref_m["mean_best_iou"],
                        "n_gt": cur["n_gt"],
                    }
            deltas[f"{key}_minus_gate_off"] = d

    # stream2 zero vs native (gate_off)
    stream2_deltas = {}
    if any(s.mode == "zero_if_low_e_bbox" for s in stream2_policies) and any(
        s.mode == "native" for s in stream2_policies
    ):
        s_nat = next(s for s in stream2_policies if s.mode == "native")
        s_zero = next(s for s in stream2_policies if s.mode == "zero_if_low_e_bbox")
        ref_key = f"{base_label}__gate_off__{s_nat.name}"
        cur_key = f"{base_label}__gate_off__{s_zero.name}"
        if ref_key in results["by_bin"] and cur_key in results["by_bin"]:
            ref = results["by_bin"][ref_key]
            cur = results["by_bin"][cur_key]
            d = {}
            for bi in ref:
                if ref[bi].get("n_gt", 0) and cur.get(bi, {}).get("n_gt", 0):
                    d[bi] = {
                        "delta_hit_0.5": cur[bi]["hit_rate_iou_0.5"] - ref[bi]["hit_rate_iou_0.5"],
                        "n_gt": cur[bi]["n_gt"],
                    }
            stream2_deltas[f"{cur_key}_minus_{ref_key}"] = d

    payload = {
        "protocol": "evalB_ebbox_bins_fd_alpha",
        "weights": {w.label: str(w.weights if w.weights.is_absolute() else _ROOT / w.weights) for w in weight_runs},
        "gate_tau_stem": gate_tau,
        "e_bbox_tau_zero_stream": e_bbox_tau,
        "stem_calibration": stem_calib,
        "gate_k": args.gate_k,
        "imgsz": args.imgsz,
        "conf_nms": args.conf_nms,
        "iou_nms": args.iou_nms,
        "n_video_gt_instances": len(records),
        "n_unique_video_frames": len({r["path"] for r in records}),
        "e_bbox_definition": "mean(abs(FD_BGR)) inside GT xyxy on native FD resolution",
        "e_frame_definition": "mean(abs(letterbox(FD,640,false))) per frame",
        "bin_edges_e_bbox": bin_edges_global.tolist(),
        "bin_labels": [
            bin_label(i, bin_edges_global, len(bin_edges_global) - 1)
            for i in range(len(bin_edges_global) - 1)
        ],
        "rel_area_bin_mode": args.rel_area_bin_mode,
        "bin_edges_rel_area": area_edges_global.tolist(),
        "rel_area_bin_labels": [
            rel_area_bin_label(i, area_edges_global) for i in range(len(area_edges_global) - 1)
        ],
        "results_by_bin": results["by_bin"],
        "results_by_bin_2d": results["by_bin_2d"],
        "hover_proxy_q20": results["hover_q20"],
        "motion_trust_alpha_by_e_bbox_bin": results["motion_trust_alpha_by_e_bbox_bin"],
        "delta_baseline_gate_vs_off": deltas,
        "delta_stream2_zero_vs_native": stream2_deltas,
        "gate_alpha_diagnostics": gate_alpha_diag,
    }

    # Краткая сводка: bin Q0–Q20 (index 0)
    print("\n=== SUMMARY bin Q0 (low e_bbox) hit@0.5 ===", flush=True)
    for cond, bins in sorted(results["by_bin"].items()):
        m = bins.get("0", {})
        if m.get("n_gt"):
            print(f"  {cond}: {m['hit_rate_iou_0.5']:.4f} (n={m['n_gt']})", flush=True)
    print("=== SUMMARY hover Q20 hit@0.5 ===", flush=True)
    for cond, h in sorted(results["hover_q20"].items()):
        if h.get("n_gt"):
            print(f"  {cond}: {h['hit_rate_iou_0.5']:.4f}", flush=True)

    print("\n=== MotionTrust α by e_bbox bin (Concat3Adaptive only) ===", flush=True)
    for cond, bins_a in sorted(results["motion_trust_alpha_by_e_bbox_bin"].items()):
        for bi in ("0", str(len(bin_edges_global) - 2)):
            row = bins_a.get(bi, {})
            if row.get("alpha_mean") is not None:
                print(
                    f"  {cond} e_bin={bi}: α_mean={row['alpha_mean']:.4f} "
                    f"(n={row['n_gt_with_alpha']})",
                    flush=True,
                )

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    out_json = args.out_json or (_ROOT / "experiments/results/evalB_bins_ebbox_fd_alpha.json")
    out_md = args.out_md or (_ROOT / "experiments/results/evalB_bins_ebbox_fd_alpha.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_json}", file=sys.stderr)

    lines = [
        "# Eval B — бины e_bbox, hover-proxy (гл. 3, блок I)\n",
        f"- GT instances (video val): **{len(records)}**",
        f"- Frames: **{payload['n_unique_video_frames']}**",
        f"- Weights baseline: `{payload['weights'].get('baseline_mixed640_zeroP', '')}`",
        f"- FD-Gate stem τ={gate_tau}, e_bbox τ (zero stream)={e_bbox_tau:.4g}, imgsz={args.imgsz}\n",
        "## Hit@0.5 by e_bbox bin (same GT boxes)\n",
        "| condition | " + " | ".join(payload["bin_labels"]) + " | hover Q20 |",
        "|-----------|" + "|".join(["---"] * (len(payload["bin_labels"]) + 1)) + "|",
    ]
    for cond, bins in results["by_bin"].items():
        cells = []
        for bi in range(len(payload["bin_labels"])):
            m = bins.get(str(bi), {})
            cells.append(f"{m.get('hit_rate_iou_0.5', 0):.3f}" if m.get("n_gt") else "—")
        h = results["hover_q20"].get(cond, {})
        cells.append(f"{h.get('hit_rate_iou_0.5', 0):.3f}" if h.get("n_gt") else "—")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")

    lines.append("\n## Hit@0.5 — 2D: e_bbox × rel_area (D1)\n")
    area_labels = payload["rel_area_bin_labels"]
    for cond, grid in results["by_bin_2d"].items():
        lines.append(f"\n### {cond}\n")
        header = "| e_bbox \\ rel_area | " + " | ".join(area_labels) + " |"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (len(area_labels) + 1)) + "|")
        for ei in range(len(payload["bin_labels"])):
            cells = []
            for ai in range(len(area_labels)):
                cell = grid.get(f"e{ei}_a{ai}", {})
                n = cell.get("n_gt", 0)
                cells.append(f"{cell.get('hit_rate_iou_0.5', 0):.3f} (n={n})" if n else "—")
            lines.append(f"| {payload['bin_labels'][ei]} | " + " | ".join(cells) + " |")

    lines.append("\n## MotionTrust α по бинам e_bbox (D3, mean по кадрам/GT)\n")
    lines.append("| condition | " + " | ".join(payload["bin_labels"]) + " |")
    lines.append("|" + "|".join(["---"] * (len(payload["bin_labels"]) + 1)) + "|")
    for cond, bins_a in results["motion_trust_alpha_by_e_bbox_bin"].items():
        cells = []
        for ei in range(len(payload["bin_labels"])):
            row = bins_a.get(str(ei), {})
            am = row.get("alpha_mean")
            cells.append(f"{am:.3f}" if am is not None else "—")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")

    lines.append("\n## Δ hit@0.5 vs gate_off (same stream2)\n")
    for dk, dv in deltas.items():
        lines.append(f"### {dk}\n")
        for bi, row in sorted(dv.items(), key=lambda x: int(x[0])):
            lines.append(
                f"- bin {bi} ({row.get('bin_label', '')}): "
                f"Δhit@0.5={row['delta_hit_0.5']:+.4f}, n_gt={row['n_gt']}"
            )
        lines.append("")
    if stream2_deltas:
        lines.append("\n## Δ hit@0.5: zero_if_low_e_bbox vs native (gate_off)\n")
        for dk, dv in stream2_deltas.items():
            lines.append(f"### {dk}\n")
            for bi, row in sorted(dv.items(), key=lambda x: int(x[0])):
                lines.append(f"- bin {bi}: Δhit@0.5={row['delta_hit_0.5']:+.4f}, n_gt={row['n_gt']}")
            lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
