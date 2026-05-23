#!/usr/bin/env python3
"""Кэш e_frame / min_e_bbox / min_rel_area для train-video (oversample Q0∩A1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def yolo_labels_xyxy(path: Path, w: int, h: int) -> np.ndarray:
    if not path.is_file():
        return np.zeros((0, 4), dtype=np.float32)
    boxes = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def fd_energy_bbox(fd_bgr: np.ndarray, box: np.ndarray) -> float:
    h, w = fd_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.mean(np.abs(fd_bgr[y1:y2, x1:x2].astype(np.float32))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-txt", type=Path, default=Path("/home/tanyadiplom/data/ard100_mixed/train.txt"))
    ap.add_argument("--train2-txt", type=Path, default=Path("/home/tanyadiplom/data/ard100_mixed/train2.txt"))
    ap.add_argument("--photo-marker", default="/mixed_photos/")
    ap.add_argument("--out-json", type=Path, default=ROOT / "experiments/train_video_strata_cache.json")
    args = ap.parse_args()

    paths = [Path(x.strip()) for x in args.train_txt.read_text().splitlines() if x.strip()]
    paths2 = [Path(x.strip()) for x in args.train2_txt.read_text().splitlines() if x.strip()]
    if len(paths) != len(paths2):
        raise SystemExit("train / train2 length mismatch")

    out: dict = {}
    e_frames: list[float] = []
    for p1, p2 in zip(paths, paths2):
        pn = str(p1).replace("\\", "/")
        if args.photo_marker in pn:
            continue
        im = cv2.imread(str(p1))
        fd = cv2.imread(str(p2))
        if im is None or fd is None:
            continue
        h0, w0 = im.shape[:2]
        lbl_path = Path(pn.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
        gt = yolo_labels_xyxy(lbl_path, w0, h0)
        e_frame = float(np.mean(np.abs(fd.astype(np.float32))))
        img_area = float(h0 * w0)
        min_rel = 1.0
        min_e = e_frame
        for box in gt:
            ra = float(((box[2] - box[0]) * (box[3] - box[1])) / max(1.0, img_area))
            min_rel = min(min_rel, ra)
            min_e = min(min_e, fd_energy_bbox(fd, box))
        e_frames.append(e_frame)
        out[pn] = {
            "e_frame": e_frame,
            "min_e_bbox": float(min_e),
            "min_rel_area": float(min_rel),
            "n_gt": int(len(gt)),
        }

    e_arr = np.array(e_frames, dtype=np.float64)
    stats = {
        "n_video": len(out),
        "e_frame_p25": float(np.quantile(e_arr, 0.25)) if len(e_arr) else 0.0,
        "e_frame_p50": float(np.quantile(e_arr, 0.50)) if len(e_arr) else 0.0,
    }
    payload = {"stats": stats, "by_path": out}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json} n_video={len(out)} e_frame_p25={stats['e_frame_p25']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
