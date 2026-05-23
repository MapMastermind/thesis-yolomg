#!/usr/bin/env python3
"""Калибровка порога YOLOMG_MIXED_VIDEO_FD_ENERGY_TAU: mean(|img2|) после letterbox (как в train).

Читает train.txt/val.txt из data/ard100_mixed.yaml, берёт только видео-строки (без mixed_photos),
для парных путей из val2 считает энергию e на кадре FD после letterbox к imgsz.

Пример:
  cd Static_YOLOMG && ./.venv/bin/python scripts/calibrate_video_fd_energy_tau.py --split val --max-samples 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.augmentations import letterbox  # noqa: E402


def load_yaml(p: Path) -> dict:
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def energy_for_pair(path_main: str, path2: str, imgsz: int) -> float | None:
    im = cv2.imread(path_main)
    im2 = cv2.imread(path2)
    if im is None or im2 is None:
        return None
    im2_lb, _, _ = letterbox(im2, (imgsz, imgsz), auto=False, scaleup=False)
    return float(np.mean(np.abs(im2_lb.astype(np.float32))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", type=Path, default=ROOT / "data/ard100_mixed.yaml")
    ap.add_argument("--split", choices=("train", "val"), default="val")
    ap.add_argument("--max-samples", type=int, default=3000)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    meta = load_yaml(args.data_yaml)
    key = "val" if args.split == "val" else "train"
    key2 = "val2" if args.split == "val" else "train2"
    p_txt = Path(meta[key])
    p2_txt = Path(meta[key2])
    if not p_txt.exists() or not p2_txt.exists():
        print("Missing lists:", p_txt, p2_txt, file=sys.stderr)
        return 1

    mains = [ln.strip() for ln in p_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    secs = [ln.strip() for ln in p2_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n = min(len(mains), len(secs))
    es: list[float] = []
    marker = "/mixed_photos/"
    for i in range(n):
        if len(es) >= args.max_samples:
            break
        m, s = mains[i], secs[i]
        if marker in m.replace("\\", "/"):
            continue
        e = energy_for_pair(m, s, args.imgsz)
        if e is not None:
            es.append(e)

    if not es:
        print("No samples collected.", file=sys.stderr)
        return 1

    a = np.array(es, dtype=np.float64)
    qs = [5, 10, 25, 50, 75, 90, 95]
    print(f"# split={args.split} video lines n={len(es)} imgsz={args.imgsz}")
    print(f"e_min={a.min():.6f} e_max={a.max():.6f} e_mean={a.mean():.6f}")
    for q in qs:
        print(f"p{q:02d}={float(np.percentile(a, q)):.6f}")
    print(
        "\n# Подсказка: выберите FD_ENERGY_TAU чуть выше типичного «слабого» FD "
        "(например между p10 и p25), зафиксируйте в plan_vkr_badfd_video_clahe_substitution.txt"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
