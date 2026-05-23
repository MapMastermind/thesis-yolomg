#!/usr/bin/env python3
"""Статистика ARD100 для главы 2: сцены, размеры, bbox, объекты, день/ночь."""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "test_code"))

OUT_ROOT = Path("/home/tanyadiplom/data/ard100")
IMAGES_OUT = OUT_ROOT / "images"
IMAGES2_OUT = OUT_ROOT / "images2"
LABELS_OUT = OUT_ROOT / "labels"
ARD100_ROOT = Path("/data/ARD100")
ANNOTATIONS = ARD100_ROOT / "annotations"

# Авторский split (YOLOMG/test_code/generate_dataset.py set1 / set2)
AUTHOR_SET_TRAIN = {
    "phantom09", "phantom10", "phantom14", "phantom17", "phantom19", "phantom20", "phantom28", "phantom29",
    "phantom30", "phantom32", "phantom36", "phantom40", "phantom42", "phantom43", "phantom44", "phantom46",
    "phantom63", "phantom65", "phantom66", "phantom68", "phantom70", "phantom71", "phantom74", "phantom75",
    "phantom76", "phantom77", "phantom78", "phantom80", "phantom81", "phantom82", "phantom84", "phantom85",
    "phantom86", "phantom87", "phantom89", "phantom90", "phantom101", "phantom103", "phantom104", "phantom105",
    "phantom106", "phantom107", "phantom108", "phantom109", "phantom111", "phantom112", "phantom114",
    "phantom115", "phantom116", "phantom117", "phantom118", "phantom120", "phantom132", "phantom137",
    "phantom138", "phantom139", "phantom140", "phantom142", "phantom143", "phantom145", "phantom146",
    "phantom147", "phantom148", "phantom149", "phantom150",
}
AUTHOR_SET_TEST = {
    "phantom02", "phantom03", "phantom04", "phantom05", "phantom08", "phantom22", "phantom39", "phantom41",
    "phantom45", "phantom47", "phantom50", "phantom54", "phantom55", "phantom56", "phantom57", "phantom58",
    "phantom60", "phantom61", "phantom64", "phantom73", "phantom79", "phantom92", "phantom93", "phantom94",
    "phantom95", "phantom97", "phantom102", "phantom110", "phantom113", "phantom119", "phantom133",
    "phantom135", "phantom136", "phantom141", "phantom144",
}
AUTHOR_LOW_LIGHT = {"phantom95", "phantom97", "phantom133", "phantom135", "phantom136"}

# из prepare_ard100_dataset.py
SET_TRAIN = {
    "phantom09", "phantom10", "phantom14", "phantom17", "phantom19", "phantom20", "phantom28", "phantom29",
    "phantom30", "phantom32", "phantom36", "phantom40", "phantom42", "phantom43", "phantom44", "phantom46",
    "phantom63", "phantom65", "phantom66", "phantom68", "phantom70", "phantom71", "phantom74", "phantom75",
    "phantom76", "phantom77", "phantom78", "phantom80", "phantom81", "phantom82", "phantom84", "phantom85",
    "phantom86", "phantom87", "phantom89", "phantom90", "phantom101", "phantom103", "phantom104", "phantom105",
    "phantom106", "phantom107", "phantom108", "phantom109", "phantom111", "phantom112", "phantom114",
    "phantom115", "phantom116", "phantom117", "phantom118", "phantom120", "phantom132", "phantom137",
    "phantom138", "phantom139", "phantom140", "phantom142", "phantom143", "phantom145", "phantom146",
    "phantom147", "phantom148", "phantom149", "phantom150",
}
SET_TEST = {
    "phantom02", "phantom03", "phantom04", "phantom05", "phantom08", "phantom22", "phantom39", "phantom41",
    "phantom45", "phantom47", "phantom50", "phantom54", "phantom55", "phantom56", "phantom57", "phantom58",
    "phantom60", "phantom61", "phantom64", "phantom73", "phantom79", "phantom92", "phantom93", "phantom94",
    "phantom95", "phantom97", "phantom102", "phantom110", "phantom113", "phantom119", "phantom133",
    "phantom135", "phantom136", "phantom141", "phantom144",
}

REL_AREA_BINS = [
    ("A0: <1e-5", 0.0, 1e-5),
    ("A1: 1e-5–5e-5", 1e-5, 5e-5),
    ("A2: 5e-5–2e-4", 5e-5, 2e-4),
    ("A3: 2e-4–1e-3", 2e-4, 1e-3),
    ("A4: ≥1e-3", 1e-3, 1.0),
]
DISTANCE_PROXY = [
    ("близко (rel≥1e-3)", 1e-3, 1.0),
    ("средняя (2e-4–1e-3)", 2e-4, 1e-3),
    ("далеко (5e-5–2e-4)", 5e-5, 2e-4),
    ("очень далеко (<5e-5)", 0.0, 5e-5),
]
NIGHT_V_THRESH = 85.0  # median V на кадре; эвристика, см. отчёт


def _scene(stem_or_path: str) -> str:
    s = Path(stem_or_path).stem if "/" in stem_or_path or "\\" in stem_or_path else stem_or_path
    return s.split("_")[0]


def _read_split_list(name: str) -> list[str]:
    p = OUT_ROOT / name
    lines = []
    with open(p) as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def _label_path_from_img(img_path: str) -> Path:
    p = Path(img_path)
    rel = p.relative_to(IMAGES_OUT) if str(p).startswith(str(IMAGES_OUT)) else p.name
    stem = Path(rel).stem
    vid = _scene(stem)
    return Path(LABELS_OUT) / vid / f"{stem}.txt"


def _parse_yolo_label(lbl_path: Path) -> list[tuple[float, float, float, float]]:
    if not lbl_path.is_file():
        return []
    boxes = []
    for line in lbl_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = map(float, parts[:5])
        boxes.append((xc, yc, w, h))
    return boxes


def _bin_count(val: float, bins: list[tuple[str, float, float]]) -> str:
    for name, lo, hi in bins:
        if lo <= val < hi or (hi >= 1.0 and val >= lo):
            return name
    return bins[-1][0]


def analyze_splits() -> dict:
    splits = {
        "train": _read_split_list("train.txt"),
        "val": _read_split_list("val.txt"),
        "test": _read_split_list("test.txt"),
    }
    out = {}
    for sp, paths in splits.items():
        scenes = Counter(_scene(p) for p in paths)
        out[sp] = {
            "n_frames": len(paths),
            "n_scenes": len(scenes),
            "scenes": dict(sorted(scenes.items())),
        }
    all_scenes = set(SET_TRAIN) | set(SET_TEST)
    out["total_scenes_in_corpus"] = len(all_scenes)
    out["set_train_phantoms"] = len(SET_TRAIN)
    out["set_test_phantoms"] = len(SET_TEST)
    return out


def analyze_image_sizes(sample_max: int = 500) -> dict:
    """Размеры из XML annotations (полный охват) + выборочная проверка jpg."""
    sizes = Counter()
    for xml in Path("/data/ARD100/annotations").glob("*/*.xml"):
        import xml.etree.ElementTree as ET

        root = ET.parse(xml).getroot()
        sz = root.find("size")
        if sz is None:
            continue
        w, h = int(sz.find("width").text), int(sz.find("height").text)
        sizes[(w, h)] += 1

    wh_from_xml = dict(sizes)
    # sample actual images from lists
    rng = random.Random(0)
    all_paths = _read_split_list("train.txt") + _read_split_list("val.txt") + _read_split_list("test.txt")
    sample = rng.sample(all_paths, min(sample_max, len(all_paths)))
    jpg_sizes = Counter()
    for p in sample:
        im = cv2.imread(p)
        if im is None:
            continue
        jpg_sizes[(im.shape[1], im.shape[0])] += 1

    return {
        "from_voc_xml": {f"{w}x{h}": c for (w, h), c in sorted(wh_from_xml.items(), key=lambda x: -x[1])},
        "xml_unique_resolutions": len(wh_from_xml),
        "jpg_sample_n": len(sample),
        "jpg_sample_sizes": {f"{w}x{h}": c for (w, h), c in jpg_sizes.items()},
    }


def analyze_bbox_and_objects() -> dict:
    gt_total = 0
    rel_areas: list[float] = []
    bbox_w_px: list[float] = []
    bbox_h_px: list[float] = []
    objs_per_image = []
    rel_by_split: dict[str, list[float]] = defaultdict(list)
    objs_by_split: dict[str, list[int]] = defaultdict(list)

    for sp, list_name in [("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")]:
        for img_path in _read_split_list(list_name):
            lbl = _label_path_from_img(img_path)
            boxes = _parse_yolo_label(lbl)
            objs_by_split[sp].append(len(boxes))
            if not boxes:
                continue
            # ARD100: 1920×1080 во всех VOC (см. analyze_image_sizes)
            W, H = 1920, 1080
            for _, _, w, h in boxes:
                rel = w * h
                rel_areas.append(rel)
                rel_by_split[sp].append(rel)
                bbox_w_px.append(w * W)
                bbox_h_px.append(h * H)
                gt_total += 1
        objs_per_image.extend(objs_by_split[sp])

    def pct(arr: list[float], q: float) -> float:
        return float(np.percentile(arr, q)) if arr else 0.0

    rel_bins = Counter(_bin_count(r, REL_AREA_BINS) for r in rel_areas)
    dist_bins = Counter(_bin_count(r, DISTANCE_PROXY) for r in rel_areas)

    return {
        "n_gt_total": gt_total,
        "n_images_with_gt": sum(1 for x in objs_per_image if x > 0),
        "n_images_empty_label": sum(1 for x in objs_per_image if x == 0),
        "objects_per_image": {
            "mean": float(np.mean(objs_per_image)) if objs_per_image else 0,
            "median": float(np.median(objs_per_image)) if objs_per_image else 0,
            "max": int(max(objs_per_image)) if objs_per_image else 0,
            "hist": dict(Counter(objs_per_image)),
        },
        "by_split_objects_per_image_mean": {
            sp: float(np.mean(v)) if v else 0 for sp, v in objs_by_split.items()
        },
        "rel_area": {
            "mean": float(np.mean(rel_areas)),
            "median": float(np.median(rel_areas)),
            "p5": pct(rel_areas, 5),
            "p95": pct(rel_areas, 95),
            "bins_A0_A4": dict(rel_bins),
        },
        "distance_proxy_bins": dict(dist_bins),
        "bbox_width_px": {"mean": float(np.mean(bbox_w_px)), "median": float(np.median(bbox_w_px)), "p5": pct(bbox_w_px, 5), "p95": pct(bbox_w_px, 95)},
        "bbox_height_px": {"mean": float(np.mean(bbox_h_px)), "median": float(np.median(bbox_h_px)), "p5": pct(bbox_h_px, 5), "p95": pct(bbox_h_px, 95)},
        "note_distance": "Дальность — proxy по rel_area (меньше bbox в доле кадра → дальше); меток дальности в VOC нет.",
    }


def analyze_day_night(sample_per_scene: int = 3, max_scenes: int | None = None) -> dict:
    """Эвристика: median V (HSV) по сцене; порог NIGHT_V_THRESH."""
    scene_frames: dict[str, list[str]] = defaultdict(list)
    for sp in ("train", "val", "test"):
        for p in _read_split_list(f"{sp}.txt"):
            scene_frames[_scene(p)].append(p)

    scenes = sorted(scene_frames.keys())
    if max_scenes:
        scenes = scenes[:max_scenes]

    scene_v: dict[str, float] = {}
    frame_v: list[float] = []
    for sc in scenes:
        paths = scene_frames[sc]
        rng = random.Random(hash(sc) % (2**32))
        pick = paths if len(paths) <= sample_per_scene else rng.sample(paths, sample_per_scene)
        vs = []
        for p in pick:
            im = cv2.imread(p)
            if im is None:
                continue
            v = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 2]
            vs.append(float(np.median(v)))
        if vs:
            scene_v[sc] = float(np.median(vs))
            frame_v.extend(vs)

    night_scenes = [s for s, v in scene_v.items() if v < NIGHT_V_THRESH]
    day_scenes = [s for s, v in scene_v.items() if v >= NIGHT_V_THRESH]

    def frames_for_scene_list(scene_list: set[str]) -> int:
        n = 0
        for sp in ("train", "val", "test"):
            for p in _read_split_list(f"{sp}.txt"):
                if _scene(p) in scene_list:
                    n += 1
        return n

    night_set = set(night_scenes)
    day_set = set(day_scenes)

    by_split = {}
    for sp in ("train", "val", "test"):
        day_n = night_n = 0
        for p in _read_split_list(f"{sp}.txt"):
            sc = _scene(p)
            if sc in night_set:
                night_n += 1
            elif sc in day_set:
                day_n += 1
        by_split[sp] = {"day_frames": day_n, "night_frames": night_n, "unknown": len(_read_split_list(f"{sp}.txt")) - day_n - night_n}

    return {
        "method": f"median V (HSV), порог ночь V<{NIGHT_V_THRESH}, {sample_per_scene} кадра/сцена",
        "n_scenes_classified": len(scene_v),
        "n_night_scenes": len(night_scenes),
        "n_day_scenes": len(day_scenes),
        "night_scenes": sorted(night_scenes),
        "day_scenes_sample": sorted(day_scenes)[:20],
        "frames_night": frames_for_scene_list(night_set),
        "frames_day": frames_for_scene_list(day_set),
        "by_split_day_night": by_split,
        "caveat": "Нет меток день/ночь в VOC; классификация по яркости, для ВКР уточнить по паспорту ARD100.",
    }


def analyze_author_split_match() -> dict:
    return {
        "prepare_SET_TRAIN_matches_author_set1": SET_TRAIN == AUTHOR_SET_TRAIN,
        "prepare_SET_TEST_matches_author_set2": SET_TEST == AUTHOR_SET_TEST,
        "n_train_phantoms": len(SET_TRAIN),
        "n_test_phantoms": len(SET_TEST),
        "author_low_light_scenes": sorted(AUTHOR_LOW_LIGHT),
        "note": "Списки SET_TRAIN/SET_TEST в prepare_ard100_dataset.py совпадают с set1/set2 в generate_dataset.py.",
    }


def analyze_storage() -> dict:
    def _du_gb(p: Path) -> float:
        if not p.exists():
            return 0.0
        import subprocess

        out = subprocess.check_output(["du", "-sb", str(p)], text=True).split()[0]
        return round(int(out) / 1e9, 2)

    return {
        "out_root_gb": _du_gb(OUT_ROOT),
        "images_gb": _du_gb(IMAGES_OUT),
        "images2_gb": _du_gb(IMAGES2_OUT),
        "labels_gb": _du_gb(LABELS_OUT),
        "approx_per_frame_mb_dual": round(_du_gb(OUT_ROOT) * 1024 / 202411, 3),
        "note": "Dual-stream ≈ RGB (images) + FD5 (images2) + YOLO labels.",
    }


def analyze_voc_quality() -> dict:
    import xml.etree.ElementTree as ET

    n_xml = 0
    n_obj = 0
    difficult = 0
    truncated = 0
    in_yolo = 0
    for xml_path in ANNOTATIONS.glob("*/*.xml"):
        n_xml += 1
        root = ET.parse(xml_path).getroot()
        for obj in root.iter("object"):
            n_obj += 1
            d = obj.find("difficult")
            if d is not None and int(d.text or 0) == 1:
                difficult += 1
            t = obj.find("truncated")
            if t is not None and int(t.text or 0) == 1:
                truncated += 1
            name = obj.find("name")
            if name is not None and name.text == "Drone":
                stem = xml_path.stem
                vid = stem.split("_")[0]
                yolo = LABELS_OUT / vid / f"{stem}.txt"
                if yolo.is_file() and yolo.read_text().strip():
                    in_yolo += 1

    return {
        "n_voc_xml": n_xml,
        "n_voc_objects_drone": n_obj,
        "difficult_eq_1_excluded_in_prepare": difficult,
        "truncated_eq_1": truncated,
        "drone_with_nonempty_yolo": in_yolo,
        "bbox_noise_note": "При median bbox ~16×11 px сдвиг границы на 1 px сильно меняет IoU@0.75.",
        "prepare_rule": "difficult=1 не попадает в YOLO; min_bbox_area по умолчанию 0.",
    }


def analyze_empty_labels_by_split() -> dict:
    out = {}
    for sp, name in [("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")]:
        empty = with_gt = 0
        for img_path in _read_split_list(name):
            lbl = _label_path_from_img(img_path)
            if lbl.is_file() and lbl.read_text().strip():
                with_gt += 1
            else:
                empty += 1
        out[sp] = {
            "empty": empty,
            "with_gt": with_gt,
            "empty_pct": round(100 * empty / max(1, empty + with_gt), 2),
        }
    out["role"] = "Пустой label — кадр без GT Drone; при train даёт негатив/фон (objectness)."
    return out


def analyze_temporal_correlation(n_scenes: int = 20, max_pairs_per_scene: int = 200) -> dict:
    """Нормированная средняя |Δ| между кадрами (grayscale ¼ разрешения)."""
    rng = random.Random(42)
    scene_frames: dict[str, list[str]] = defaultdict(list)
    for p in _read_split_list("train.txt"):
        scene_frames[_scene(p)].append(p)
    scenes = rng.sample(sorted(scene_frames.keys()), min(n_scenes, len(scene_frames)))

    def _norm_diff(path_a: str, path_b: str) -> float:
        a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            return float("nan")
        a = cv2.resize(a, (480, 270))
        b = cv2.resize(b, (480, 270))
        return float(np.mean(cv2.absdiff(a, b)) / 255.0)

    deltas = {1: [], 5: [], 30: []}
    for sc in scenes:
        paths = sorted(scene_frames[sc])
        if len(paths) < 31:
            continue
        idxs = rng.sample(range(len(paths) - 31), min(max_pairs_per_scene, len(paths) - 31))
        for i in idxs:
            for dt, bucket in [(1, 1), (5, 5), (30, 30)]:
                d = _norm_diff(paths[i], paths[i + dt])
                if not np.isnan(d):
                    deltas[bucket].append(d)

    med = {f"median_norm_diff_dt{k}": float(np.median(v)) for k, v in deltas.items() if v}
    med["interpretation"] = (
        "Малый Δ при dt=1 → соседние кадры почти дубли; 202k кадров ≠ 202k независимых выборок."
    )
    med["heuristic_effective_factor"] = round(
        med.get("median_norm_diff_dt1", 0.02) / max(med.get("median_norm_diff_dt30", 0.15), 1e-6), 2
    )
    return med


def analyze_fps() -> dict:
    fps_list = []
    duration_sec = []
    for folder in (ARD100_ROOT / "train_videos", ARD100_ROOT / "test_videos"):
        if not folder.is_dir():
            continue
        for mp4 in sorted(folder.glob("*.mp4")):
            cap = cv2.VideoCapture(str(mp4))
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            if fps > 0:
                fps_list.append(fps)
                duration_sec.append(n / fps)
    arr = np.array(fps_list)
    dur = np.array(duration_sec)
    return {
        "n_videos": len(fps_list),
        "fps_unique": sorted(set(round(x, 3) for x in fps_list)),
        "fps_median": float(np.median(arr)),
        "duration_sec_median": float(np.median(dur)),
        "duration_sec_mean": float(np.mean(dur)),
    }


def _e_bbox_from_fd(fd_bgr: np.ndarray, xc: float, yc: float, w: float, h: float) -> float:
    H, W = fd_bgr.shape[:2]
    x1 = int(max(0, (xc - w / 2) * W))
    x2 = int(min(W, (xc + w / 2) * W))
    y1 = int(max(0, (yc - h / 2) * H))
    y2 = int(min(H, (yc + h / 2) * H))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = fd_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(np.mean(np.abs(patch.astype(np.float32))))


def analyze_rel_area_e_bbox_val() -> dict:
    """2D: rel_area × e_bbox на val (нативный images2)."""
    rel_list: list[float] = []
    eb_list: list[float] = []
    cross: Counter[str] = Counter()

    pairs = []
    for img_path in _read_split_list("val.txt"):
        if "/mixed_photos/" in img_path.replace("\\", "/"):
            continue
        lbl = _label_path_from_img(img_path)
        boxes = _parse_yolo_label(lbl)
        if not boxes:
            continue
        stem = Path(img_path).stem
        vid = _scene(stem)
        fd_path = IMAGES2_OUT / vid / f"{stem}.jpg"
        fd = cv2.imread(str(fd_path))
        if fd is None:
            continue
        for xc, yc, w, h in boxes:
            rel = w * h
            eb = _e_bbox_from_fd(fd, xc, yc, w, h)
            rel_list.append(rel)
            eb_list.append(eb)
            pairs.append((rel, eb))

    if not pairs:
        return {"error": "no pairs"}

    eb_arr = np.array([p[1] for p in pairs])
    q_edges = [np.quantile(eb_arr, q) for q in (0.2, 0.4, 0.6, 0.8)]

    def eb_bin(e: float) -> str:
        if e <= q_edges[0]:
            return "Eb_Q0"
        if e <= q_edges[1]:
            return "Eb_Q1"
        if e <= q_edges[2]:
            return "Eb_Q2"
        if e <= q_edges[3]:
            return "Eb_Q3"
        return "Eb_Q4"

    for rel, eb in pairs:
        ra = _bin_count(rel, REL_AREA_BINS)
        cross[f"{ra}|{eb_bin(eb)}"] += 1

    tau_q20 = float(np.quantile(eb_arr, 0.2))
    small_far = sum(1 for rel, eb in pairs if rel < 5e-5 and eb <= tau_q20)
    return {
        "n_gt_val_video": len(pairs),
        "e_bbox_q20_tau": round(tau_q20, 4),
        "e_bbox_median": float(np.median(eb_arr)),
        "rel_area_median": float(np.median([p[0] for p in pairs])),
        "joint_small_rel_and_low_eb": small_far,
        "joint_small_rel_and_low_eb_pct": round(100 * small_far / len(pairs), 1),
        "cross_tab_top10": dict(cross.most_common(10)),
        "cross_tab_full": dict(cross),
    }


def analyze_sharpness(sample_n: int = 2500) -> dict:
    rng = random.Random(0)
    all_paths = _read_split_list("train.txt") + _read_split_list("val.txt") + _read_split_list("test.txt")
    sample = rng.sample(all_paths, min(sample_n, len(all_paths)))
    vars_lap = []
    for p in sample:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = cv2.resize(im, (640, 360))
        lap = cv2.Laplacian(im, cv2.CV_64F)
        vars_lap.append(float(lap.var()))
    arr = np.array(vars_lap)
    p10 = float(np.percentile(arr, 10))
    blurry = int(np.sum(arr <= p10))
    return {
        "sample_n": len(vars_lap),
        "laplacian_var_median": float(np.median(arr)),
        "laplacian_var_p10": p10,
        "bottom_decile_blurry_count": blurry,
        "bottom_decile_pct": round(100 * blurry / len(arr), 1),
        "note": "Нижний дециль по резкости — эвристика motion blur / расфокус.",
    }


def scene_table_markdown(split_data: dict) -> str:
    lines = ["| Сцена (phantom) | Train | Val | Test | Всего |", "|-----------------|------:|----:|-----:|------:|"]
    all_scenes = set()
    for sp in ("train", "val", "test"):
        all_scenes.update(split_data[sp]["scenes"].keys())
    totals = Counter()
    for sc in sorted(all_scenes, key=lambda x: (int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0, x)):
        tr = split_data["train"]["scenes"].get(sc, 0)
        va = split_data["val"]["scenes"].get(sc, 0)
        te = split_data["test"]["scenes"].get(sc, 0)
        tot = tr + va + te
        totals["tr"] += tr
        totals["va"] += va
        totals["te"] += te
        lines.append(f"| {sc} | {tr} | {va} | {te} | {tot} |")
    lines.append(f"| **Итого** | **{totals['tr']}** | **{totals['va']}** | **{totals['te']}** | **{sum(totals.values())}** |")
    return "\n".join(lines)


def main() -> None:
    print("ARD100 analytics…", flush=True)
    split_data = analyze_splits()
    sizes = analyze_image_sizes()
    bbox = analyze_bbox_and_objects()
    print("  day/night…", flush=True)
    daynight = analyze_day_night(sample_per_scene=5, max_scenes=None)
    print("  extended (1,2,3,7,8,9)…", flush=True)
    author = analyze_author_split_match()
    storage = analyze_storage()
    voc = analyze_voc_quality()
    empty = analyze_empty_labels_by_split()
    temporal = analyze_temporal_correlation()
    fps = analyze_fps()
    print("  val rel×e_bbox…", flush=True)
    joint = analyze_rel_area_e_bbox_val()
    sharp = analyze_sharpness()

    report = {
        "splits": split_data,
        "image_sizes": sizes,
        "bbox_objects": bbox,
        "day_night": daynight,
        "author_split": author,
        "storage": storage,
        "voc_quality": voc,
        "empty_labels": empty,
        "temporal_correlation": temporal,
        "fps": fps,
        "rel_area_e_bbox_val": joint,
        "sharpness": sharp,
    }
    out_json = _REPO / "experiments" / "ard100_chapter2_analytics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON: {out_json}")

    md = _REPO / "experiments" / "ard100_chapter2_analytics.md"
    md.write_text(
        "\n".join(
            [
                "# ARD100 — аналитика для гл. 2",
                "",
                "## Сводка по сплитам",
                f"- Всего сцен (phantom): **{split_data['total_scenes_in_corpus']}** (train pool {split_data['set_train_phantoms']}, test {split_data['set_test_phantoms']})",
                f"- Train: **{split_data['train']['n_frames']}** кадров, **{split_data['train']['n_scenes']}** сцен",
                f"- Val: **{split_data['val']['n_frames']}** кадров, **{split_data['val']['n_scenes']}** сцен",
                f"- Test: **{split_data['test']['n_frames']}** кадров, **{split_data['test']['n_scenes']}** сцен",
                "",
                "## Кадры по сценам",
                scene_table_markdown(split_data),
                "",
                "## Размеры изображений",
                f"- VOC/XML: {sizes['from_voc_xml']}",
                f"- Проверка jpg (n={sizes['jpg_sample_n']}): {sizes['jpg_sample_sizes']}",
                "",
                "## Bbox / дальность (proxy)",
                json.dumps(bbox["rel_area"], ensure_ascii=False, indent=2),
                f"- Proxy дальности: {bbox['distance_proxy_bins']}",
                f"- Ширина bbox (px): {bbox['bbox_width_px']}",
                f"- Высота bbox (px): {bbox['bbox_height_px']}",
                "",
                "## Объекты на изображение",
                json.dumps(bbox["objects_per_image"], ensure_ascii=False, indent=2),
                "",
                "## День / ночь (эвристика)",
                json.dumps(daynight, ensure_ascii=False, indent=2),
                "",
                "## Расширенная аналитика (гл. 2 доп.)",
                json.dumps(
                    {
                        "author_split": author,
                        "storage": storage,
                        "voc_quality": voc,
                        "empty_labels": empty,
                        "temporal": temporal,
                        "fps": fps,
                        "rel_e_bbox": joint,
                        "sharpness": sharp,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        ),
        encoding="utf-8",
    )
    print(f"MD: {md}")
    print(json.dumps({k: split_data[k] for k in ("train", "val", "test")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
