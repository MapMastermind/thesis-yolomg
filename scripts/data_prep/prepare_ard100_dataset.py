#!/usr/bin/env python3
"""
Сборка ARD100 для YOLOMG (dual: RGB + motion mask) под этот сервер.

Источник: /data/ARD100/{train_videos,test_videos,annotations}
Вывод:   /home/tanyadiplom/data/ard100/{images,images2,labels} + train*.txt

Запуск из корня Static_YOLOMG:
  ./.venv/bin/python scripts/prepare_ard100_dataset.py

Опции:
  --skip-extract   не извлекать кадры из mp4 (если images/ уже готовы)
  --skip-masks     не считать FD5-маски (если images2/ уже готовы)
  --skip-labels    не конвертировать VOC→YOLO и не писать списки
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# test_code на PYTHONPATH
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "test_code"))

from FD5_mask import FD5_mask  # noqa: E402

from paths_ard100 import (  # noqa: E402
    ANNOTATIONS,
    IMAGES2_OUT,
    IMAGES_OUT,
    LABELS_OUT,
    OUT_ROOT,
    TEST_VIDEOS,
    TRAIN_VIDEOS,
)

# Сплит роликов как у авторов в generate_dataset.py
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

CLASSES = ["Drone"]
VAL_FRACTION = 0.1
VAL_SEED = 42


def _ensure_dirs() -> None:
    for d in (IMAGES_OUT, IMAGES2_OUT, LABELS_OUT):
        os.makedirs(d, exist_ok=True)


def _list_mp4_dirs() -> list[tuple[str, Path]]:
    """(video_id, path_to_mp4)"""
    out: list[tuple[str, Path]] = []
    for folder in (TRAIN_VIDEOS, TEST_VIDEOS):
        p = Path(folder)
        if not p.is_dir():
            print(f"Пропуск (нет каталога): {folder}")
            continue
        for mp4 in sorted(p.glob("*.mp4")):
            vid = mp4.stem
            out.append((vid, mp4))
    return out


def extract_frames(skip: bool) -> None:
    if skip:
        print("Шаг extract: пропуск (--skip-extract)")
        return
    for vid, mp4 in _list_mp4_dirs():
        out_dir = os.path.join(IMAGES_OUT, vid)
        os.makedirs(out_dir, exist_ok=True)
        vc = cv2.VideoCapture(str(mp4))
        if not vc.isOpened():
            print(f"Не открыть видео: {mp4}")
            continue
        c = 0
        while True:
            ok, frame = vc.read()
            if not ok:
                break
            c += 1
            name = str(c).zfill(4)
            fn = f"{vid}_{name}.jpg"
            cv2.imwrite(os.path.join(out_dir, fn), frame)
        vc.release()
        print(f"Кадры: {vid} -> {c} файлов в {out_dir}")


def process_masks_for_video(video_name: str, mp4_path: Path) -> None:
    cap = cv2.VideoCapture(str(mp4_path))
    lastFrame1 = lastFrame2 = lastFrame3 = lastFrame4 = None
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        currentFrame = frame
        count += 1
        if lastFrame4 is None:
            if lastFrame3 is None:
                if lastFrame2 is None:
                    if lastFrame1 is None:
                        lastFrame1 = currentFrame
                    else:
                        lastFrame2 = currentFrame
                else:
                    lastFrame3 = currentFrame
            else:
                lastFrame4 = currentFrame
            continue
        FD5_mask(lastFrame1, lastFrame3, currentFrame, video_name, count - 2, mask_root=IMAGES2_OUT)
        lastFrame1 = lastFrame2
        lastFrame2 = lastFrame3
        lastFrame3 = lastFrame4
        lastFrame4 = currentFrame
    cap.release()


def generate_masks(skip: bool) -> None:
    if skip:
        print("Шаг masks: пропуск (--skip-masks)")
        return
    for vid, mp4 in _list_mp4_dirs():
        print(f"Маски FD5: {vid} …")
        process_masks_for_video(vid, mp4)
        backfill_masks(vid)


def backfill_masks(video_name: str) -> None:
    """Первые кадры без FD5 — заполняем ближайшей существующей маской или чёрным кадром."""
    rgb_dir = os.path.join(IMAGES_OUT, video_name)
    mdir = os.path.join(IMAGES2_OUT, video_name)
    if not os.path.isdir(rgb_dir):
        return
    jpgs = sorted(f for f in os.listdir(rgb_dir) if f.lower().endswith(".jpg"))
    if not jpgs:
        return
    os.makedirs(mdir, exist_ok=True)
    first_rgb = cv2.imread(os.path.join(rgb_dir, jpgs[0]))
    if first_rgb is None:
        return
    h, w = first_rgb.shape[:2]
    black = np.zeros((h, w), dtype=np.uint8)
    existing: dict[int, np.ndarray] = {}
    for f in os.listdir(mdir):
        if not f.endswith(".jpg"):
            continue
        parts = f.replace(".jpg", "").split("_")
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[-1])
        except ValueError:
            continue
        p = os.path.join(mdir, f)
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            existing[idx] = im
    for jf in jpgs:
        stem = jf[:-4]
        try:
            idx = int(stem.split("_")[-1])
        except ValueError:
            continue
        out_m = os.path.join(mdir, f"{video_name}_{str(idx).zfill(4)}.jpg")
        if os.path.isfile(out_m):
            continue
        donor = None
        for d in range(0, 30):
            if idx + d in existing:
                donor = existing[idx + d]
                break
            if idx - d in existing:
                donor = existing[idx - d]
                break
        if donor is None:
            donor = black
        cv2.imwrite(out_m, donor)
    print(f"  backfill масок: {video_name}")


def convert_size(box: tuple[float, float, float, float], wh: tuple[int, int]) -> tuple[float, float, float, float]:
    dw, dh = 1.0 / wh[0], 1.0 / wh[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    bw, bh = box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, bw * dw, bh * dh


def convert_xml_to_yolo(xml_path: Path, out_txt: Path, min_area: float) -> bool:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return False
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    lines: list[str] = []
    for obj in root.iter("object"):
        difficult = obj.find("difficult")
        if difficult is not None and int(difficult.text) == 1:
            continue
        name_el = obj.find("name")
        if name_el is None or name_el.text not in CLASSES:
            continue
        cls_id = CLASSES.index(name_el.text)
        xmlbox = obj.find("bndbox")
        if xmlbox is None:
            continue
        b1 = float(xmlbox.find("xmin").text)
        b2 = float(xmlbox.find("xmax").text)
        b3 = float(xmlbox.find("ymin").text)
        b4 = float(xmlbox.find("ymax").text)
        if b1 <= 0:
            b1 = 1
        if b3 <= 0:
            b3 = 1
        if b2 > w:
            b2 = float(w)
        if b4 > h:
            b4 = float(h)
        area = (b2 - b1) * (b4 - b3)
        if area < min_area:
            continue
        bb = convert_size((b1, b2, b3, b4), (w, h))
        lines.append(str(cls_id) + " " + " ".join(f"{a:.6f}" for a in bb))
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""))
    return bool(lines)


def build_labels(min_area: float, skip: bool) -> None:
    if skip:
        print("Шаг labels: пропуск (--skip-labels)")
        return
    n_ok = 0
    for xml_path in sorted(Path(ANNOTATIONS).glob("*/*.xml")):
        stem = xml_path.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        vid = parts[0]
        img_path = Path(IMAGES_OUT) / vid / f"{stem}.jpg"
        if not img_path.is_file():
            continue
        rel = Path(vid) / f"{stem}.txt"
        out_txt = Path(LABELS_OUT) / rel
        if convert_xml_to_yolo(xml_path, out_txt, min_area):
            n_ok += 1
    print(f"VOC→YOLO: записано размеченных label-файлов: {n_ok}")


def _stem_in_split(stem: str, split: set[str]) -> bool:
    vid = stem.split("_")[0]
    return vid in split


def _is_ready(stem: str) -> bool:
    vid = stem.split("_")[0]
    img = Path(IMAGES_OUT) / vid / f"{stem}.jpg"
    im2 = Path(IMAGES2_OUT) / vid / f"{stem}.jpg"
    lbl = Path(LABELS_OUT) / vid / f"{stem}.txt"
    return img.is_file() and im2.is_file() and lbl.is_file()


def write_lists() -> None:
    stems: list[str] = []
    for vid_dir in sorted(Path(IMAGES_OUT).iterdir()):
        if not vid_dir.is_dir():
            continue
        for jf in sorted(vid_dir.glob("*.jpg")):
            stems.append(jf.stem)

    train_pool = [s for s in stems if _stem_in_split(s, SET_TRAIN) and _is_ready(s)]
    test_list = [s for s in stems if _stem_in_split(s, SET_TEST) and _is_ready(s)]

    rnd = random.Random(VAL_SEED)
    train_pool_shuffled = train_pool[:]
    rnd.shuffle(train_pool_shuffled)
    n_val = max(1, int(len(train_pool_shuffled) * VAL_FRACTION)) if train_pool_shuffled else 0
    val_set = set(train_pool_shuffled[:n_val])
    train_final = [s for s in train_pool if s not in val_set]

    def abspath_img(stem: str) -> str:
        vid = stem.split("_")[0]
        return str((Path(IMAGES_OUT) / vid / f"{stem}.jpg").resolve())

    def abspath_img2(stem: str) -> str:
        vid = stem.split("_")[0]
        return str((Path(IMAGES2_OUT) / vid / f"{stem}.jpg").resolve())

    def write_list(name: str, items: list[str]) -> None:
        p = Path(OUT_ROOT) / name
        with open(p, "w") as f:
            for s in sorted(items):
                f.write(abspath_img(s) + "\n")
        print(f"Записан {p} ({len(items)} строк)")

    def write_list2(name: str, items: list[str]) -> None:
        p = Path(OUT_ROOT) / name
        with open(p, "w") as f:
            for s in sorted(items):
                f.write(abspath_img2(s) + "\n")
        print(f"Записан {p} ({len(items)} строк)")

    write_list("train.txt", train_final)
    write_list2("train2.txt", train_final)
    write_list("val.txt", sorted(val_set))
    write_list2("val2.txt", sorted(val_set))
    write_list("test.txt", test_list)
    write_list2("test2.txt", test_list)


def patch_yaml() -> None:
    ypath = _REPO / "data" / "ard100.yaml"
    text = f"""# Автогенерация: scripts/prepare_ard100_dataset.py
train: {OUT_ROOT}/train.txt
train2: {OUT_ROOT}/train2.txt
val: {OUT_ROOT}/val.txt
val2: {OUT_ROOT}/val2.txt
test: {OUT_ROOT}/test.txt

nc: 1
names: ['Drone']
"""
    ypath.write_text(text)
    print(f"Обновлён {ypath}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-masks", action="store_true")
    ap.add_argument("--skip-labels", action="store_true")
    ap.add_argument("--min-bbox-area", type=float, default=0.0, help="как у авторов для части пайплайна часто 25; 0 = не фильтровать")
    args = ap.parse_args()

    print("OUT_ROOT:", OUT_ROOT)
    print("Источник видео:", TRAIN_VIDEOS, TEST_VIDEOS)
    print("Аннотации:", ANNOTATIONS)
    _ensure_dirs()

    extract_frames(args.skip_extract)
    generate_masks(args.skip_masks)
    build_labels(args.min_bbox_area, args.skip_labels)
    if not args.skip_labels:
        write_lists()
        patch_yaml()


if __name__ == "__main__":
    main()
