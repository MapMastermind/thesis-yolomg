#!/usr/bin/env python3
"""
Метрики на val ARD100 mixed: отдельно «фото» (/mixed_photos/) и «видео» (остальные пути).

Фото: второй поток = CLAHE→DoG из letterbox RGB (как YOLOMG_PHOTO_STREAM_MODE=clahe_dog при mixed_policy),
  либо нули (--photo-zero-stream2, как YOLOMG_MIXED_PHOTO_ZERO_STREAM2=1).
Видео: второй поток = native с диска (val2), letterbox как в compare_ard100_model_errors.

Для каждого GT-бокса: best_iou = max_j IoU(gt_i, pred_j), conf_best = conf предикта с максимальным IoU
(при равенстве IoU — больший conf). NMS: conf_thres низкий, затем отбор предиктов.

Опция --photo-tile: SAHI-style срезы полноразмерного фото (overlap), инференс на каждом тайле
(CLAHE→DoG как на полном кадре), перевод боксов в координаты полного кадра, merge + NMS.

Счётчики:
  - hit-rate @0.5 / @0.75: доля GT с best_iou >= порога
  - mean/median best_iou по GT
  - mean conf_best по всем GT (если нет предиктов — conf=0)
  - pred rate (image): доля кадров с хотя бы одним боксом после NMS(conf_thres)
  - pred rate (conf>=t): доля кадров с хотя бы одним боксом с conf>=t
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_orig_torch_load = torch.load


def _torch_load_weights(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.clip(union, 1e-6, None)).astype(np.float32)


def yolo_txt_to_xyxy(path: Path, w: int, h: int) -> np.ndarray:
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


def label_path_from_rgb(rgb_path: Path) -> Path:
    s = str(rgb_path).replace("\\", "/")
    return Path(s.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")


def stream2_zero_bgr(im_lb_bgr: np.ndarray) -> np.ndarray:
    """Второй поток как при mixed_policy + YOLOMG_MIXED_PHOTO_ZERO_STREAM2=1."""
    return np.zeros_like(im_lb_bgr)


def clahe_dog_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Совпадает с utils.datasets.LoadImagesAndLabels._clahe_dog_second_stream_from_rgb."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    b9 = cv2.GaussianBlur(g, (9, 9), 0)
    b21 = cv2.GaussianBlur(g, (21, 21), 0)
    b41 = cv2.GaussianBlur(g, (41, 41), 0)
    d1 = cv2.absdiff(g, b9)
    d2 = cv2.absdiff(g, b21)
    d3 = cv2.absdiff(g, b41)
    return cv2.merge([d1, d2, d3])

def dog_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Multi-scale DoG по серому (как LoadImagesAndLabels._dog_second_stream_from_rgb)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    b9 = cv2.GaussianBlur(gray, (9, 9), 0)
    b21 = cv2.GaussianBlur(gray, (21, 21), 0)
    b41 = cv2.GaussianBlur(gray, (41, 41), 0)
    d1 = cv2.absdiff(gray, b9)
    d2 = cv2.absdiff(gray, b21)
    d3 = cv2.absdiff(gray, b41)
    return cv2.merge([d1, d2, d3])


def motion_edge_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Совпадает с utils.datasets.LoadImagesAndLabels._motion_edge_second_stream_from_rgb."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    j11 = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
    j22 = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
    j12 = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)
    tr = j11 + j22 + 1e-6
    det_term = np.sqrt(np.maximum((j11 - j22) * (j11 - j22) + 4.0 * j12 * j12, 0.0))
    l1 = 0.5 * (tr + det_term)
    l2 = 0.5 * (tr - det_term)
    anis = (l1 - l2) / (l1 + l2 + 1e-6)
    anis_u8 = np.clip(anis * 255.0, 0, 255).astype(np.uint8)

    b7 = cv2.GaussianBlur(g, (7, 7), 0)
    b31 = cv2.GaussianBlur(g, (31, 31), 0)
    dog = cv2.absdiff(b7, b31)
    return cv2.merge([mag_u8, anis_u8, dog])


def photo_tile_origins(dim: int, tile: int, overlap: float) -> list[int]:
    """Начала срезов по одной оси (в пикселях исходного кадра). overlap ∈ [0, 0.95]."""
    tile = int(tile)
    if tile <= 0:
        return [0]
    if dim <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - float(overlap)))))
    positions: list[int] = []
    pos = 0
    while pos + tile <= dim:
        positions.append(pos)
        nxt = pos + step
        if nxt + tile > dim:
            break
        pos = nxt
    last = max(0, dim - tile)
    if not positions:
        return [0]
    if positions[-1] != last:
        positions.append(last)
    return sorted(set(positions))


def photo_tile_slices(h: int, w: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Список (y0, x0, y1, x1) включительно-эксклюзивно для im0[y0:y1, x0:x1]."""
    ys = photo_tile_origins(h, tile, overlap)
    xs = photo_tile_origins(w, tile, overlap)
    out: list[tuple[int, int, int, int]] = []
    for y0 in ys:
        for x0 in xs:
            y1 = min(h, y0 + tile)
            x1 = min(w, x0 + tile)
            out.append((y0, x0, y1, x1))
    return out


def nms_xyxy_torch(
    xyxy: np.ndarray, conf: np.ndarray, iou_thres: float, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """torchvision.ops.nms по xyxy (пиксели)."""
    if xyxy.size == 0:
        return xyxy, conf
    import torchvision

    boxes = torch.as_tensor(xyxy, dtype=torch.float32, device=device)
    scores = torch.as_tensor(conf, dtype=torch.float32, device=device)
    keep = torchvision.ops.nms(boxes, scores, float(iou_thres))
    keep = keep.cpu().numpy()
    return xyxy[keep], conf[keep]


def infer_photo_tiled(
    y5: torch.nn.Module,
    im0: np.ndarray,
    dev: torch.device,
    imgsz: int,
    stride: int,
    conf_nms: float,
    iou_nms: float,
    tile: int,
    overlap: float,
    non_max_suppression,
    scale_coords,
    letterbox,
    stream2_fn,
) -> tuple[np.ndarray, np.ndarray]:
    h0, w0 = im0.shape[:2]
    slices = photo_tile_slices(h0, w0, tile, overlap)
    all_xy: list[np.ndarray] = []
    all_cf: list[np.ndarray] = []
    for (y0, x0, y1, x1) in slices:
        crop = im0[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        im_lb, _, _ = letterbox(crop, (imgsz, imgsz), auto=True, stride=stride)
        im_lb2 = stream2_fn(im_lb)
        xy, cf = run_inference_batch(
            y5,
            im_lb,
            im_lb2,
            (ch, cw),
            dev,
            conf_nms,
            iou_nms,
            non_max_suppression,
            scale_coords,
        )
        if xy.size:
            xy = xy.copy()
            xy[:, [0, 2]] += float(x0)
            xy[:, [1, 3]] += float(y0)
            all_xy.append(xy)
            all_cf.append(cf)
    if not all_xy:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    xy_cat = np.vstack(all_xy)
    cf_cat = np.concatenate(all_cf)
    xy_cat, cf_cat = nms_xyxy_torch(xy_cat, cf_cat, iou_nms, dev)
    xy_cat = xy_cat.copy()
    xy_cat[:, [0, 2]] = xy_cat[:, [0, 2]].clip(0.0, float(w0))
    xy_cat[:, [1, 3]] = xy_cat[:, [1, 3]].clip(0.0, float(h0))
    return xy_cat, cf_cat


def load_val_pairs(val_txt: Path, val2_txt: Path) -> list[tuple[Path, Path]]:
    a = [Path(x.strip()) for x in val_txt.read_text().splitlines() if x.strip() and not x.strip().startswith("#")]
    b = [Path(x.strip()) for x in val2_txt.read_text().splitlines() if x.strip() and not x.strip().startswith("#")]
    if len(a) != len(b):
        raise SystemExit(f"val.txt ({len(a)}) и val2.txt ({len(b)}) разной длины")
    return list(zip(a, b))


def run_inference_batch(
    y5: torch.nn.Module,
    im_lb: np.ndarray,
    im_lb2: np.ndarray,
    im0_hw: tuple[int, int],
    dev: torch.device,
    conf_thres: float,
    iou_nms: float,
    non_max_suppression,
    scale_coords,
    area_dependent_conf: bool = False,
    conf_base: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    im_t = torch.from_numpy(np.ascontiguousarray(im_lb.transpose((2, 0, 1))[::-1])).to(dev)
    im_t = im_t.float().unsqueeze(0) / 255.0
    im_t2 = torch.from_numpy(np.ascontiguousarray(im_lb2.transpose((2, 0, 1))[::-1])).to(dev)
    im_t2 = im_t2.float().unsqueeze(0) / 255.0
    with torch.no_grad():
        raw, _ = y5(im_t, im_t2)
    det = non_max_suppression(raw, conf_thres, iou_nms, agnostic=False)[0]
    if det is None or len(det) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    det = det.clone()
    # Как в compare_ard100_model_errors: из letterbox-тензора в оригинальный кадр (h0, w0)
    scale_coords(im_t.shape[2:], det[:, :4], im0_hw)
    pred_xyxy = det[:, :4].cpu().numpy().astype(np.float32)
    pred_conf = det[:, 4].cpu().numpy().astype(np.float32)
    if area_dependent_conf and pred_xyxy.shape[0] > 0:
        from utils.infer_area_conf import filter_preds_area_conf

        pred_xyxy, pred_conf = filter_preds_area_conf(
            pred_xyxy, pred_conf, im0_hw, conf_base=conf_base, use_area_dependent=True
        )
    return pred_xyxy, pred_conf


def aggregate_split(
    name: str,
    per_image: list[dict],
) -> dict:
    n_img = len(per_image)
    n_gt = sum(x["n_gt"] for x in per_image)
    if n_gt == 0:
        return {"name": name, "n_images": n_img, "n_gt": 0}

    best_ious = []
    best_confs = []
    hit50 = hit75 = 0
    for x in per_image:
        for bi, iou in enumerate(x["best_iou_per_gt"]):
            best_ious.append(float(iou))
            best_confs.append(float(x["best_conf_per_gt"][bi]))
            if iou >= 0.5:
                hit50 += 1
            if iou >= 0.75:
                hit75 += 1

    arr_iou = np.array(best_ious, dtype=np.float64)
    arr_c = np.array(best_confs, dtype=np.float64)
    pred_any = sum(1 for x in per_image if x["n_pred"] > 0)
    pred_ge = sum(1 for x in per_image if x["n_pred_ge_thr"] > 0)

    return {
        "name": name,
        "n_images": n_img,
        "n_gt": n_gt,
        "hit_rate_iou_0.5": hit50 / n_gt,
        "hit_rate_iou_0.75": hit75 / n_gt,
        "mean_best_iou": float(arr_iou.mean()),
        "median_best_iou": float(np.median(arr_iou)),
        "mean_conf_at_best_iou": float(arr_c.mean()),
        "pred_rate_any_box": pred_any / n_img,
        "pred_rate_conf_ge": pred_ge / n_img,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--val-txt", type=Path, default=Path("/home/tanyadiplom/data/ard100_mixed/val.txt"))
    ap.add_argument("--val2-txt", type=Path, default=Path("/home/tanyadiplom/data/ard100_mixed/val2.txt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf-nms", type=float, default=0.001, help="Порог conf перед NMS (как val)")
    ap.add_argument("--iou-nms", type=float, default=0.6)
    ap.add_argument("--pred-rate-conf", type=float, default=0.25, help="Второй pred rate: доля кадров с conf>=this")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Печатать прогресс каждые N обработанных val-пар (0 = только старт/финиш)",
    )
    ap.add_argument(
        "--photo-tile",
        type=int,
        default=0,
        metavar="PX",
        help="Для /mixed_photos/: размер квадратного среза в пикселях на полном кадре (0 = один letterbox как раньше)",
    )
    ap.add_argument(
        "--photo-tile-overlap",
        type=float,
        default=0.25,
        help="Доля overlap между срезами (0..0.9), только при --photo-tile > 0",
    )
    ap.add_argument(
        "--photo-stream-mode",
        choices=("clahe_dog", "dog", "motion_edge"),
        default="clahe_dog",
        help="Режим синтеза второго потока для фото (/mixed_photos/); игнорируется, если задан --photo-zero-stream2.",
    )
    ap.add_argument(
        "--photo-zero-stream2",
        action="store_true",
        help="Второй поток для фото = нули (как при обучении с YOLOMG_MIXED_PHOTO_ZERO_STREAM2=1).",
    )
    ap.add_argument(
        "--photo-only",
        action="store_true",
        help="Только кадры /mixed_photos/ (пропуск видео, быстрее).",
    )
    ap.add_argument(
        "--area-dependent-conf",
        action="store_true",
        help="S-1: после NMS отфильтровать предсказания по conf(rel_area); мелкие боксы — выше порог.",
    )
    args = ap.parse_args()
    if not (0.0 <= args.photo_tile_overlap < 0.95):
        raise SystemExit("--photo-tile-overlap must be in [0, 0.95)")

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from models.experimental import attempt_load
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_coords

    pairs = load_val_pairs(args.val_txt, args.val2_txt)
    n_pairs = len(pairs)
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    wpath = args.weights if args.weights.is_absolute() else root / args.weights
    t0 = time.perf_counter()
    tile_info = ""
    if args.photo_tile > 0:
        tile_info = f" photo_tile={args.photo_tile}px overlap={args.photo_tile_overlap}"
    if args.photo_zero_stream2:
        photo_stream_fn = stream2_zero_bgr
        photo_stream_name = "photo_zero_stream2"
        photo_stream_mode_out = "zero"
    else:
        if args.photo_stream_mode == "clahe_dog":
            photo_stream_fn = clahe_dog_bgr
        elif args.photo_stream_mode == "dog":
            photo_stream_fn = dog_bgr
        else:
            photo_stream_fn = motion_edge_bgr
        photo_stream_name = f"photo_{args.photo_stream_mode}_stream2"
        photo_stream_mode_out = args.photo_stream_mode
    extra = ""
    if args.photo_only:
        extra += " photo_only=1"
    if args.photo_zero_stream2:
        extra += " photo_zero_stream2=1"
    print(f"[eval] pairs={n_pairs} weights={wpath} device={dev} imgsz={args.imgsz}{tile_info}{extra}", flush=True)
    y5 = attempt_load(str(wpath), map_location=dev)
    y5.eval()
    stride = int(y5.stride.max())
    print(f"[eval] model loaded stride={stride} ({time.perf_counter() - t0:.1f}s)", flush=True)

    photo_images: list[dict] = []
    video_images: list[dict] = []

    for idx, (ipath, ipath2) in enumerate(pairs):
        pnorm = str(ipath).replace("\\", "/")
        is_photo = "/mixed_photos/" in pnorm
        if args.photo_only and not is_photo:
            continue

        im0 = cv2.imread(str(ipath))
        im02 = cv2.imread(str(ipath2)) if not args.photo_only or not is_photo else None
        if im0 is None:
            continue
        h0, w0 = im0.shape[:2]
        gt = yolo_txt_to_xyxy(label_path_from_rgb(ipath), w0, h0)
        n_gt = len(gt)

        if is_photo and args.photo_tile > 0:
            pred_xyxy, pred_conf = infer_photo_tiled(
                y5,
                im0,
                dev,
                args.imgsz,
                stride,
                args.conf_nms,
                args.iou_nms,
                args.photo_tile,
                args.photo_tile_overlap,
                non_max_suppression,
                scale_coords,
                letterbox,
                photo_stream_fn,
            )
        elif is_photo:
            im_lb, _, _ = letterbox(im0, (args.imgsz, args.imgsz), auto=True, stride=stride)
            im2_syn = photo_stream_fn(im_lb)
            im_lb2 = im2_syn
            pred_xyxy, pred_conf = run_inference_batch(
                y5,
                im_lb,
                im_lb2,
                (h0, w0),
                dev,
                args.conf_nms,
                args.iou_nms,
                non_max_suppression,
                scale_coords,
                area_dependent_conf=args.area_dependent_conf,
                conf_base=args.pred_rate_conf,
            )
        else:
            if im02 is None:
                continue
            im_lb, _, _ = letterbox(im0, (args.imgsz, args.imgsz), auto=True, stride=stride)
            im_lb2, _, _ = letterbox(im02, (args.imgsz, args.imgsz), auto=True, stride=stride)
            pred_xyxy, pred_conf = run_inference_batch(
                y5,
                im_lb,
                im_lb2,
                (h0, w0),
                dev,
                args.conf_nms,
                args.iou_nms,
                non_max_suppression,
                scale_coords,
                area_dependent_conf=args.area_dependent_conf,
                conf_base=args.pred_rate_conf,
            )

        n_pred = int(pred_xyxy.shape[0])
        n_pred_ge = int((pred_conf >= args.pred_rate_conf).sum()) if n_pred else 0

        best_iou = []
        best_cf = []
        if n_gt == 0:
            rec = {
                "path": str(ipath),
                "n_gt": 0,
                "n_pred": n_pred,
                "n_pred_ge_thr": n_pred_ge,
                "best_iou_per_gt": [],
                "best_conf_per_gt": [],
            }
            (photo_images if is_photo else video_images).append(rec)
            continue

        if n_pred == 0:
            best_iou = [0.0] * n_gt
            best_cf = [0.0] * n_gt
        else:
            ious = box_iou_xyxy(gt, pred_xyxy)
            for gi in range(n_gt):
                row = ious[gi]
                j = int(np.argmax(row))
                mx = float(row[j])
                js = np.where(row >= mx - 1e-6)[0]
                j_best = int(js[np.argmax(pred_conf[js])])
                best_iou.append(float(row[j_best]))
                best_cf.append(float(pred_conf[j_best]))

        rec = {
            "path": str(ipath),
            "n_gt": n_gt,
            "n_pred": n_pred,
            "n_pred_ge_thr": n_pred_ge,
            "best_iou_per_gt": best_iou,
            "best_conf_per_gt": best_cf,
        }
        (photo_images if is_photo else video_images).append(rec)

        pe = args.progress_every
        if pe and (idx + 1) % pe == 0:
            dt = time.perf_counter() - t0
            n_ph = len(photo_images)
            n_vid = len(video_images)
            rate = (idx + 1) / max(dt, 1e-6)
            print(
                f"[eval] {idx + 1}/{n_pairs}  photo_frames={n_ph}  video_frames={n_vid}  "
                f"elapsed={dt:.0f}s  ~{rate:.2f} img/s",
                flush=True,
            )

    print(f"[eval] done all {n_pairs} pairs in {time.perf_counter() - t0:.1f}s", flush=True)

    out: dict = {
        "weights": str(wpath),
        "imgsz": args.imgsz,
        "conf_nms_input": args.conf_nms,
        "iou_nms": args.iou_nms,
        "pred_rate_conf_threshold": args.pred_rate_conf,
        "photo_stream_mode": photo_stream_mode_out,
        "photo_only": bool(args.photo_only),
        "mixed_photo_zero_stream2": bool(args.photo_zero_stream2),
        "photo": aggregate_split(
            f"{photo_stream_name}_tiled" if args.photo_tile > 0 else photo_stream_name,
            photo_images,
        ),
        "video": aggregate_split("video_native_stream2", video_images),
    }
    if args.photo_tile > 0:
        out["photo_tiling"] = {
            "tile_size_px": args.photo_tile,
            "overlap_ratio": args.photo_tile_overlap,
            "merge_nms_iou": args.iou_nms,
            "note": "photo: overlapping crops on full-res BGR, CLAHE→DoG per crop letterbox, boxes merged + NMS in full image",
        }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Wrote", args.out_json, file=sys.stderr)


if __name__ == "__main__":
    main()
