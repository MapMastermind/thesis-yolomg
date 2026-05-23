#!/usr/bin/env python3
"""
Случайные одиночные кадры из ARD100 как «фото» для mixed_policy.

Кадры берутся из тех же файлов, что и базовый сплит ARD100 (train.txt / val.txt),
чтобы не смешивать train/val. В каталоге .../images/mixed_photos/ путь содержит
маркер /mixed_photos/ — датасет подставит pseudo второго потока из RGB.

Режимы:
  1) Дописать фиксированное число строк (--n-train / --n-val).
  2) Задать долю фото за эпоху: --balance-photo-fraction 0.5
     По умолчанию без пересечения: каждый кадр либо только в видео-части, либо только как
     «фото» (симлинк mixed_photos). Общее число строк = исходное число видео-строк;
     N_photo = round(N_total * F), N_video = N_total - N_photo.
     --allow-photo-video-overlap: старое поведение — все видео + доп. N фото (N = N_video*F/(1-F)),
     одни и те же RGB-файлы могут встречаться и как видео, и как фото.

Запуск из корня Static_YOLOMG:
  ./.venv/bin/python scripts/build_mixed_photo_subset_from_ard100.py --balance-photo-fraction 0.5

По умолчанию симлинки. Ключ --copy — копирование файлов.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


def _read_list(p: Path) -> list[str]:
    if not p.is_file():
        raise FileNotFoundError(p)
    lines = []
    with open(p) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(os.path.normpath(s))
    return lines


def _stem_from_rgb_line(line: str) -> str:
    return Path(line).stem


def _rgb_to_img2_path(rgb_line: str) -> str:
    if os.sep + "images" + os.sep not in rgb_line.replace("/", os.sep):
        raise ValueError(f"Ожидался путь с /images/: {rgb_line}")
    return rgb_line.replace(os.sep + "images" + os.sep, os.sep + "images2" + os.sep, 1)


def _rgb_to_label_path(rgb_line: str) -> str:
    if os.sep + "images" + os.sep not in rgb_line.replace("/", os.sep):
        raise ValueError(f"Ожидался путь с /images/: {rgb_line}")
    base = rgb_line.rsplit(".", 1)[0]
    return base.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep, 1) + ".txt"


def _dest_rgb(ard_root: Path, stem: str) -> Path:
    return ard_root / "images" / "mixed_photos" / f"{stem}.jpg"


def _dest_img2(ard_root: Path, stem: str) -> Path:
    return ard_root / "images2" / "mixed_photos" / f"{stem}.jpg"


def _dest_lbl(ard_root: Path, stem: str) -> Path:
    return ard_root / "labels" / "mixed_photos" / f"{stem}.txt"


def _is_photo_line(line: str) -> bool:
    return "/mixed_photos/" in line.replace(os.sep, "/")


def _link_or_copy(src: Path, dst: Path, *, use_copy: bool, dry_run: bool, force: bool) -> None:
    if dst.is_symlink() or dst.is_file():
        if not force:
            return
        if not dry_run:
            dst.unlink()
    if dry_run:
        print(f"  would create: {dst} <- {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_res = src.resolve()
    if use_copy:
        import shutil

        shutil.copy2(src_res, dst)
    else:
        dst.symlink_to(src_res)


def _strip_video_pairs(
    paths1: list[str], paths2: list[str]
) -> tuple[list[str], list[str]]:
    if len(paths1) != len(paths2):
        raise ValueError(
            f"train и train2 разной длины: {len(paths1)} vs {len(paths2)} — исправьте списки вручную."
        )
    v1, v2 = [], []
    for a, b in zip(paths1, paths2):
        if _is_photo_line(a):
            continue
        v1.append(a)
        v2.append(b)
    return v1, v2


def _photo_count_from_video(n_video: int, fraction: float) -> int:
    """N_photo / (N_photo + n_video) = fraction  →  N_photo = n_video * fraction / (1 - fraction)."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("balance-photo-fraction должен быть строго между 0 и 1")
    return max(0, int(round(n_video * fraction / (1.0 - fraction))))


def main() -> int:
    ap = argparse.ArgumentParser(description="ARD100 → mixed_photos + списки ard100_mixed")
    ap.add_argument("--ard-root", type=Path, default=Path("/home/tanyadiplom/data/ard100"))
    ap.add_argument("--mixed-lists", type=Path, default=Path("/home/tanyadiplom/data/ard100_mixed"))
    ap.add_argument(
        "--balance-photo-fraction",
        type=float,
        default=None,
        metavar="F",
        help="доля фото в train/val (например 0.5); перезаписывает mixed *.txt",
    )
    ap.add_argument(
        "--allow-photo-video-overlap",
        action="store_true",
        help="с --balance: не делить кадры; все видео + отдельные строки фото (дубли RGB возможны)",
    )
    ap.add_argument("--n-train", type=int, default=5000, help="с --balance не используется")
    ap.add_argument("--n-val", type=int, default=800, help="с --balance не используется")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="копировать файлы вместо симлинков")
    ap.add_argument("--no-append", action="store_true", help="только режим --balance: не писать списки")
    ap.add_argument(
        "--force-resymlink",
        action="store_true",
        help="пересоздавать симлинки bal_* даже если уже есть (для --balance)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ard: Path = args.ard_root
    mixed_dir: Path = args.mixed_lists
    rnd = random.Random(args.seed)

    train_lines = _read_list(ard / "train.txt")
    val_lines = _read_list(ard / "val.txt")

    def filter_video_rgb(lines: list[str]) -> list[str]:
        out = []
        for x in lines:
            xn = x.replace("/", os.sep)
            if "/mixed_photos/" in x.replace(os.sep, "/"):
                continue
            if os.sep + "images" + os.sep not in xn:
                continue
            out.append(x)
        return out

    train_pool = filter_video_rgb(train_lines)
    val_pool = filter_video_rgb(val_lines)
    if not train_pool or not val_pool:
        print("Пустой train или val пул после фильтрации.", file=sys.stderr)
        return 1

    use_copy: bool = args.copy
    dry: bool = args.dry_run
    force = bool(args.force_resymlink)

    def process_one(rgb_line: str, dest_stem: str) -> tuple[str, str] | None:
        src_rgb = Path(rgb_line)
        src_i2 = Path(_rgb_to_img2_path(rgb_line))
        src_lb = Path(_rgb_to_label_path(rgb_line))
        if not src_rgb.is_file() or not src_i2.is_file() or not src_lb.is_file():
            return None
        dr = _dest_rgb(ard, dest_stem)
        d2 = _dest_img2(ard, dest_stem)
        dl = _dest_lbl(ard, dest_stem)
        _link_or_copy(src_rgb, dr, use_copy=use_copy, dry_run=dry, force=force)
        _link_or_copy(src_i2, d2, use_copy=use_copy, dry_run=dry, force=force)
        _link_or_copy(src_lb, dl, use_copy=use_copy, dry_run=dry, force=force)
        return str(dr), str(d2)

    if args.balance_photo_fraction is not None:
        frac = float(args.balance_photo_fraction)
        overlap = bool(args.allow_photo_video_overlap)
        # При пересборке disjoint перезаписываем симлинки bal_* под новые цели
        nonlocal_force = force or (not overlap)

        def process_one_bal(rgb_line: str, dest_stem: str) -> tuple[str, str] | None:
            src_rgb = Path(rgb_line)
            src_i2 = Path(_rgb_to_img2_path(rgb_line))
            src_lb = Path(_rgb_to_label_path(rgb_line))
            if not src_rgb.is_file() or not src_i2.is_file() or not src_lb.is_file():
                return None
            dr = _dest_rgb(ard, dest_stem)
            d2 = _dest_img2(ard, dest_stem)
            dl = _dest_lbl(ard, dest_stem)
            _link_or_copy(src_rgb, dr, use_copy=use_copy, dry_run=dry, force=nonlocal_force)
            _link_or_copy(src_i2, d2, use_copy=use_copy, dry_run=dry, force=nonlocal_force)
            _link_or_copy(src_lb, dl, use_copy=use_copy, dry_run=dry, force=nonlocal_force)
            return str(dr), str(d2)

        m_train1 = _read_list(mixed_dir / "train.txt")
        m_train2 = _read_list(mixed_dir / "train2.txt")
        m_val1 = _read_list(mixed_dir / "val.txt")
        m_val2 = _read_list(mixed_dir / "val2.txt")

        v_tr1, v_tr2 = _strip_video_pairs(m_train1, m_train2)
        v_va1, v_va2 = _strip_video_pairs(m_val1, m_val2)

        train_pairs: list[tuple[str, str]] = []
        val_pairs: list[tuple[str, str]] = []
        skipped = 0
        v_tr1_out, v_tr2_out = v_tr1, v_tr2
        v_va1_out, v_va2_out = v_va1, v_va2

        if overlap:
            n_tr_photo = _photo_count_from_video(len(v_tr1), frac)
            n_va_photo = _photo_count_from_video(len(v_va1), frac)
            pick_tr = rnd.choices(train_pool, k=n_tr_photo)
            pick_va = rnd.choices(val_pool, k=n_va_photo)
            for i, line in enumerate(pick_tr):
                stem = f"bal_{i:07d}"
                r = process_one_bal(line, stem)
                if r is None:
                    skipped += 1
                    continue
                train_pairs.append(r)
            for i, line in enumerate(pick_va):
                stem = f"balv_{i:07d}"
                r = process_one_bal(line, stem)
                if r is None:
                    skipped += 1
                    continue
                val_pairs.append(r)
            mode_msg = "overlap (все видео + доп. фото, RGB могут дублироваться)"
        else:
            # Разбиение без пересечения: каждый исходный кадр — только видео ИЛИ только фото.
            def disjoint_build(
                v1: list[str], v2: list[str], prefix: str
            ) -> tuple[list[str], list[str], list[tuple[str, str]], int]:
                n_total = len(v1)
                if n_total == 0:
                    return [], [], [], 0
                n_photo = int(round(n_total * frac))
                n_photo = max(0, min(n_photo, n_total))
                paired = list(zip(v1, v2))
                rnd.shuffle(paired)
                photo_zip = paired[:n_photo]
                video_zip = paired[n_photo:]
                sk = 0
                out_pairs: list[tuple[str, str]] = []
                for i, (rgb_line, _) in enumerate(photo_zip):
                    stem = f"{prefix}_{i:07d}"
                    r = process_one_bal(rgb_line, stem)
                    if r is None:
                        sk += 1
                        continue
                    out_pairs.append(r)
                v1k = [a for a, _ in video_zip]
                v2k = [b for _, b in video_zip]
                return v1k, v2k, out_pairs, sk

            v_tr1_out, v_tr2_out, train_pairs, skt = disjoint_build(v_tr1, v_tr2, "bal")
            skipped += skt
            v_va1_out, v_va2_out, val_pairs, skv = disjoint_build(v_va1, v_va2, "balv")
            skipped += skv
            mode_msg = "disjoint (кадр не встречается и как видео, и как фото)"

        print(
            f"Баланс fraction={frac:.4f}, {mode_msg}: "
            f"train видео {len(v_tr1_out)} + фото {len(train_pairs)} = {len(v_tr1_out) + len(train_pairs)}; "
            f"val видео {len(v_va1_out)} + фото {len(val_pairs)} = {len(v_va1_out) + len(val_pairs)}. "
            f"Пропусков (нет файлов): {skipped}. {'копии' if use_copy else 'симлинки'}"
        )

        if args.no_append:
            if dry:
                print("(dry-run: списки не перезаписаны)")
            return 0

        def write_mixed(p1: Path, p2: Path, vid1: list[str], vid2: list[str], pairs: list[tuple[str, str]]) -> None:
            if dry:
                print(f"would rewrite {p1} / {p2}: {len(vid1)} video + {len(pairs)} photo")
                return
            with open(p1, "w") as f1, open(p2, "w") as f2:
                for a, b in zip(vid1, vid2):
                    f1.write(a + "\n")
                    f2.write(b + "\n")
                for a, b in pairs:
                    f1.write(a + "\n")
                    f2.write(b + "\n")

        write_mixed(mixed_dir / "train.txt", mixed_dir / "train2.txt", v_tr1_out, v_tr2_out, train_pairs)
        write_mixed(mixed_dir / "val.txt", mixed_dir / "val2.txt", v_va1_out, v_va2_out, val_pairs)
        if not dry:
            print(f"Записаны {mixed_dir / 'train.txt'} и val (video + photo, выровнено по строкам).")

        if dry:
            print("(dry-run)")
        return 0

    # --- режим дописывания фиксированного числа ---
    n_tr = min(args.n_train, len(train_pool))
    n_va = min(args.n_val, len(val_pool))
    pick_tr = rnd.sample(train_pool, n_tr)
    pick_va = rnd.sample(val_pool, n_va)

    def process_one_simple(rgb_line: str) -> tuple[str, str] | None:
        stem = _stem_from_rgb_line(rgb_line)
        return process_one(rgb_line, stem)

    train_pairs: list[tuple[str, str]] = []
    val_pairs: list[tuple[str, str]] = []
    skipped = 0
    for line in pick_tr:
        r = process_one_simple(line)
        if r is None:
            skipped += 1
            continue
        train_pairs.append(r)
    for line in pick_va:
        r = process_one_simple(line)
        if r is None:
            skipped += 1
            continue
        val_pairs.append(r)

    print(
        f"Готово кадров: train {len(train_pairs)} / val {len(val_pairs)} "
        f"(запрошено {n_tr}/{n_va}, пропусков без файлов: {skipped}), "
        f"{'копии' if use_copy else 'симлинки'}"
    )

    if not args.no_append:
        for name, pairs in (("train.txt", train_pairs), ("val.txt", val_pairs)):
            if not pairs:
                continue
            p1 = mixed_dir / name
            p2 = mixed_dir / name.replace(".txt", "2.txt")
            if dry:
                print(f"would append {len(pairs)} lines to {p1} and {p2}")
                continue
            with open(p1, "a") as f1, open(p2, "a") as f2:
                for a, b in pairs:
                    f1.write(a + "\n")
                    f2.write(b + "\n")
            print(f"Дописано в {p1} и {p2}: {len(pairs)} строк")

    if dry:
        print("(dry-run: файлы и append не выполнялись полностью — см. would create / would append)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
