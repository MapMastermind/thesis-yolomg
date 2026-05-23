# Датасет ARD100 (вне репозитория)

Подготовленный корпус **не хранится в этой папке** (~163 ГБ). Здесь только **yaml** и **hyp** для `train.py`.

**Исходная выгрузка:** https://pan.baidu.com/s/1ycAoKbzQ1rlzvKr8VRakgw?pwd=1x2z (пароль: `1x2z`)

## Пути на машине экспериментов

| Путь | Содержание |
|------|------------|
| `/data/ARD100/` | Исходные MP4 + разметка Pascal VOC |
| `~/data/ard100/` | `images/`, `images2/` (FD5), `labels/`, `train.txt`, `val.txt`, `test.txt` |
| `~/data/ard100_mixed/` | `mixed_photos/`, списки train/val для mixed-экспериментов |

Пути в yaml задаются через `test_code/paths_ard100.py` и переменные окружения (см. комментарии в `ard100.yaml`).

## Файлы в `data/`

| Файл | Назначение |
|------|------------|
| `ard100.yaml` | Полный ARD100 (train+val+test) |
| `ard100_video.yaml` | Video-only, native FD5 — **блок I** |
| `ard100_mixed.yaml` | Mixed photo + video — **блок II** |
| `hyps/hyp.scratch-low.yaml` | Scratch video-only |
| `hyps/hyp.mixed_finetune_640_minimal.yaml` | Fine-tune mixed |

## Подготовка

```bash
cd ../Static_YOLOMG
python ../diplom_ard100_thesis/scripts/data_prep/prepare_ard100_dataset.py
python ../diplom_ard100_thesis/scripts/data_prep/build_mixed_photo_subset_from_ard100.py   # для mixed
```

Зависимости offline motion: `test_code/FD5_mask.py`, `test_code/MOD_Functions.py`.

Статистика датасета (гл. 2): `experiments/analytics/ard100_chapter2_analytics.md`.
