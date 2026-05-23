# scripts/

Запуск из **`../Static_YOLOMG/`** (`.venv`).

## `data_prep/`

| Скрипт | Назначение |
|--------|------------|
| `prepare_ard100_dataset.py` | MP4+VOC → `~/data/ard100` |
| `build_mixed_photo_subset_from_ard100.py` | → `~/data/ard100_mixed` |
| `build_train_video_strata_cache.py` | Кэш стратификации video |
| `calibrate_video_fd_energy_tau.py` | Калибровка τ (FD energy) |

## `analytics/`

| Скрипт | Назначение |
|--------|------------|
| `ard100_dataset_analytics.py` | → `experiments/analytics/` |
| `ard100_chapter2_figures.py` | → `experiments/figures/chapter2/` |

## `eval/`

| Скрипт | Выход (в Static_YOLOMG) | В этом репо после rename |
|--------|-------------------------|---------------------------|
| `eval_yolomg_val_photo_video_metrics.py` | `eval_a_*.json` | `evalA_hit_by_iou__*.json` |
| `eval_ebbox_bins_fd_alpha.py` | Eval B (гл. 3) | `evalB_bins_ebbox_fd_alpha__*.json` |
| `eval_hover_significance_mcnemar.py` | `hover_significance_mcnemar.*` | `stat_mcnemar_hover_proxy__*` |
