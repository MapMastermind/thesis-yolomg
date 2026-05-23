# experiments/

Итоговые JSON оценки, инженерные протоколы прогонов, справочники метрик и аналитика датасета.

```
experiments/
├── protocols/          # воспроизведение train / eval
├── reference/          # пороги e_bbox, hover-proxy, протокол Eval B
├── results/            # агрегированные результаты по прогонам
└── analytics/          # ard100_chapter2_analytics.{json,md}
```

Логи и промежуточные eval в каталогах RUN: [`../runs/train/`](../runs/train/).

## Имена файлов результатов

Шаблон: `{тип}__{суть}__{модель}.json`

| Префикс | Содержание |
|---------|------------|
| `evalA_hit_by_iou` | Eval A — hit@0.5 / 0.75, photo и video |
| `evalB_bins_ebbox_fd_alpha` | Eval B — бины e_bbox, FD в GT-bbox, α fusion |
| `stat_mcnemar_hover_proxy` | McNemar для hover-proxy |
| `summary__all_models` | Сводка по конфигурациям (`.md` — только основной текст сводки) |

Суффиксы модели: `baseline_concat3`, `v1_adaptive_fusion`, `mixed_zeroP`, `mixed_adaptive_fusion`.

## `results/`

| Каталог | Split / назначение |
|---------|-------------------|
| `block_I__val__video_only/` | Блок I, val, video-only |
| `block_I__test/` | Блок I, test, 35 роликов |
| `block_I__exploratory/` | Блок I, exploratory (копии JSON; не fair-сравнение с блоком II) |

В каждой подпапке один markdown: `summary__all_models.md`.

## `reference/`

| Файл | Назначение |
|------|------------|
| `reference__ebbox_bins__multi_model.{json,md}` | Сводка Eval B по бинам e_bbox |
| `reference__hover_proxy__q20_calib.{json,md}` | Калибровка hover-proxy (Q20, τ) |
| `PROTOCOL__evalB_ebbox_fd_bins.txt` | Условия Eval B |

## `protocols/`

| Файл | Прогон |
|------|--------|
| `PROTOCOL_concat3_adaptive_fusion_ablation.md` | Concat3 vs Concat3Adaptive (mixed) |
| `PROTOCOL_photo_strategy_mixed_video_native_fd.md` | Стратегии 2-го потока на фото |
| `mixed_pipeline_protocol.txt` | Пайплайн mixed_policy |

## `analytics/`

`ard100_chapter2_analytics.{json,md}` — статистика ARD100 (сплиты, rel_area, e_bbox, сцены).

Скрипты: [`../scripts/`](../scripts/).
