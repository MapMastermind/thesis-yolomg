# diplom_ard100_thesis

Репозиторий артефактов по **ARD100** и **YOLOMG**: конфиги данных, архитектуры (**Concat3**, **Concat3Adaptive / V1**), скрипты подготовки и оценки, агрегированные результаты eval и веса обученных прогонов. Сырые данные в репозиторий не входят.

**Исходный датасет ARD100:** https://pan.baidu.com/s/1ycAoKbzQ1rlzvKr8VRakgw?pwd=1x2z 

После скачивания — подготовка каталогов `~/data/ard100/` по [`data/README_DATASET.md`](data/README_DATASET.md) и [`scripts/data_prep/prepare_ard100_dataset.py`](scripts/data_prep/prepare_ard100_dataset.py).


---

## Навигация: глава 2, глава 3, блоки I и II

Материалы разложены по **номерам глав** (описание данных / эксперименты) и **блокам экспериментального плана** (video-only vs mixed).

### Глава 2 — датасет ARD100 и метрики

Статистика сплитов, **e_bbox**, бины **Q0–Q4**, **hover-proxy (Q20)**.

| Содержание | Путь |
|------------|------|
| Аналитика датасета | [`experiments/analytics/ard100_chapter2_analytics.md`](experiments/analytics/ard100_chapter2_analytics.md) |
| Пороги e_bbox, калибровка hover-proxy | [`experiments/reference/`](experiments/reference/) |
| Скрипт аналитики | [`scripts/analytics/ard100_dataset_analytics.py`](scripts/analytics/ard100_dataset_analytics.py) |
| Yaml, гиперпараметры, сплиты | [`data/`](data/) — [`data/README_DATASET.md`](data/README_DATASET.md) |

### Глава 3 — протоколы и оценка моделей

**Eval A** (hit@0.5 / 0.75, photo / video), **Eval B** (бины e_bbox, hover-proxy, α fusion).

| Содержание | Путь |
|------------|------|
| Итоговые JSON (по блокам — ниже) | [`experiments/results/`](experiments/results/) |
| Протоколы train / eval | [`experiments/protocols/`](experiments/protocols/) |
| Условия Eval B | [`experiments/reference/PROTOCOL__evalB_ebbox_fd_bins.txt`](experiments/reference/PROTOCOL__evalB_ebbox_fd_bins.txt) |
| Скрипты eval | [`scripts/eval/`](scripts/eval/) |
| Имена и форматы файлов | [`experiments/README.md`](experiments/README.md) |

### Блок I — video-only

Train только на видео ARD100; baseline **Concat3** и **V1 Concat3Adaptive**; val / test на video-only split.

| Содержание | Путь |
|------------|------|
| Val: сводка, V1 vs baseline, McNemar | [`experiments/results/block_I__val__video_only/`](experiments/results/block_I__val__video_only/) — `summary__all_models.md` |
| Test: baseline | [`experiments/results/block_I__test/`](experiments/results/block_I__test/) |
| Exploratory (копии JSON) | [`experiments/results/block_I__exploratory/`](experiments/results/block_I__exploratory/) |
| Веса, конфиги RUN | [`runs/train/`](runs/train/) — блок I в [`runs/README.md`](runs/README.md) |

Основные RUN: `ard100_640_b64_e30_p7_video_native` (Concat3), `ard100_videoOnly_adaptiveFusion_b56_e30_pat7_from_videoNative_physgpu0` (**V1**).

### Блок II — mixed (фото + видео)

Mixed train и стратегии второго потока на фото. Здесь сравниваются **базовый Concat3** и **V1** — реализация **Concat3Adaptive** (адаптивное fusion по MotionTrust, см. [`models/NPS_uav_s_adaptive.yaml`](models/NPS_uav_s_adaptive.yaml)). Это та же архитектура V1, что в блоке I, но обучение на **mixed** (фото + видео), а не video-only.

| Содержание | Путь |
|------------|------|
| Протоколы | [`PROTOCOL_photo_strategy_mixed_video_native_fd.md`](experiments/protocols/PROTOCOL_photo_strategy_mixed_video_native_fd.md), [`PROTOCOL_concat3_adaptive_fusion_ablation.md`](experiments/protocols/PROTOCOL_concat3_adaptive_fusion_ablation.md) |
| Eval по RUN | [`runs/train/`](runs/train/) — блок II в [`runs/README.md`](runs/README.md) |
| Mixed-модели на video-only val | JSON в [`block_I__val__video_only/`](experiments/results/block_I__val__video_only/) (`mixed_zeroP`, `mixed_adaptive_fusion` — V1 после mixed train) |

**Concat3 vs V1 на mixed** — каталоги RUN:

| Модель | RUN (суффикс) |
|--------|----------------|
| Concat3 (baseline, фиксированное fusion) | `ard100_mixed640_concat3Control_*` |
| **V1** (Concat3Adaptive) | `ard100_mixed640_adaptiveFusion_*` |

Exploratory-вариант V1: `ard100_mixed640_adaptiveFusion_v2_q0os_*`. Протокол и метрики mixed val отличаются от блока I (video-only val/test).

---

## Структура репозитория

```
├── data/                   # глава 2–3: yaml, hyp
├── models/                 # глава 3: NPS_uav_s, Concat3Adaptive
├── scripts/
│   ├── analytics/          # глава 2
│   ├── data_prep/          # глава 3
│   └── eval/               # глава 3
├── experiments/
│   ├── analytics/          # глава 2
│   ├── reference/          # глава 2 (метрики), глава 3 (Eval B)
│   ├── protocols/          # глава 3
│   └── results/            # глава 3; подкаталоги — блок I
└── runs/train/             # 10 RUN: блок I и блок II
```

Дополнительно: [`test_code/`](test_code/) (FD, пути ARD100).

---

## Быстрый старт

```bash
cd ../Static_YOLOMG && source .venv/bin/activate

# подготовка данных (пути в data/*.yaml)
python ../diplom_ard100_thesis/scripts/data_prep/prepare_ard100_dataset.py

# обучение — из Static_YOLOMG
python train.py --data ../diplom_ard100_thesis/data/ard100_video.yaml ...

# оценка
python ../diplom_ard100_thesis/scripts/eval/eval_yolomg_val_photo_video_metrics.py ...
python ../diplom_ard100_thesis/scripts/eval/eval_ebbox_bins_fd_alpha.py ...
python ../diplom_ard100_thesis/scripts/eval/eval_hover_significance_mcnemar.py ...
```
