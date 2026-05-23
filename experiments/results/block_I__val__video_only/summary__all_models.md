# Блок I — ARD100 val (video-only)

## Протокол

| | |
|---|---|
| **Val** | `~/data/ard100` val + val2 — 12 978 кадров, 12 774 GT, 100% video |
| **Eval A** | `eval_yolomg_val_photo_video_metrics.py`, imgsz **960**, native FD |
| **Eval B (hover)** | `eval_ebbox_bins_fd_alpha.py`, imgsz **640**, gate_off, stream2=native |
| **Hover-proxy** | e_bbox ≤ **Q20** (гл. 2); бин **Q0** — нижний квинтиль e_bbox |

## Сводная таблица

| Модель | Train | video hit@0.5 | video hit@0.75 | Q0 hit@0.5 | hover Q20 | Q4 hit@0.5 |
|--------|-------|---------------|----------------|------------|-----------|------------|
| baseline_concat3 | video-only, 30 ep | **0.956** | 0.606 | 0.855 | 0.855 | 0.997 |
| mixed_zeroP | mixed | 0.927 | 0.457 | **0.866** | **0.865** | 0.995 |
| mixed_adaptive_fusion | mixed, 12 ep | 0.901 | 0.397 | 0.862 | 0.862 | 0.996 |
| v1_adaptive_fusion | video-only, 30 ep, best ep 22 | 0.933 | 0.498 | **0.899** | **0.899** | 0.997 |

### Δ к baseline_concat3 (val)

| Модель | video hit@0.5 | Q0 / hover Q20 |
|--------|---------------|----------------|
| mixed_zeroP | −2.9 п.п. | +1.0 п.п. |
| mixed_adaptive_fusion | −5.5 п.п. | +0.7 п.п. |
| **v1_adaptive_fusion** | −2.3 п.п. | **+4.4 п.п.** |

## V1 vs baseline

| | |
|---|---|
| **RUN** | `ard100_videoOnly_adaptiveFusion_b56_e30_pat7_from_videoNative_physgpu0` |
| **best.pt** | train-эпоха **22** |
| **hover Q20** | **0.899** (+4.4 п.п. vs baseline) |
| **video hit@0.5** | 0.933 (−2.3 п.п. vs baseline) |

### McNemar (hover-proxy, n=2557)

| | Значение |
|--|----------|
| Δ hit@0.5 (V1 − baseline) | **+4.38 п.п.** |
| p-value | **≈ 1.4×10⁻¹⁶** |
| Bootstrap 95% CI | **[+3.32, +5.44] п.п.** |

Детали: `stat_mcnemar_hover_proxy__baseline_vs_v1.json`.

## Артефакты

| Файл | Содержание |
|------|------------|
| `evalA_hit_by_iou__*.json` | Eval A |
| `evalB_bins_ebbox_fd_alpha__*.json` | Eval B |
| `summary__all_models.json` | Сводка (машиночитаемая) |
| `stat_mcnemar_hover_proxy__baseline_vs_v1.json` | McNemar |
