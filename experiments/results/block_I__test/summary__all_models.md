# Блок I — ARD100 test (35 роликов)

## Протокол

| | |
|---|---|
| **Test** | `~/data/ard100` test — video-only |
| **Eval A** | imgsz **960**, native FD |
| **Eval B** | imgsz **640**, gate_off, stream2=native |

## Результаты (baseline_concat3)

| Метрика | Значение |
|---------|----------|
| video hit@0.5 | 0.956 |
| video hit@0.75 | 0.606 |

## Артефакты

| Файл | Содержание |
|------|------------|
| `evalA_hit_by_iou__baseline_concat3.json` | Eval A |
| `evalB_bins_ebbox_fd_alpha__baseline_concat3.json` | Eval B |
| `summary__all_models.json` | Сводка |
