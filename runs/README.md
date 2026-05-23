# runs/

Артефакты **обучения** для 10 прогонов (блок I и блок II). Итоговые сводки eval — в [`../experiments/results/`](../experiments/results/).

## `train/<RUN_ID>/`

| Содержимое | Файлы |
|------------|--------|
| Конфиг | `opt.yaml`, `hyp.yaml` |
| Метрики | `results.csv`, `results.png` |
| Веса | `weights/best.pt`, `weights/last.pt` |
| Eval в каталоге RUN | `eval_a_production_*.json`, `eval_b_*` (Eval B), `eval_b2_trainmatch_*.json` |

## Прогоны

| RUN_ID | Блок | Роль |
|--------|------|------|
| `ard100_640_b64_e30_p7_video_native` | I | Baseline Concat3, video-only |
| `ard100_videoOnly_adaptiveFusion_b56_e30_pat7_from_videoNative_physgpu0` | I | V1 Concat3Adaptive |
| `ard100_mixed640_zeroP_vidDrop03_from_vidbest_gpu0` | II | mixed zeroP (θ₀ для finetune) |
| `ard100_mixed640_zeroP_vidNativeFD_e10p5_from_mixed640best_gpu1` | II | photo zero + native FD |
| `ard100_mixed640_dog_vidNativeFD_e10p5_from_mixed640best_gpu2` | II | photo DoG |
| `ard100_mixed640_clahe_dog_vidNativeFD_e10p5_from_mixed640best_gpu1` | II | photo CLAHE+DoG |
| `ard100_mixed640_motion_edge_vidNativeFD_e10p5_from_mixed640best_gpu1` | II | photo motion_edge |
| `ard100_mixed640_concat3Control_b56_e12_from_zeroP_physgpu1` | II | fusion A/B CONTROL |
| `ard100_mixed640_adaptiveFusion_b56_e12_from_zeroP_physgpu1` | II | fusion A/B EXP |
| `ard100_mixed640_adaptiveFusion_v2_q0os_b56_e12_from_zeroP_physgpu1` | II | fusion v2 (exploratory) |

Протоколы: [`../experiments/protocols/`](../experiments/protocols/).
