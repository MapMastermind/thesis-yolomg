# Concat3 (CONTROL) vs Concat3Adaptive (EXP) — воспроизведение

Корень обучения: `../Static_YOLOMG/`. Артефакты: `runs/train/` (копии в этом репо).

## Отличие прогонов

| | CONTROL | EXP |
|---|---------|-----|
| cfg | `models/NPS_uav_s.yaml` | `models/NPS_uav_s_adaptive.yaml` |
| fusion | `Concat3` | `Concat3Adaptive` + `motion_trust.*` |
| load от zeroP | полное совпадение | 510/514 ключей + 4 новых тензора |

Остальное (data, hyp, init, env) — **одинаково**.

## Train

| Параметр | Значение |
|----------|----------|
| Init | `runs/train/ard100_mixed640_zeroP_vidDrop03_from_vidbest_gpu0/weights/best.pt` |
| data | `data/ard100_mixed.yaml` |
| hyp | `data/hyps/hyp.mixed_finetune_640_minimal.yaml` |
| epochs / patience / batch / imgsz | 12 / 5 / 56 / 640 |
| Скрипты | `scripts/_mixed640_finetune_ablation_common.sh`, `train_ard100_640_single_gpu_ablation.sh` |

```bash
export YOLOMG_IMG2_EXP=mixed_policy
export YOLOMG_PHOTO_PATH_MARKER=/mixed_photos/
export YOLOMG_MIXED_PHOTO_ZERO_STREAM2=1
export YOLOMG_MIXED_VIDEO_STREAM2_ZERO_P=0.3
export YOLOMG_MIXED_VIDEO_LOW_FD_SUBSTITUTE=off
export YOLOMG_CONCAT3_FD_GATE=off
unset YOLOMG_MIXED_VIDEO_FD_ENERGY_TAU
```

| RUN_ID | Скрипт запуска |
|--------|----------------|
| `ard100_mixed640_concat3Control_b56_e12_from_zeroP_physgpu1` | `scripts/run_mixed640_concat3_control_e12_from_zeroP.sh` |
| `ard100_mixed640_adaptiveFusion_b56_e12_from_zeroP_physgpu1` | `scripts/run_mixed640_adaptiveFusion_e12_from_zeroP.sh` |
| `ard100_mixed640_adaptiveFusion_v2_q0os_b56_e12_from_zeroP_physgpu1` | (v2, отдельный скрипт очереди в Static_YOLOMG) |

Пример:

```bash
cd ../Static_YOLOMG
CUDA_VISIBLE_DEVICES=1 DEVICE=0 USE_NOHUP=1 \
  NAME=ard100_mixed640_concat3Control_b56_e12_from_zeroP_physgpu1 \
  bash scripts/run_mixed640_concat3_control_e12_from_zeroP.sh
```

Логи: `runs/train/<NAME>/console.log`, `opt.yaml`, `hyp.yaml`, `weights/best.pt`.

## Eval A (photo + video, mixed val)

```bash
cd ../Static_YOLOMG
export YOLOMG_IMG2_EXP=mixed_policy YOLOMG_PHOTO_PATH_MARKER=/mixed_photos/
export YOLOMG_MIXED_PHOTO_ZERO_STREAM2=1 YOLOMG_MIXED_VIDEO_STREAM2_ZERO_P=0.3
export YOLOMG_MIXED_VIDEO_LOW_FD_SUBSTITUTE=off YOLOMG_CONCAT3_FD_GATE=off
unset YOLOMG_MIXED_VIDEO_FD_ENERGY_TAU

python scripts/eval_yolomg_val_photo_video_metrics.py \
  --weights runs/train/<NAME>/weights/best.pt \
  --imgsz 960 --device 0 \
  --conf-nms 0.001 --iou-nms 0.4 --pred-rate-conf 0.25 \
  --photo-zero-stream2 \
  --out-json runs/train/<NAME>/eval_a_production.json
```

## Eval B (бины e_bbox, α)

```bash
python scripts/eval_ebbox_bins_fd_alpha.py \
  --device 0 --skip-fdgate-weights \
  --weights-baseline runs/train/<NAME>/weights/best.pt \
  --weights-label "<concat3Control|adaptiveFusion>" \
  --gate-modes gate_off --stream2-policies native \
  --out-json runs/train/<NAME>/eval_b_ebbox_bins.json \
  --out-md runs/train/<NAME>/eval_b_ebbox_bins.md
```

Протокол бинов: `experiments/reference/PROTOCOL__evalB_ebbox_fd_bins.txt`.  
Пайплайн mixed: `mixed_pipeline_protocol.txt`.
