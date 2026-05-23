# Mixed: стратегия 2-го потока на фото (видео = native FD) — воспроизведение

Корень: `../Static_YOLOMG/`. Реализация: `utils/datasets.py` при `YOLOMG_IMG2_EXP=mixed_policy`.

## Данные

| | Путь |
|---|------|
| yaml | `data/ard100_mixed.yaml` |
| корень | `~/data/ard100_mixed/` |
| маркер фото | `YOLOMG_PHOTO_PATH_MARKER=/mixed_photos/` (подстрока в пути RGB) |

Val: `val.txt` + `val2.txt` (12 978 пар; фото / видео по маркеру).

## Общий train (все руки S)

| Параметр | Значение |
|----------|----------|
| θ₀ | `runs/train/ard100_mixed640_zeroP_vidDrop03_from_vidbest_gpu0/weights/best.pt` |
| data / hyp | `ard100_mixed.yaml`, `hyp.mixed_finetune_640_minimal.yaml` |
| epochs / patience / batch / imgsz | 10 / 5 / 56 / 640 |

Видео (одинаково для всех S):

```bash
export YOLOMG_IMG2_EXP=mixed_policy
export YOLOMG_PHOTO_PATH_MARKER=/mixed_photos/
export YOLOMG_MIXED_VIDEO_STREAM2_ZERO_P=0.3
export YOLOMG_MIXED_VIDEO_LOW_FD_SUBSTITUTE=off
unset YOLOMG_MIXED_VIDEO_FD_ENERGY_TAU
```

## Руки S — только фото отличается

| S | Train env (дополнительно) | NAME (`runs/train/`) |
|---|---------------------------|----------------------|
| zero | `YOLOMG_MIXED_PHOTO_ZERO_STREAM2=1`, `unset YOLOMG_PHOTO_STREAM_MODE` | `ard100_mixed640_zeroP_vidNativeFD_e10p5_from_mixed640best_gpu1` |
| dog | `unset YOLOMG_MIXED_PHOTO_ZERO_STREAM2`, `YOLOMG_PHOTO_STREAM_MODE=dog` | `ard100_mixed640_dog_vidNativeFD_e10p5_from_mixed640best_gpu2` |
| clahe_dog | `YOLOMG_PHOTO_STREAM_MODE=clahe_dog` | `ard100_mixed640_clahe_dog_vidNativeFD_e10p5_from_mixed640best_gpu1` |
| motion_edge | `YOLOMG_PHOTO_STREAM_MODE=motion_edge` | `ard100_mixed640_motion_edge_vidNativeFD_e10p5_from_mixed640best_gpu1` |

Запуск одной руки:

```bash
cd ../Static_YOLOMG
export CUDA_VISIBLE_DEVICES=1 DEVICE=0 USE_NOHUP=1
export WEIGHTS="$PWD/runs/train/ard100_mixed640_zeroP_vidDrop03_from_vidbest_gpu0/weights/best.pt"
export EPOCHS=10 PATIENCE=5 BATCH=56
export NAME=ard100_mixed640_zeroP_vidNativeFD_e10p5_from_mixed640best_gpu1
PHOTO_STRATEGY=zero bash scripts/run_mixed_photo_strategy_arm.sh
```

(`PHOTO_STRATEGY` ∈ `zero`, `clahe_dog`, `dog`, `motion_edge`; подставить свой `NAME`.)

**Правило:** на eval для руки S использовать **тот же** режим фото, что при train.

## Eval (train-matched)

```bash
cd ../Static_YOLOMG
export YOLOMG_IMG2_EXP=mixed_policy YOLOMG_PHOTO_PATH_MARKER=/mixed_photos/
export YOLOMG_MIXED_VIDEO_STREAM2_ZERO_P=0.3
export YOLOMG_MIXED_VIDEO_LOW_FD_SUBSTITUTE=off
unset YOLOMG_MIXED_VIDEO_FD_ENERGY_TAU YOLOMG_MIXED_PHOTO_ZERO_STREAM2 YOLOMG_PHOTO_STREAM_MODE
```

| S | Аргументы eval |
|---|----------------|
| zero | `--photo-zero-stream2` |
| clahe_dog | `--photo-stream-mode clahe_dog` |
| dog | `--photo-stream-mode dog` |
| motion_edge | `--photo-stream-mode motion_edge` |

```bash
python scripts/eval_yolomg_val_photo_video_metrics.py \
  --weights runs/train/<NAME>/weights/best.pt \
  --imgsz 960 --device 0 \
  --conf-nms 0.001 --iou-nms 0.4 --pred-rate-conf 0.25 \
  <флаги S из таблицы> \
  --out-json runs/train/<NAME>/eval_b2_trainmatch_<S>.json
```

Эталонный eval родителя (θ₀ без finetune этой сетки):

```bash
# те же export, что выше
python scripts/eval_yolomg_val_photo_video_metrics.py \
  --weights runs/train/ard100_mixed640_zeroP_vidDrop03_from_vidbest_gpu0/weights/best.pt \
  --imgsz 960 --device 0 --conf-nms 0.001 --iou-nms 0.4 --pred-rate-conf 0.25 \
  --photo-zero-stream2 \
  --out-json runs/train/ard100_mixed640_zeroP_vidDrop03_from_vidbest_gpu0/eval_b2_trainmatch_zero_refvideo.json
```

Пайплайн mixed: `mixed_pipeline_protocol.txt`.
