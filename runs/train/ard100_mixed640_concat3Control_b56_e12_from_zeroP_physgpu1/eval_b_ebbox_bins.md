# Eval B — бины e_bbox, hover-proxy (гл. 3, блок I)

- GT instances (video val): **3840**
- Frames: **3840**
- Weights baseline: ``
- FD-Gate stem τ=0.015, e_bbox τ (zero stream)=3.79, imgsz=640

## Hit@0.5 by e_bbox bin (same GT boxes)

| condition | Q0–Q20 e∈[-inf,3.79) | Q20–Q40 e∈[3.79,7.721) | Q40–Q60 e∈[7.721,13.48) | Q60–Q80 e∈[13.48,21.6) | Q80–Q100 e∈[21.6,inf) | hover Q20 |
|-----------|---|---|---|---|---|---|
| concat3Control_b56__gate_off__stream2_native | 0.859 | 0.969 | 0.986 | 0.993 | 0.995 | 0.859 |

## Δ hit@0.5 vs gate_off (same stream2)
