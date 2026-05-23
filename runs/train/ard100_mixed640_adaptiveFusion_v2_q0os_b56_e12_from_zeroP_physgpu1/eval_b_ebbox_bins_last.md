# Eval B — бины e_bbox, hover-proxy (гл. 3, блок I)

- GT instances (video val): **3840**
- Frames: **3840**
- Weights baseline: ``
- FD-Gate stem τ=0.015, e_bbox τ (zero stream)=3.79, imgsz=640

## Hit@0.5 by e_bbox bin (same GT boxes)

| condition | Q0–Q20 e∈[-inf,3.79) | Q20–Q40 e∈[3.79,7.721) | Q40–Q60 e∈[7.721,13.48) | Q60–Q80 e∈[13.48,21.6) | Q80–Q100 e∈[21.6,inf) | hover Q20 |
|-----------|---|---|---|---|---|---|
| adaptiveFusion_v2_last__gate_off__stream2_native | 0.853 | 0.957 | 0.983 | 0.992 | 0.995 | 0.853 |
| fdgate_trained_tau0015__gate_off__stream2_native | 0.849 | 0.961 | 0.979 | 0.992 | 0.995 | 0.849 |

## Hit@0.5 — 2D: e_bbox × rel_area (D1)


### adaptiveFusion_v2_last__gate_off__stream2_native

| e_bbox \ rel_area | A0 rel∈[0e+00,1e-05) | A1 rel∈[1e-05,5e-05) | A2 rel∈[5e-05,2e-04) | A3 rel∈[2e-04,1e-03) | A4 rel∈[1e-03,1.000] |
|---|---|---|---|---|---|
| Q0–Q20 e∈[-inf,3.79) | — | 0.742 (n=295) | 0.912 (n=421) | 1.000 (n=52) | — |
| Q20–Q40 e∈[3.79,7.721) | — | 0.905 (n=201) | 0.971 (n=453) | 0.991 (n=112) | 1.000 (n=2) |
| Q40–Q60 e∈[7.721,13.48) | — | 0.952 (n=210) | 0.993 (n=411) | 1.000 (n=146) | 1.000 (n=1) |
| Q60–Q80 e∈[13.48,21.6) | — | 0.973 (n=150) | 0.998 (n=402) | 1.000 (n=214) | 0.500 (n=2) |
| Q80–Q100 e∈[21.6,inf) | — | 0.949 (n=59) | 1.000 (n=416) | 1.000 (n=285) | 0.875 (n=8) |

### fdgate_trained_tau0015__gate_off__stream2_native

| e_bbox \ rel_area | A0 rel∈[0e+00,1e-05) | A1 rel∈[1e-05,5e-05) | A2 rel∈[5e-05,2e-04) | A3 rel∈[2e-04,1e-03) | A4 rel∈[1e-03,1.000] |
|---|---|---|---|---|---|
| Q0–Q20 e∈[-inf,3.79) | — | 0.739 (n=295) | 0.907 (n=421) | 1.000 (n=52) | — |
| Q20–Q40 e∈[3.79,7.721) | — | 0.891 (n=201) | 0.982 (n=453) | 1.000 (n=112) | 1.000 (n=2) |
| Q40–Q60 e∈[7.721,13.48) | — | 0.938 (n=210) | 0.993 (n=411) | 1.000 (n=146) | 1.000 (n=1) |
| Q60–Q80 e∈[13.48,21.6) | — | 0.973 (n=150) | 0.998 (n=402) | 1.000 (n=214) | 0.500 (n=2) |
| Q80–Q100 e∈[21.6,inf) | — | 0.949 (n=59) | 1.000 (n=416) | 1.000 (n=285) | 0.875 (n=8) |

## MotionTrust α по бинам e_bbox (D3, mean по кадрам/GT)

| condition | Q0–Q20 e∈[-inf,3.79) | Q20–Q40 e∈[3.79,7.721) | Q40–Q60 e∈[7.721,13.48) | Q60–Q80 e∈[13.48,21.6) | Q80–Q100 e∈[21.6,inf) |
|---|---|---|---|---|---|
| adaptiveFusion_v2_last__gate_off__stream2_native | 0.885 | 0.885 | 0.885 | 0.885 | 0.885 |
| fdgate_trained_tau0015__gate_off__stream2_native | — | — | — | — | — |

## Δ hit@0.5 vs gate_off (same stream2)
