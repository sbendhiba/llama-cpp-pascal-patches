| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| v090-patched | pp512 | 206.66 | 0.26 | 3 | 48-50 C |
| v090-patched | tg128 | 18.26 | 0.09 | 3 | 48-50 C |
| v090-patched | pp2048 | 226.92 | 0.17 | 3 | 48-50 C |
| b12211-stock | pp512 | 144.07 | 0.71 | 3 | 40-49 C |
| b12211-stock | tg128 | 13.21 | 0.04 | 3 | 40-49 C |
| b12211-stock | pp2048 | 224.70 | 0.46 | 3 | 40-49 C |
| b12211-patched | pp512 | 206.72 | 0.45 | 3 | 45-49 C |
| b12211-patched | tg128 | 18.32 | 0.08 | 3 | 45-49 C |
| b12211-patched | pp2048 | 227.16 | 0.57 | 3 | 45-49 C |
| v090-stock | pp512 | 143.37 | 0.42 | 3 | 47-49 C |
| v090-stock | tg128 | 13.20 | 0.03 | 3 | 47-49 C |
| v090-stock | pp2048 | 224.78 | 0.43 | 3 | 47-49 C |

delta of each other arm against b12211-stock:
   v090-patched on pp512: +43.45%
   v090-patched on tg128: +38.26%
   v090-patched on pp2048: +0.99%
   b12211-patched on pp512: +43.49%
   b12211-patched on tg128: +38.70%
   b12211-patched on pp2048: +1.09%
   v090-stock on pp512: -0.49%
   v090-stock on tg128: -0.08%
   v090-stock on pp2048: +0.03%
