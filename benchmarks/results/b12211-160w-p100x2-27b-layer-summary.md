| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| v090-patched | pp512 | 207.31 | 0.94 | 3 | 46-49 C |
| v090-patched | tg128 | 18.34 | 0.04 | 3 | 46-49 C |
| v090-patched | pp2048 | 227.15 | 0.33 | 3 | 46-49 C |
| b12211-stock | pp512 | 144.30 | 0.35 | 3 | 40-49 C |
| b12211-stock | tg128 | 13.22 | 0.05 | 3 | 40-49 C |
| b12211-stock | pp2048 | 224.77 | 0.13 | 3 | 40-49 C |
| b12211-patched | pp512 | 207.41 | 0.76 | 3 | 44-49 C |
| b12211-patched | tg128 | 18.32 | 0.06 | 3 | 44-49 C |
| b12211-patched | pp2048 | 227.30 | 0.49 | 3 | 44-49 C |

delta of each other arm against b12211-stock:
   v090-patched on pp512: +43.66%
   v090-patched on tg128: +38.69%
   v090-patched on pp2048: +1.06%
   b12211-patched on pp512: +43.73%
   b12211-patched on tg128: +38.55%
   b12211-patched on pp2048: +1.12%
