| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| v090-patched | pp512 | 377.50 | 1.55 | 3 | 48-52 C |
| v090-patched | tg128 | 29.59 | 0.04 | 3 | 48-52 C |
| v090-patched | pp2048 | 419.35 | 0.92 | 3 | 48-52 C |
| b12211-stock | pp512 | 230.15 | 0.18 | 3 | 40-52 C |
| b12211-stock | tg128 | 20.95 | 0.03 | 3 | 40-52 C |
| b12211-stock | pp2048 | 377.05 | 0.86 | 3 | 40-52 C |
| b12211-patched | pp512 | 378.16 | 2.89 | 3 | 46-52 C |
| b12211-patched | tg128 | 29.61 | 0.12 | 3 | 46-52 C |
| b12211-patched | pp2048 | 419.87 | 1.81 | 3 | 46-52 C |

delta of each other arm against b12211-stock:
   v090-patched on pp512: +64.03%
   v090-patched on tg128: +41.29%
   v090-patched on pp2048: +11.22%
   b12211-patched on pp512: +64.31%
   b12211-patched on tg128: +41.36%
   b12211-patched on pp2048: +11.36%
