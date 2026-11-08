| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| v090-patched | pp512 | 377.13 | 1.77 | 3 | 48-53 C |
| v090-patched | tg128 | 29.59 | 0.11 | 3 | 48-53 C |
| v090-patched | pp2048 | 419.33 | 1.06 | 3 | 48-53 C |
| b12211-stock | pp512 | 230.38 | 0.16 | 3 | 37-52 C |
| b12211-stock | tg128 | 20.88 | 0.08 | 3 | 37-52 C |
| b12211-stock | pp2048 | 376.94 | 2.46 | 3 | 37-52 C |
| b12211-patched | pp512 | 378.21 | 2.78 | 3 | 43-52 C |
| b12211-patched | tg128 | 29.58 | 0.16 | 3 | 43-52 C |
| b12211-patched | pp2048 | 419.72 | 2.12 | 3 | 43-52 C |
| v090-stock | pp512 | 230.45 | 0.25 | 3 | 46-52 C |
| v090-stock | tg128 | 20.87 | 0.05 | 3 | 46-52 C |
| v090-stock | pp2048 | 376.21 | 0.59 | 3 | 46-52 C |

delta of each other arm against b12211-stock:
   v090-patched on pp512: +63.70%
   v090-patched on tg128: +41.71%
   v090-patched on pp2048: +11.25%
   b12211-patched on pp512: +64.16%
   b12211-patched on tg128: +41.64%
   b12211-patched on pp2048: +11.35%
   v090-stock on pp512: +0.03%
   v090-stock on tg128: -0.06%
   v090-stock on pp2048: -0.19%
