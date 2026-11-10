| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| b12211-stock | pp512 | 230.53 | 0.09 | 2 | 44-49 C |
| b12211-stock | tg128 | 21.03 | 0.02 | 2 | 44-49 C |
| b12211-patched | pp512 | 379.23 | 0.61 | 2 | 48-48 C |
| b12211-patched | tg128 | 29.71 | 0.04 | 2 | 48-48 C |

delta of each other arm against b12211-stock:
   b12211-patched on pp512: +64.50%
   b12211-patched on tg128: +41.30%
