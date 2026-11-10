| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| b12211-stock | pp512 | 234.30 | 0.78 | 2 | 43-48 C |
| b12211-stock | tg128 | 23.80 | 0.06 | 2 | 43-48 C |
| b12211-patched | pp512 | 390.77 | 0.87 | 2 | 47-48 C |
| b12211-patched | tg128 | 34.13 | 0.00 | 2 | 47-48 C |

delta of each other arm against b12211-stock:
   b12211-patched on pp512: +66.78%
   b12211-patched on tg128: +43.42%
