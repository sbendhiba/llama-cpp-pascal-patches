| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| b12211-stock | pp512 | 146.98 | 0.27 | 2 | 42-46 C |
| b12211-stock | tg128 | 15.45 | 0.01 | 2 | 42-46 C |
| b12211-patched | pp512 | 212.30 | 0.20 | 2 | 45-46 C |
| b12211-patched | tg128 | 21.24 | 0.01 | 2 | 45-46 C |

delta of each other arm against b12211-stock:
   b12211-patched on pp512: +44.44%
   b12211-patched on tg128: +37.45%
