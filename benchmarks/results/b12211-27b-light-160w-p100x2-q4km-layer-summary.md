| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| b12211-stock | pp512 | 144.18 | 0.03 | 2 | 43-47 C |
| b12211-stock | tg128 | 13.23 | 0.01 | 2 | 43-47 C |
| b12211-patched | pp512 | 207.97 | 0.40 | 2 | 46-46 C |
| b12211-patched | tg128 | 18.38 | 0.01 | 2 | 46-46 C |

delta of each other arm against b12211-stock:
   b12211-patched on pp512: +44.24%
   b12211-patched on tg128: +38.89%
