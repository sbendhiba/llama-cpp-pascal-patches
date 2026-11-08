| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| v090-patched | pp512 | 1579.12 | 1.10 | 3 | 41-42 C |
| v090-patched | tg128 | 84.38 | 0.02 | 3 | 41-42 C |
| b12211-stock | pp512 | 989.23 | 0.37 | 3 | 39-42 C |
| b12211-stock | tg128 | 58.86 | 0.74 | 3 | 39-42 C |
| b12211-patched | pp512 | 1565.70 | 1.14 | 3 | 41-42 C |
| b12211-patched | tg128 | 84.65 | 0.06 | 3 | 41-42 C |

delta of each other arm against b12211-stock:
   v090-patched on pp512: +59.63%
   v090-patched on tg128: +43.36%
   b12211-patched on pp512: +58.27%
   b12211-patched on tg128: +43.81%
