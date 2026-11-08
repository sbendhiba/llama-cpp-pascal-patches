| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| v090-patched | pp512 | 1578.93 | 1.03 | 3 | 43-44 C |
| v090-patched | tg128 | 84.36 | 0.14 | 3 | 43-44 C |
| b12211-stock | pp512 | 987.00 | 0.97 | 3 | 41-43 C |
| b12211-stock | tg128 | 57.86 | 0.43 | 3 | 41-43 C |
| b12211-patched | pp512 | 1564.20 | 1.13 | 3 | 42-44 C |
| b12211-patched | tg128 | 84.58 | 0.11 | 3 | 42-44 C |
| v090-stock | pp512 | 766.34 | 1.32 | 3 | 42-44 C |
| v090-stock | tg128 | 59.93 | 1.16 | 3 | 42-44 C |

delta of each other arm against b12211-stock:
   v090-patched on pp512: +59.97%
   v090-patched on tg128: +45.80%
   b12211-patched on pp512: +58.48%
   b12211-patched on tg128: +46.19%
   v090-stock on pp512: -22.36%
   v090-stock on tg128: +3.58%
