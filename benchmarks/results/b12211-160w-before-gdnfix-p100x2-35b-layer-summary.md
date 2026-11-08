| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| v090-patched | pp512 | 1065.91 | 6.71 | 3 | 42-43 C |
| v090-patched | tg128 | 118.97 | 0.39 | 3 | 42-43 C |
| b12211-stock | pp512 | 658.05 | 0.76 | 3 | 40-43 C |
| b12211-stock | tg128 | 58.19 | 0.03 | 3 | 40-43 C |
| b12211-patched | pp512 | 1065.40 | 2.75 | 3 | 41-44 C |
| b12211-patched | tg128 | 95.59 | 0.03 | 3 | 41-44 C |
| v090-stock | pp512 | 488.72 | 0.38 | 3 | 41-43 C |
| v090-stock | tg128 | 58.12 | 0.01 | 3 | 41-43 C |

delta of each other arm against b12211-stock:
   v090-patched on pp512: +61.98%
   v090-patched on tg128: +104.45%
   b12211-patched on pp512: +61.90%
   b12211-patched on tg128: +64.27%
   v090-stock on pp512: -25.73%
   v090-stock on tg128: -0.13%
