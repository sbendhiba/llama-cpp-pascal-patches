| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| v090-patched | pp512 | 1072.33 | 3.19 | 3 | 39-41 C |
| v090-patched | tg128 | 119.15 | 0.07 | 3 | 39-41 C |
| b12211-stock | pp512 | 658.04 | 0.24 | 3 | 38-40 C |
| b12211-stock | tg128 | 58.19 | 0.05 | 3 | 38-40 C |
| b12211-patched | pp512 | 1060.92 | 6.68 | 3 | 39-40 C |
| b12211-patched | tg128 | 119.22 | 0.25 | 3 | 39-40 C |

delta of each other arm against b12211-stock:
   v090-patched on pp512: +62.96%
   v090-patched on tg128: +104.76%
   b12211-patched on pp512: +61.22%
   b12211-patched on tg128: +104.88%
