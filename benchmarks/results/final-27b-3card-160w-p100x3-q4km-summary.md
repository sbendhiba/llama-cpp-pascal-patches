| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| p91 | pp512 | 494.27 | 0.90 | 2 | 43-44 C |
| p91 | tg128 | 39.17 | 0.04 | 2 | 43-44 C |
| ours | pp512 | 490.79 | 1.20 | 2 | 37-46 C |
| ours | tg128 | 34.88 | 0.00 | 2 | 37-46 C |
| theirs | pp512 | 379.83 | 0.22 | 2 | 41-46 C |
| theirs | tg128 | 33.90 | 0.13 | 2 | 41-46 C |

delta of each other arm against ours:
   p91 on pp512: +0.71%
   p91 on tg128: +12.30%
   theirs on pp512: -22.61%
   theirs on tg128: -2.81%
