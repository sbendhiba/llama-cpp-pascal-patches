| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| p91 | pp512 | 469.00 | 0.93 | 2 | 43-45 C |
| p91 | tg128 | 38.45 | 0.00 | 2 | 43-45 C |
| ours | pp512 | 466.51 | 0.32 | 2 | 38-47 C |
| ours | tg128 | 35.69 | 0.06 | 2 | 38-47 C |
| theirs | pp512 | 365.20 | 0.12 | 2 | 41-46 C |
| theirs | tg128 | 32.04 | 0.01 | 2 | 41-46 C |

delta of each other arm against ours:
   p91 on pp512: +0.53%
   p91 on tg128: +7.72%
   theirs on pp512: -21.72%
   theirs on tg128: -10.24%
