| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| p91 | pp512 | 513.76 | 0.32 | 2 | 46-48 C |
| p91 | tg128 | 34.07 | 0.12 | 2 | 46-48 C |
| ours | pp512 | 509.26 | 0.33 | 2 | 41-49 C |
| ours | tg128 | 29.34 | 0.02 | 2 | 41-49 C |
| theirs | pp512 | 397.39 | 0.53 | 2 | 45-49 C |
| theirs | tg128 | 31.20 | 0.02 | 2 | 45-49 C |

delta of each other arm against ours:
   p91 on pp512: +0.88%
   p91 on tg128: +16.13%
   theirs on pp512: -21.97%
   theirs on tg128: +6.33%
