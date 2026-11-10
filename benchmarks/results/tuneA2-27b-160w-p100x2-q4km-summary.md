| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| candidateA2 | pp512 | 380.33 | 0.28 | 2 | 45-46 C |
| candidateA2 | tg128 | 32.56 | 0.04 | 2 | 45-46 C |
| ours | pp512 | 380.53 | 1.45 | 2 | 41-47 C |
| ours | tg128 | 29.78 | 0.01 | 2 | 41-47 C |

delta of each other arm against ours:
   candidateA2 on pp512: -0.05%
   candidateA2 on tg128: +9.34%
