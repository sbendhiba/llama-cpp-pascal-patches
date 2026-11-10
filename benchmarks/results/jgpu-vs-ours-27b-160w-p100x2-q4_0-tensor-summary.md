| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| junkgpulab | pp512 | 401.72 | 1.09 | 2 | 38-44 C |
| junkgpulab | tg128 | 39.73 | 0.03 | 2 | 38-44 C |
| ours-b12211 | pp512 | 395.19 | 1.11 | 2 | 41-42 C |
| ours-b12211 | tg128 | 34.12 | 0.01 | 2 | 41-42 C |

delta of each other arm against junkgpulab:
   ours-b12211 on pp512: -1.63%
   ours-b12211 on tg128: -14.12%
