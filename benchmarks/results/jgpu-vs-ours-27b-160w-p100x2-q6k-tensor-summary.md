| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| junkgpulab | pp512 | 411.57 | 1.08 | 2 | 43-48 C |
| junkgpulab | tg128 | 29.94 | 0.08 | 2 | 43-48 C |
| ours-b12211 | pp512 | 396.09 | 0.05 | 2 | 46-47 C |
| ours-b12211 | tg128 | 23.73 | 0.00 | 2 | 46-47 C |

delta of each other arm against junkgpulab:
   ours-b12211 on pp512: -3.76%
   ours-b12211 on tg128: -20.75%
