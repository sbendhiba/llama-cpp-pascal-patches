| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| junkgpulab | pp512 | 391.91 | 0.36 | 2 | 41-47 C |
| junkgpulab | tg128 | 35.40 | 0.07 | 2 | 41-47 C |
| ours-b12211 | pp512 | 381.00 | 0.14 | 2 | 45-46 C |
| ours-b12211 | tg128 | 29.75 | 0.04 | 2 | 45-46 C |

delta of each other arm against junkgpulab:
   ours-b12211 on pp512: -2.78%
   ours-b12211 on tg128: -15.96%
