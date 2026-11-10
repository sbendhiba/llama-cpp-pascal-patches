| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| jgpu | pp512 | 401.82 | 0.97 | 2 | 42-44 C |
| jgpu | tg128 | 39.77 | 0.00 | 2 | 42-44 C |
| ours | pp512 | 394.45 | 1.92 | 2 | 37-46 C |
| ours | tg128 | 34.04 | 0.04 | 2 | 37-46 C |
| adopt | pp512 | 396.61 | 0.87 | 2 | 41-44 C |
| adopt | tg128 | 38.72 | 0.02 | 2 | 41-44 C |

delta of each other arm against ours:
   jgpu on pp512: +1.87%
   jgpu on tg128: +16.84%
   adopt on pp512: +0.55%
   adopt on tg128: +13.76%
