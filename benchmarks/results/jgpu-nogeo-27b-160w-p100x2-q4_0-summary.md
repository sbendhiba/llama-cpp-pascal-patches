| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| jgpu-nogeo | pp512 | 402.69 | 0.04 | 2 | 41-45 C |
| jgpu-nogeo | tg128 | 29.85 | 0.11 | 2 | 41-45 C |
| jgpu | pp512 | 400.96 | 3.88 | 2 | 38-46 C |
| jgpu | tg128 | 39.76 | 0.06 | 2 | 38-46 C |
| ours | pp512 | 393.46 | 1.13 | 2 | 43-44 C |
| ours | tg128 | 34.07 | 0.08 | 2 | 43-44 C |

delta of each other arm against jgpu:
   jgpu-nogeo on pp512: +0.43%
   jgpu-nogeo on tg128: -24.94%
   ours on pp512: -1.87%
   ours on tg128: -14.31%
