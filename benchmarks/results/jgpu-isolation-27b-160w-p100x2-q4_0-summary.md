| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| jgpu-novec | pp512 | 401.74 | 0.18 | 2 | 41-45 C |
| jgpu-novec | tg128 | 39.77 | 0.07 | 2 | 41-45 C |
| jgpu | pp512 | 402.33 | 1.92 | 2 | 38-46 C |
| jgpu | tg128 | 39.78 | 0.00 | 2 | 38-46 C |
| ours | pp512 | 394.76 | 0.97 | 2 | 42-44 C |
| ours | tg128 | 34.11 | 0.01 | 2 | 42-44 C |

delta of each other arm against jgpu:
   jgpu-novec on pp512: -0.15%
   jgpu-novec on tg128: -0.02%
   ours on pp512: -1.88%
   ours on tg128: -14.24%
