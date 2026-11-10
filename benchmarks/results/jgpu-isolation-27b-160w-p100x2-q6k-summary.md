| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| jgpu-novec | pp512 | 411.50 | 2.17 | 2 | 46-48 C |
| jgpu-novec | tg128 | 28.69 | 0.10 | 2 | 46-48 C |
| jgpu | pp512 | 411.62 | 4.82 | 2 | 42-49 C |
| jgpu | tg128 | 29.93 | 0.14 | 2 | 42-49 C |
| ours | pp512 | 397.68 | 2.95 | 2 | 46-48 C |
| ours | tg128 | 23.72 | 0.00 | 2 | 46-48 C |

delta of each other arm against jgpu:
   jgpu-novec on pp512: -0.03%
   jgpu-novec on tg128: -4.13%
   ours on pp512: -3.39%
   ours on tg128: -20.75%
