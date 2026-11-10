| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| jgpu | pp512 | 400.53 | 3.56 | 2 | 39-47 C |
| jgpu | tg128 | 39.76 | 0.03 | 2 | 39-47 C |
| ours | pp512 | 390.40 | 3.12 | 2 | 43-44 C |
| ours | tg128 | 34.12 | 0.01 | 2 | 43-44 C |
| jgpu-nostage | pp512 | 402.84 | 0.95 | 2 | 43-46 C |
| jgpu-nostage | tg128 | 38.82 | 0.04 | 2 | 43-46 C |

delta of each other arm against jgpu:
   ours on pp512: -2.53%
   ours on tg128: -14.19%
   jgpu-nostage on pp512: +0.58%
   jgpu-nostage on tg128: -2.38%
