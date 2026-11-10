| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| jgpu-novec | pp512 | 391.16 | 0.44 | 2 | 45-48 C |
| jgpu-novec | tg128 | 31.94 | 0.06 | 2 | 45-48 C |
| jgpu | pp512 | 391.34 | 1.07 | 2 | 42-49 C |
| jgpu | tg128 | 35.39 | 0.07 | 2 | 42-49 C |
| ours | pp512 | 379.48 | 0.68 | 2 | 46-48 C |
| ours | tg128 | 29.78 | 0.00 | 2 | 46-48 C |

delta of each other arm against jgpu:
   jgpu-novec on pp512: -0.05%
   jgpu-novec on tg128: -9.74%
   ours on pp512: -3.03%
   ours on tg128: -15.84%
