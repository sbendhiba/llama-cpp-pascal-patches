| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| jgpu | pp512 | 392.10 | 2.44 | 2 | 43-49 C |
| jgpu | tg128 | 35.41 | 0.05 | 2 | 43-49 C |
| ours | pp512 | 379.80 | 1.08 | 2 | 47-48 C |
| ours | tg128 | 29.72 | 0.06 | 2 | 47-48 C |
| jgpu-nostage | pp512 | 392.22 | 0.58 | 2 | 46-48 C |
| jgpu-nostage | tg128 | 34.37 | 0.02 | 2 | 46-48 C |

delta of each other arm against jgpu:
   ours on pp512: -3.14%
   ours on tg128: -16.07%
   jgpu-nostage on pp512: +0.03%
   jgpu-nostage on tg128: -2.95%
