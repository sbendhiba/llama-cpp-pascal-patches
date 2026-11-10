| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| jgpu-nogeo | pp512 | 392.17 | 1.07 | 2 | 46-49 C |
| jgpu-nogeo | tg128 | 25.38 | 0.09 | 2 | 46-49 C |
| jgpu | pp512 | 393.18 | 0.50 | 2 | 43-49 C |
| jgpu | tg128 | 35.43 | 0.10 | 2 | 43-49 C |
| ours | pp512 | 379.79 | 1.08 | 2 | 47-48 C |
| ours | tg128 | 29.72 | 0.02 | 2 | 47-48 C |

delta of each other arm against jgpu:
   jgpu-nogeo on pp512: -0.26%
   jgpu-nogeo on tg128: -28.38%
   ours on pp512: -3.41%
   ours on tg128: -16.12%
