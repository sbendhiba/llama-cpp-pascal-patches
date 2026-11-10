| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| jgpu-nogeo | pp512 | 411.33 | 2.30 | 2 | 46-49 C |
| jgpu-nogeo | tg128 | 23.64 | 0.09 | 2 | 46-49 C |
| jgpu | pp512 | 411.58 | 3.30 | 2 | 43-49 C |
| jgpu | tg128 | 29.94 | 0.11 | 2 | 43-49 C |
| ours | pp512 | 396.44 | 0.71 | 2 | 47-48 C |
| ours | tg128 | 23.70 | 0.02 | 2 | 47-48 C |

delta of each other arm against jgpu:
   jgpu-nogeo on pp512: -0.06%
   jgpu-nogeo on tg128: -21.02%
   ours on pp512: -3.68%
   ours on tg128: -20.84%
