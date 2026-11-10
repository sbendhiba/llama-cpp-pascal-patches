| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| jgpu | pp512 | 410.48 | 0.54 | 2 | 47-48 C |
| jgpu | tg128 | 29.92 | 0.02 | 2 | 47-48 C |
| ours | pp512 | 398.90 | 2.73 | 2 | 42-49 C |
| ours | tg128 | 23.72 | 0.03 | 2 | 42-49 C |
| adopt | pp512 | 400.33 | 0.53 | 2 | 46-48 C |
| adopt | tg128 | 28.05 | 0.05 | 2 | 46-48 C |

delta of each other arm against ours:
   jgpu on pp512: +2.90%
   jgpu on tg128: +26.11%
   adopt on pp512: +0.36%
   adopt on tg128: +18.22%
