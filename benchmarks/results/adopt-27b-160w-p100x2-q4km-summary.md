| arm | test | mean t/s | sd across rounds | rounds | temps C |
|---|---|---:|---:|---:|---|
| jgpu | pp512 | 392.15 | 0.33 | 2 | 47-47 C |
| jgpu | tg128 | 35.41 | 0.01 | 2 | 47-47 C |
| ours | pp512 | 380.03 | 1.59 | 2 | 43-48 C |
| ours | tg128 | 29.71 | 0.04 | 2 | 43-48 C |
| adopt | pp512 | 381.67 | 1.76 | 2 | 46-48 C |
| adopt | tg128 | 34.19 | 0.02 | 2 | 46-48 C |

delta of each other arm against ours:
   jgpu on pp512: +3.19%
   jgpu on tg128: +19.21%
   adopt on pp512: +0.43%
   adopt on tg128: +15.10%
