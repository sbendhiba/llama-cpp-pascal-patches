# What the P60 block still needs

Findings from 2026-10-30, exploring every candidate left over from the port. Measured on
`/root/v0.8.0-patch2` (this set at `v0.8.0`) on 2x P100 at 160 W. Nothing here is speculation about
upstream intent - each item is a measurement or a graph dump.

## 1. The persistent kernels are not reproducible run to run at this ref

**Status after this was written: contained by patch 85, not fixed.** 85 refuses CUDA graph capture for
a context once a persistent half has matched a layer, which makes the path reproducible (three
identical greedy runs, one md5) and MTP crash-free over twelve paired requests, and it is keyed on a
kernel having run so the tensor split keeps its graphs. What follows is the diagnosis that led there -
and the real fix is still this: give the barrier a per-launch, self-resetting protocol instead of a
host epoch baked into the arguments, so the kernels can have their graphs back. That is task #33.

Same command, same seed, same binary, greedy, `-sm layer`, Qwen3.5/3.6 MoE UD-Q4_K_XL, 48 tokens:

| config | run 1 | run 2 | run 3 |
|---|---|---|---|
| `GGML_CUDA_P60_MEGA=1` | `b7687ea05c2a` | `99a9fef918fb` | `96b73b87b1bd` |
| the shipped default (`MMVQ` on, no MEGA) | `218ff145fba5` | `218ff145fba5` | - |

(md5 of the generated text.) Three identical MEGA runs give three different answers; the shipped
default is bit-stable. So the persistent layer halves are **nondeterministic on llama.cpp v0.8.0
in plain decode**, before speculative decoding or the GDN matcher question comes up at all.

This is the item to fix first, because it reinterprets the other two:

- the `--spec-type draft-mtp` illegal memory access is most likely the same fault, one step further
  (it did not reproduce in two later runs: 96.35 t/s with graphs, 100.76 without, and
  `compute-sanitizer` completed a full MTP request with 0 errors);
- the garbled greedy decode seen when the persistent GDN half was allowed to match is probably also
  it, not a conv-state layout bug.

The mechanism to start from is the grid barrier's target:

```c
kp.bar  = mega_bar(ctx);              // one unsigned per device context
kp.base = ctx.p60_mega_epoch;         // host-computed at launch time
ctx.p60_mega_epoch += n_barriers * nblk;
```

`base` is baked into the kernel's arguments, and CUDA graphs replay those arguments unchanged. At
`v0.8.0` graphs are captured in **chunks** (`280141e6d`, `9ff0d7e90`) and MTP replays two graphs
against the same device context, so a replayed barrier either passes without waiting (no
synchronisation, blocks read each other's scratch) or waits on a target that never arrives. A fix
has to make the barrier self-contained per launch - an arrival counter that returns to zero inside
the kernel (sense-reversed), or one counter slot per barrier - rather than carrying a host epoch
into a captured graph.

Supporting measurements, same day, MTP enabled (`--spec-type draft-mtp --spec-draft-n-max 5`,
`-c 4096/8192`, 48 tokens, greedy):

| arm | result |
|---|---|
| `MEGA=1`, graphs on | 96.35 t/s, acceptance 35/59, **no crash** |
| `MEGA=1`, `GGML_CUDA_DISABLE_GRAPHS=1` | 100.76 t/s, acceptance 35/58, no crash |
| `MEGA=1`, graphs on (earlier build, larger ctx) | illegal memory access on device 1 |
| `compute-sanitizer --tool memcheck`, `MEGA=1`, MTP | completed, 0 errors, acceptance 0.47 |

The fault is intermittent - two runs that crashed did not crash later, and memcheck never caught it.
That fits a race, and it also means the throughput figure above (96-101 t/s against 73-74 with the
block off) is what MTP *could* have once the barrier is graph-safe: roughly +30%.

## 2. The persistent GDN half: what v0.8.0's graph actually looks like

Their matcher refuses the layer at the conv-state `GET_ROWS`, which it identifies by the node name
`blk.N.conv_states`. A dump of one GDN half at this ref (nodes 2..60 of a 2-token batch, `il=0`)
answers the question that was open at the end of the port:

- the gather itself is unnamed (`node_9`), and its **consumer** carries the name: node 8 is
  `VIEW conv_states-0` of node 7. So the match needs "the gather whose single user is named
  `conv_states-<il>`", not a name on the gather.
- the state side is unchanged in the ways they rely on: node 25 is the state gather (matched by
  `ne[0] == 128*128*32` as before), node 48 is still named `new_state-0`, node 50 is the write-back
  CPY, and the conv-state update CPY at node 19 still has `conv_state_last-0` as its source.
- what v0.8.0 adds is the `build_rs` state-clear chain: nodes 4-6 (`RESHAPE` -> `VIEW` -> in-place
  `SCALE` on `cache_r_l0`, and 22-24 for `cache_s_l0`) and a pair of zero-row
  `VIEW`/`CPY` rollback nodes (9-11, 27-29). Their matcher already handles both - the `SCALE` goes
  to `prerun` and the zero-element CPY is skipped - so neither is the reason it bails.

Accepting the gather by cache name therefore *should* be enough to match the half, and when it was,
the decode came out wrong. Given item 1, the honest reading is that the mismatch is the barrier
race, not these bindings - which is also why refusing the kernel but keeping the match changed the
output again (a dry-run match suppresses the residual-add fusion, and the run was not reproducible).

Order of work: fix the barrier, re-run the determinism test, then widen the gather match and measure
the GDN half for real. The prize is what their tree shows: 125.61 tg128 against our 100.05 at the
same power, i.e. ~+25% decode on the MoE.

## 3. Shipped: the staged Q6_K matvec, re-authored for `VDR_Q6_K_Q8_1_MMVQ == 2`

Their patch 0014 asserted `VDR == 1` and was dropped by the port. At `v0.8.0` a warp takes **two**
superblocks at a time (`qi/vdr == 16` lanes per block, `blocks_per_iter == 8`), so a chunk stages 8
superblocks per row instead of 4 - that is 1680 bytes per row, still 8-byte aligned, and the short
last chunk (when `ne00/QK_K % 8 == 4`, i.e. `ne00` a multiple of 1024 but not 2048) is 840 bytes,
also aligned. Patch 84 re-derives the mapping from stock (`kbx = tid/16`, `iqs = 2*(tid%16)`,
`kby = kbx*4`, full-warp reduce in stock's order), so it is bit-identical: greedy 160-token output
md5 is the same with the kernel on and off, on both a dense Q6_K model and the MoE.

| model | split | `Q6_STAGED=0` | `Q6_STAGED=1` |
|---|---|---:|---:|
| Qwen3.8-27B UD-Q4_K_M | tensor | 386.86 pp512 / 29.71 tg128 | 385.40 / **29.90** (+0.6% tg) |
| Qwen3.8-27B UD-Q4_K_M | layer | 207.99 / 18.34 | 206.51 / **18.54** (+1.1% tg) |
| Cyber-Tiel-35B-A3B Q4_K_XL | tensor | 1114.13 / 87.08 | 1113.68 / 86.95 (path not taken) |

Modest, but it is the only item in the whole block that helps a **dense** model, and it costs nothing
where it does not apply: the gate is dense Q6_K, 1-3 columns, `ne00 % 1024 == 0`, pre-Turing,
GENERIC MMVQ table, `GGML_CUDA_P60_Q6_STAGED=0` turns it off. The pp512 differences are inside the
round spread (sd 0.8-1.7). Their 27B claim was +5.3% MTP decode; we measured plain decode only.

## 4. Not taken

- **Their patch 0007** (fuse `rms_norm * w * silu(z)`, the GDN output norm): +0.4% on their MTP
  benchmark, and it needs a `do_gate` variant of `rms_norm_f32` written against the post-`623c16393`
  vec/scalar dispatch plus a node reorder in `qwen35moe`. Small gain, real work, and the graph is
  already 4 nodes adjacent here (nodes 52-57 in the dump). Deprioritised behind item 1.
- **shinbunbun 45** (qwen4exp QSA sparse gather): 1 of 6 hunks applied, the rest sit on the
  `gather_mask` -> `sel_mask` rename and the pool-level selection rework. Different model family,
  blocked on its own porting task, unrelated to the P60 block.
