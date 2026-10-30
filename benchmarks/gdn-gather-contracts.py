#!/usr/bin/env python3
"""Contract test for patch 87's gather matcher, against a real ggml context.

    ./gdn-gather-contracts.py /path/to/patched/tree

Extracts p60_gdn_gather() out of that tree's ggml-cuda.cu, compiles it alone against libggml-base, and
checks what it accepts and what it refuses. The persistent GDN half reads one row of the conv-state and
recurrent-state caches; v0.8.0 puts the layer name on a zero-offset VIEW *above* the gather, so the
matcher has to walk down through VIEW/RESHAPE edges and prove the row contract rather than trust a name.
Every refusal below is a graph that must not be matched.
"""
import subprocess
import sys
from pathlib import Path

tree = Path(sys.argv[1] if len(sys.argv) > 1 else '/root/v0.8.0-patch3')
src = tree / 'ggml/src/ggml-cuda/ggml-cuda.cu'
s = src.read_text()
start = s.index('static const ggml_tensor * p60_gdn_gather(')
end = s.index('\n}\n', start) + 3
helper = s[start:end]

test = r'''#include "ggml.h"
#include <cassert>
#include <cstdio>
static bool p60_f32_vec(const ggml_tensor * t, int64_t n) {
    return t && t->type == GGML_TYPE_F32 && ggml_nelements(t) == n && ggml_is_contiguous(t);
}
HELPER
int main() {
    auto * c = ggml_init({8 * 1024 * 1024, nullptr, true});
    assert(c);
    constexpr int64_t W = 3 * 8192;                 // one row of the conv-state cache
    auto * cache = ggml_new_tensor_2d(c, GGML_TYPE_F32, W, 8);
    auto * ids   = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    auto * gr    = ggml_get_rows(c, cache, ids);
    auto * v     = ggml_view_2d(c, gr, W, 1, gr->nb[1], 0);      // what build_rs names
    auto * rs    = ggml_reshape_3d(c, v, 3, 8192, 1);            // what SSM_CONV reads
    assert(p60_gdn_gather(rs, W) == gr);
    assert(p60_gdn_gather(gr, W) == gr);
    auto shifted = *v; shifted.view_offs = sizeof(float);        assert(!p60_gdn_gather(&shifted, W));
    auto strided = *rs; strided.nb[1] += sizeof(float);          assert(!p60_gdn_gather(&strided, W));
    auto * ids2  = ggml_new_tensor_1d(c, GGML_TYPE_I32, 2);
    assert(!p60_gdn_gather(ggml_get_rows(c, cache, ids2), W));   // two rows is not one
    auto * cache16 = ggml_new_tensor_2d(c, GGML_TYPE_F16, W, 8);
    assert(!p60_gdn_gather(ggml_get_rows(c, cache16, ids), W));  // wrong cache type
    auto * state  = ggml_new_tensor_2d(c, GGML_TYPE_F32, 128 * 128 * 32, 8);
    auto * sg     = ggml_get_rows(c, state, ids);
    auto * sr     = ggml_reshape_4d(c, sg, 128, 128, 32, 1);
    assert(p60_gdn_gather(sr, 128 * 128 * 32) == sg);            // the recurrent-state side
    assert(!p60_gdn_gather(sr, W));                              // and not at the conv width
    auto badcache = *cache; badcache.nb[1] += sizeof(float);
    auto badgr    = *gr;  badgr.src[0] = &badcache;              assert(!p60_gdn_gather(&badgr, W));
    auto badids   = *ids; badids.type = GGML_TYPE_I64;
    auto badgr2   = *gr;  badgr2.src[1] = &badids;               assert(!p60_gdn_gather(&badgr2, W));
    ggml_free(c);
    puts("PASS gather contracts: valid conv/state chains, offsets, strides, rows, types");
}
'''.replace('HELPER', helper)

stem = Path('/tmp/gdn-gather-contracts')
stem.with_suffix('.cpp').write_text(test)
subprocess.run(['g++', '-std=c++17', str(stem.with_suffix('.cpp')),
                '-I' + str(tree / 'ggml/include'), '-L' + str(tree / 'build/bin'),
                '-Wl,-rpath,' + str(tree / 'build/bin'), '-lggml-base', '-o', str(stem)], check=True)
subprocess.run([str(stem)], check=True)
