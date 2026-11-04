#!/usr/bin/env python3
"""Digest a directory of llama-perplexity logits dumps and say whether the arm repeats itself.

    ./det-summarize.py <dir with *.gguf dumps>

--save-all-logits writes a raw '_logits_' image despite the file name, and its header carries the run's
shape, not its result - but that is a claim, not a fact, so the header is digested apart from the body.
If two dumps with different answers share a header digest, the header holds nothing run-varying and the
body digest is a fair test of identity; if headers differ, only body-to-body comparisons mean anything.
"""
import hashlib
import os
import re
import sys

d = sys.argv[1] if len(sys.argv) > 1 else '.'
files = sorted(f for f in os.listdir(d) if f.endswith('.gguf'))
if not files:
    sys.exit('no .gguf dumps in %s' % d)

head, body = {}, {}
for f in files:
    buf = open(os.path.join(d, f), 'rb').read()
    if buf[:8] != b'_logits_':
        print('  !! %s does not start with the _logits_ magic - check the tool version' % f)
    name = f[:-5]
    head[name] = hashlib.sha256(buf[:64]).hexdigest()[:12]
    body[name] = hashlib.sha256(buf[64:]).hexdigest()[:16]
    print('  %-16s %10d bytes  header %s  body %s' % (name, len(buf), head[name], body[name]))

print('\n  headers: %d distinct across %d dumps%s' % (
    len(set(head.values())), len(head),
    ' - the header carries nothing run-varying, so body digests are comparable'
    if len(set(head.values())) == 1 else ' - compare bodies only'))


def verdict(label, keys, expect_same=True):
    vals = [body.get(k) for k in keys]
    if any(v is None for v in vals):
        return
    same = len(set(vals)) == 1
    ok = same == expect_same
    print('  %-42s %-9s %s' % (label, 'IDENTICAL' if same else '%d digests' % len(set(vals)),
                               'PASS' if ok else 'FAIL'))


print('\nverdicts:')
for shape, defaults, nog in (('verify  (-ub 4)',    ['default-1', 'default-2', 'default-3'], 'nographs'),
                             ('prefill (-ub 256)',  ['default-pf-1', 'default-pf-2'], 'nographs-pf')):
    keys = [k for k in defaults if k in body]
    if keys:
        verdict('%s: the default repeated %d times' % (shape, len(keys)), keys)
    if nog in body and keys:
        verdict('%s: graphs off vs the default' % shape, [nog] + keys)
