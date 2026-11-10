#!/usr/bin/env python3
"""Strict-output probe: the failure modes a prose corpus cannot see.

Run once per arm, against a llama-server that agentic-gate.sh has already identity-checked:

    python3 agentic-probe.py <label> [port]

16 fixed prompts at temperature 0 - JSON that must parse, exact tool and git commands, code completions,
format constraints. Why not compare generated text instead: wikitext perplexity and logit KL say nothing
about the things agentic use breaks on, and a greedy sample comparison both over- and under-reports (a
schema that stops parsing is a hard failure no PPL will show). The absolute pass rate is deliberately not
the point - the validators assume particular key names, so a correct answer spelled differently counts
as a miss. The gate's answer is the *difference* between two builds on identical prompts.

FAIL lines go to stderr as `CASE <i> <kind> <PASS|FAIL> <note> <reply>` so the driver can diff arms.
"""
import json
import sys
import urllib.request

CASES = [
    # (kind, prompt, validator)
    ('json', 'Extract as JSON only, no prose, keys name (string), amount (integer): invoice from ACME for 42 dollars',
     lambda t: isinstance(json.loads(t.strip()).get('name'), str) and isinstance(json.loads(t.strip()).get('amount'), int)),
    ('json', 'Output only a JSON object with key items, an array of 3 strings: colors of a traffic light',
     lambda t: len(json.loads(t.strip())['items']) == 3 and all(isinstance(x, str) for x in json.loads(t.strip())['items'])),
    ('json', 'Respond with JSON only: {"ok": <boolean>, "n": <integer 1-3>} for "yes, second one"',
     lambda t: (lambda o: o['ok'] is True and 1 <= o['n'] <= 3)(json.loads(t.strip()))),
    ('json', 'Convert to JSON, nothing else: user id 7, active true, tags a,b',
     lambda t: (lambda o: o.get('user') == 7 or o.get('id') == 7 or o.get('user_id') == 7)(json.loads(t.strip()) or {})),
    ('json', 'Return a JSON array of objects with keys k and v, only that: x=1, y=2',
     lambda t: all(set(o.keys()) == {'k', 'v'} for o in json.loads(t.strip()))),

    ('exact', 'Complete the tool call name exactly: to list files in a directory, call read_dir? no -- the tool is named__',
     lambda t: 'list_dir' in t or 'list_files' in t or 'ls' in t),
    ('exact', 'What is the GitHub CLI command to view pull request 12 in repo o/r? Answer with the command only.',
     lambda t: 'gh pr view' in t and '12' in t),
    ('exact', 'Output only the git command that discards all uncommitted changes in the working tree.',
     lambda t: 'git checkout' in t or 'git restore' in t),

    ('code', 'Write only the body of: def merge(a, b): that returns a merged sorted list, use a while loop, no explanation',
     lambda t: 'while' in t and 'return' in t and '```' not in t),
    ('code', 'Complete exactly, no prose: if __name__ == "__main__":',
     lambda t: 'main' in t),
    ('code', 'Give only the Python expression that safely reads d["k"] with default 0 when missing.',
     lambda t: 'get(' in t),

    ('fmt', 'Answer with exactly three bullet lines, each starting with "- ", about TCP: ',
     lambda t: len([l for l in t.strip().splitlines() if l.startswith('- ')]) == 3 and t.strip().splitlines()[0].startswith('- ')),
    ('fmt', 'Answer in exactly two words, no punctuation: opposite of "increase"',
     lambda t: len(t.strip().split()) <= 3),
    ('fmt', 'Output only a valid cron expression for "every weekday at 9am", nothing else.',
     lambda t: len(t.strip().split()) == 5 and t.strip().split()[4] in ('1-5', '1,2,3,4,5', 'mon-fri', 'MON-FRI')),
    ('fmt', 'Reply with only the ISO 8601 date for 2 November 2026, no time, no text.',
     lambda t: t.strip().startswith('2026-11-02')),
    ('fmt', 'List exactly the numbers 1 to 5 separated by commas, nothing else.',
     lambda t: t.strip().replace(' ', '') in ('1,2,3,4,5',)),
]


def ask(prompt, port, n_predict=96):
    body = json.dumps({'prompt': prompt, 'n_predict': n_predict, 'temperature': 0}).encode()
    req = urllib.request.Request('http://127.0.0.1:%d/completion' % port, data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode())['content']


def main(label, port):
    rows = []
    for i, (kind, prompt, ok) in enumerate(CASES):
        try:
            out = ask(prompt, port)
        except Exception as e:
            rows.append(dict(i=i, kind=kind, pass_=False, err=repr(e)[:120], text=''))
            continue
        try:
            passed = bool(ok(out))
        except Exception:
            passed = False          # a reply that cannot be parsed by its own validator is a failure
        rows.append(dict(i=i, kind=kind, pass_=passed, err='' if passed else 'validator rejected',
                         text=out.strip().replace('\n', '\\n')[:160]))
    n = len(rows)
    p = sum(1 for r in rows if r['pass_'])
    by_kind = {}
    for r in rows:
        a = by_kind.setdefault(r['kind'], [0, 0])
        a[0] += 1
        a[1] += 1 if r['pass_'] else 0
    print(json.dumps(dict(label=label, pass_rate=round(100.0 * p / n, 1), passed=p, total=n,
                          by_kind={k: '%d/%d' % (v[1], v[0]) for k, v in sorted(by_kind.items())})))
    for r in rows:
        print('CASE %s %-5s %-5s %s %s' % (r['i'], r['kind'], 'PASS' if r['pass_'] else 'FAIL',
                                           r['err'][:40], r['text']), file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'arm',
         int(sys.argv[2]) if len(sys.argv) > 2 else 8099)
