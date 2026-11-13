#!/usr/bin/env python3
"""One MTP arm, four prompts, temperature 0, fixed n_predict.

Emits one JSON object per line - `mtp-ab.sh` appends them and summarizes - so the collector keeps
whatever llama.cpp exposes today without guessing at field names.

What llama.cpp v1.0.0 actually puts in `timings` for a speculative run is `draft_n` and
`draft_n_accepted`, and nothing else about cycle length. The two derived numbers:

    steps      = predicted_n - draft_n_accepted   (every step emits at least one token, the bonus
                                                  token, so accepted tokens are the ones that came
                                                  for free on top of the step's own)
    mean_len   = predicted_n / steps              (tokens per decode cycle)
    per_cycle  = steps / predicted_ms * 1000      (cycles per second)

per_cycle is the number to compare between builds: raw t/s contains the acceptance rate, and an arm
that drafts badly looks slow for reasons that have nothing to do with its kernels. Both arms run the
same prompts at temperature 0, so if the acceptance differs it is because the build changed the
draft logits, which is itself a finding - that is why `mean_len` is printed next to it rather than
buried.

`ignore_eos` and temperature 0 make the token sequence a function of the build alone; an arm that
stops early on its own looks fast and is not.
"""
import json
import sys
import urllib.request

port = sys.argv[1]
prompts_file = sys.argv[2]
n_gen = int(sys.argv[3])
label = sys.argv[4]
rep = sys.argv[5] if len(sys.argv) > 5 else "1"


def load(path):
    out, name, buf = [], None, []
    for line in open(path, encoding="utf-8"):
        if line.startswith("=== "):
            if name is not None:
                out.append((name, "".join(buf).strip()))
            name, buf = line[4:].split()[0], []
        else:
            buf.append(line)
    if name is not None:
        out.append((name, "".join(buf).strip()))
    return out


for name, text in load(prompts_file):
    body = json.dumps({
        "prompt": text,
        "n_predict": n_gen,
        "temperature": 0.0,
        "stream": False,
        "cache_prompt": False,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%s/completion" % port, body, {"Content-Type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=3600).read().decode())
    except Exception as exc:                                    # noqa: BLE001
        print(json.dumps({"label": label, "rep": rep, "prompt": name, "error": repr(exc)}), flush=True)
        continue
    t = d.get("timings", {})
    n = t.get("predicted_n") or 0
    acc = t.get("draft_n_accepted", t.get("accepted"))
    ms = t.get("predicted_ms") or 0.0
    steps = n - acc if acc is not None else None
    rec = {
        "label": label,
        "rep": rep,
        "prompt": name,
        "n_gen": n,
        "pps": t.get("predicted_per_second"),
        "draft_n": t.get("draft_n", t.get("drafted")),
        "accepted": acc,
        "steps": steps,
        "mean_len": (n / steps) if steps else None,
        "per_cycle": (steps * 1000.0 / ms) if (steps and ms) else None,
        "raw": t,
    }
    print(json.dumps(rec), flush=True)
