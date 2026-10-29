"""(research, 2026-10-29: needs scikit-learn and the bench sorting/*.tsv; HANDOVER 1zq)
Does the cloud's gate let honest answers through, and do they make the sorter decide more of new downloads?
current = trained on the library; users answer half of the download packs (all, or only what current isn't sure of);
measured on the other half (unseen packs)."""
import sys
from dataclasses import replace

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from app import base, train  # noqa: E402

scheme, rows = base.load(HERE.parent / "data" / "base.tsv.gz")
lib = [r for r in rows if r.source == "library"]
new = [r for r in rows if r.source != "library"]
packs = sorted({r.pack for r in new})


def total(checks, rs):
    d = w = 0
    by = {}
    for r in rs:
        by.setdefault(r.check, []).append(r)
    for name, ex in by.items():
        s = train.score(checks.get(name), ex)
        d += s.decided
        w += s.wrong
    return d, w, len(rs)


_, current = train.train_model(lib, "current", [])
for split in (0, 1):
    a_packs = set(packs[split::2])
    answered = [r for r in new if r.pack in a_packs]
    unseen = [r for r in new if r.pack not in a_packs]
    unsure = [r for r in answered if (c := current.get(r.check)) is None or c.sure > 1 or not
              (c.prob(__import__("numpy").array([r.x]))[0] >= c.sure or c.prob(__import__("numpy").array([r.x]))[0] <= 1 - c.sure)]
    print(f"\n=== split {split}: users answer {len(a_packs)} packs, {len(unseen)} rows of {len(packs) - len(a_packs)} packs unseen")
    d, w, n = total(current, unseen)
    print(f"  current                         unseen: decides {d:5}/{n} ({100*d/n:.1f}%), wrong {w}")
    for label, shared in (("answers to every sound", answered), ("answers to unsure ones only", unsure)):
        shared = [replace(r, source="shared") for r in shared]
        _, cand = train.train_model(lib + shared, "candidate", [])
        g = train.gate(cand, current, lib)
        d, w, n = total(cand, unseen)
        print(f"  + {label:<28}  unseen: decides {d:5}/{n} ({100*d/n:.1f}%), wrong {w}  | {len(shared)} rows | gate: {'PASS' if g.passed else 'REJECT'}")
        for why in g.why:
            print(f"        {why}")
