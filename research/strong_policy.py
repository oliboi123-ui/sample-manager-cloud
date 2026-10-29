"""(research, 2026-10-29: needs scikit-learn; HANDOVER 1zq) Scheme 2 measured with the app's REAL decision.
The base for scheme 2 (SampleManagerDev --lab-strongbase: each row = the check's own ++ fingerprint ++ name hint) is
split by pack group; for each pair of groups held out, three model sets are trained on the rest, in the app's format:
    logistic     today's checks (own features), trained the same held-out way: the fair "today"
    trees        gradient-boosted trees on own ++ fingerprint (the name hint left out)
    trees-hint   the same ++ the name hint
then SampleManagerDev --lab-policy2 decides every bench sound with the model of its group (Sorter.Decide).

    python research/strong_policy.py equiv <base2.tsv.gz> <dir>     the app computes what Python does (C# vs here)
    python research/strong_policy.py models <base2.tsv.gz> <dir>    dir/{logistic,trees,trees-hint}/g0..g7.txt
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from app import base, train, trees  # noqa: E402

FP, HINT = 64, 8


def load(path):
    scheme, rows = base.load(path)
    assert scheme == 2, f"scheme {scheme}: not a scheme 2 base (--lab-strongbase)"
    by = {}
    for r in rows:
        by.setdefault(r.check, []).append(r)
    return rows, by


def tree_set(name, ex, hint):
    x = np.array([r.x for r in ex], dtype=float)
    if not hint:
        x[:, -HINT:] = np.nan
    y = np.array([r.label for r in ex], dtype=bool)
    if len(ex) < 40 or y.all() or not y.any():
        return None
    c, _ = trees.train_and_test(name, x, y, train.folds_of(ex))
    return c


def logistic(name, ex):
    own = [replace(r, x=r.x[:len(r.x) - FP - HINT]) for r in ex]
    return train.train_and_test(name, own, [])


def models(path, out):
    rows, by = load(path)
    out = Path(out)
    for g in range(4):
        held = {g, g + 4}
        sets = {"logistic": [], "trees": [], "trees-hint": []}
        for name, ex in by.items():
            tr = [r for r, f in zip(ex, train.folds_of(ex)) if f not in held]
            c = logistic(name, tr)
            if c is not None:
                sets["logistic"].append(c.to_text())
            for s, hint in (("trees", False), ("trees-hint", True)):
                c = tree_set(name, tr, hint)
                if c is not None:
                    sets[s].append(c.to_text())
            print(f"groups {sorted(held)}: {name} ({len(tr)} rows)", flush=True)
        for s, blocks in sets.items():
            (out / s).mkdir(parents=True, exist_ok=True)
            text = f"# research: {s}, trained without pack groups {sorted(held)}\n" + "".join(blocks)
            for k in held:
                (out / s / f"g{k}.txt").write_text(text, encoding="utf-8", newline="\n")


def equiv(path, out):
    """One model of every check on all rows (trees with the hint, and logistic), and Python's P of every row."""
    rows, by = load(path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    blocks = []
    checks = {}
    for name, ex in by.items():
        x = np.array([r.x for r in ex], dtype=float)
        y = np.array([r.label for r in ex], dtype=bool)
        c = trees.read(trees.fit(name, x, y).to_text())[name]
        checks[name] = c
        blocks.append(c.to_text())
    (out / "equiv-model.txt").write_text("# research: equivalence\n" + "".join(blocks), encoding="utf-8", newline="\n")
    p = [checks[r.check].prob(np.array([r.x]))[0] for r in rows]
    (out / "equiv-python.txt").write_text("\n".join(repr(float(v)) for v in p) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rows)} rows, {len(checks)} checks written")


if __name__ == "__main__":
    {"models": models, "equiv": equiv}[sys.argv[1]](sys.argv[2], sys.argv[3])
