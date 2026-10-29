"""(research, 2026-10-29: needs scikit-learn and the bench sorting/*.tsv; HANDOVER 1zq)
Do honest shared answers make a check decide more of downloads it never saw?  (as the cloud would: the "sure" bar
from packs held out of what it trained on, then judged on unseen download packs)
  today      logistic on the check's own features (the app's checks)
  strong     gradient-boosted trees on the features + the whole fingerprint
each trained on the library alone, then on the library + answers to half of the download packs (all of their sounds, or
only those today's checks weren't sure of); judged on the other half."""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from app import base, train  # noqa: E402

ROOT = HERE.parent.parent / "sorting"  # (the developer's test bench, beside cloud\)
fps = {}
for name, lib in (("library.tsv", True), ("downloads.tsv", False)):
    for line in open(ROOT / name, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) < 10:
            continue
        try:
            fps[f"{c[0]}\\{c[1]}" if lib else c[1]] = np.array([float(t) if t else 0.0 for t in c[2:]])
        except ValueError:
            pass

scheme, rows = base.load(HERE.parent / "data" / "base.tsv.gz")
rows = [r for r in rows if r.path in fps]
lib = [r for r in rows if r.source == "library"]
new = [r for r in rows if r.source != "library"]
packs = sorted({r.pack for r in new})
print(f"{len(lib)} library rows, {len(new)} download rows ({len(packs)} packs), all with fingerprints")
LADDER = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999]


def trees_fit(X, y):
    w = np.where(y, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=20,
                                       l2_regularization=1.0, random_state=1)
    return m.fit(X, y, sample_weight=w)


def feats(rs, strong):
    X = np.array([r.x for r in rs], dtype=float)
    return np.hstack([X, np.array([fps[r.path] for r in rs])]) if strong else X


def fit_with_bar(rs, name, strong):
    """(predict, bar): trained on rs; the bar from packs of rs held out (<= 0.3% wrong, >= 20 decided)."""
    y = np.array([r.label for r in rs], dtype=bool)
    X = feats(rs, strong)
    fold = train.folds_of(rs)
    p = np.full(len(y), np.nan)
    for f in range(8):
        hold, tr = fold == f, fold != f
        if tr.sum() < 20 or y[tr].all() or not y[tr].any() or not hold.any():
            continue
        p[hold] = (trees_fit(X[tr], y[tr]).predict_proba(X[hold])[:, 1] if strong else train.train_check(name, X[tr], y[tr]).prob(X[hold]))
    has = ~np.isnan(p)
    bar = None
    for th in (LADDER if strong else LADDER[:8]):
        dec = (p[has] >= th) | (p[has] <= 1 - th)
        d, w = int(dec.sum()), int(((p[has][dec] >= th) != y[has][dec]).sum())
        if d >= 20 and w <= 0.003 * d:
            bar = th
            break
    if strong:
        m = trees_fit(X, y)
        return (lambda Z: m.predict_proba(Z)[:, 1]), bar
    c = train.train_check(name, X, y)
    return c.prob, bar


def judge(pred, bar, rs, strong):
    if bar is None or not rs:
        return 0, 0
    p = pred(feats(rs, strong))
    y = np.array([r.label for r in rs], dtype=bool)
    dec = (p >= bar) | (p <= 1 - bar)
    return int(dec.sum()), int(((p[dec] >= bar) != y[dec]).sum())


def by_check(rs):
    out = {}
    for r in rs:
        out.setdefault(r.check, []).append(r)
    return out


for split in (0, 1):
    a_packs = set(packs[split::2])
    answered = [r for r in new if r.pack in a_packs]
    unseen = by_check([r for r in new if r.pack not in a_packs])
    n_unseen = sum(len(v) for v in unseen.values())
    # what today's checks (trained on the library) aren't sure of: what NEEDS A LOOK would ask about
    today = {name: fit_with_bar(ex, name, False) for name, ex in by_check(lib).items() if len(ex) >= 40}
    unsure = []
    for r in answered:
        t = today.get(r.check)
        if t is None or t[1] is None:
            unsure.append(r)
            continue
        q = t[0](np.array([r.x]))[0]
        if not (q >= t[1] or q <= 1 - t[1]):
            unsure.append(r)
    print(f"\n=== split {split}: answers from {len(a_packs)} packs ({len(answered)} rows, {len(unsure)} unsure), judged on {n_unseen} rows of unseen packs")
    for strong in (False, True):
        for label, extra in (("library only", []), ("+ answers to every sound", answered), ("+ answers to unsure ones", unsure)):
            train_by = by_check(lib + extra)
            d = w = 0
            for name, test in unseen.items():
                ex = train_by.get(name, [])
                yv = [r.label for r in ex]
                if len(ex) < 40 or all(yv) or not any(yv):
                    continue
                pred, bar = fit_with_bar(ex, name, strong)
                dd, ww = judge(pred, bar, test, strong)
                d += dd
                w += ww
            print(f"  {'strong' if strong else 'today '}  {label:<26} unseen: decides {100 * d / n_unseen:5.1f}%  ({d:5}), wrong {w:3} ({100 * w / max(1, d):.2f}%)", flush=True)
