"""(research, 2026-10-29: needs scikit-learn and the bench sorting/*.tsv; HANDOVER 1zq)
How much could each check decide on its own (packs held out, <= 0.3% wrong) with a stronger model?
  logistic   the app's checks today (the server's port of Check.Train), on each check's own features
  trees      gradient-boosted trees on the same features (the model is the limit?)
  trees+fp   trees on the features + the sound's whole fingerprint (what it hears is the limit?)"""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from app import base, train  # noqa: E402

ROOT = HERE.parent.parent / "sorting"  # (the developer's test bench, beside cloud\)
fps = {}
for name in ("library.tsv", "downloads.tsv"):
    for line in open(ROOT / name, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) < 10:
            continue
        try:
            fps[f"{c[0]}\\{c[1]}"] = np.array([float(t) if t else 0.0 for t in c[2:]])
        except ValueError:
            pass

scheme, rows = base.load(HERE.parent / "data" / "base.tsv.gz")
joined = sum(r.path in fps for r in rows)
print(f"{len(rows)} rows, {joined} with a fingerprint ({100 * joined / len(rows):.1f}%)")
LADDER = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999]


def bar(p, y, ladder):
    """The lowest threshold at which held-out decisions are <= 0.3% wrong (>= 20 decided): (threshold, decided, wrong)."""
    for th in ladder:
        dec = (p >= th) | (p <= 1 - th)
        d = int(dec.sum())
        w = int(((p[dec] >= th) != y[dec]).sum())
        if d >= 20 and w <= 0.003 * d:
            return th, d, w
    return None, 0, 0


def held_out(make, X, y, fold):
    p = np.full(len(y), np.nan)
    for f in range(8):
        hold, tr = fold == f, fold != f
        if tr.sum() < 20 or y[tr].all() or not y[tr].any() or not hold.any():
            continue
        p[hold] = make(X[tr], y[tr], X[hold])
    return p


def logistic(Xt, yt, Xh, name):
    return train.train_check(name, Xt, yt).prob(Xh)


def trees(Xt, yt, Xh):
    w = np.where(yt, 0.5 / yt.mean(), 0.5 / (1 - yt.mean()))
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20,
                                       l2_regularization=1.0, random_state=1)
    m.fit(Xt, yt, sample_weight=w)
    return m.predict_proba(Xh)[:, 1]


by = {}
for r in rows:
    by.setdefault(r.check, []).append(r)
tot = {k: [0, 0] for k in ("logistic", "trees", "trees+fp")}
n_all = 0
print(f"\n{'check':<22}{'rows':>6} | {'logistic today':>18} | {'trees, same features':>21} | {'trees + fingerprint':>20}")
for name, ex in by.items():
    ex = [r for r in ex if r.path in fps]
    y = np.array([r.label for r in ex], dtype=bool)
    if len(ex) < 40 or y.all() or not y.any():
        continue
    X = np.array([r.x for r in ex], dtype=float)
    F = np.hstack([X, np.array([fps[r.path] for r in ex])])
    fold = train.folds_of(ex)
    out = {}
    out["logistic"] = bar(held_out(lambda a, b, c: logistic(a, b, c, name), X, y, fold), y, LADDER[:8])
    out["trees"] = bar(held_out(trees, X, y, fold), y, LADDER)
    out["trees+fp"] = bar(held_out(trees, F, y, fold), y, LADDER)
    n_all += len(ex)
    cells = []
    for k in ("logistic", "trees", "trees+fp"):
        th, d, w = out[k]
        tot[k][0] += d
        tot[k][1] += w
        cells.append(f"{100 * d / len(ex):5.1f}% ({w:2} wrong)" if th else f"{'never':>16}")
    print(f"{name:<22}{len(ex):>6} | {cells[0]:>18} | {cells[1]:>21} | {cells[2]:>20}")
print(f"{'all':<22}{n_all:>6} | " + " | ".join(f"{100 * tot[k][0] / n_all:5.1f}% ({tot[k][1]:3} wrong)" for k in tot))
