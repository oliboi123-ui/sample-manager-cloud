"""Scheme 2's checks: gradient-boosted trees (scikit-learn's HistGradientBoosting, both answers weighing the same) on a
check's own features, the sound's whole fingerprint and the name hint. Written as text in the model file (the app's
Sorter.cs TreeCheck reads the same); the text IS the model: what the bar is set with and what the app computes are the
very same numbers (prob() here parses it back, it doesn't ask scikit-learn). Its numbers are written as their 64 bits
(16 hex digits): .NET Framework reads some decimals one bit off ("-1.89637"), and a sound exactly on a threshold would
then go the other way in the app.

    trees <name> <dims>
    base <raw score before the first tree, bits>
    tree <nodes>
    <feature> <threshold bits> <missing goes left 0/1> <left> <right>   (a split: x[feature] <= threshold -> left)
    leaf <value bits>                                                    (a leaf)
    ...
    sure <bar>
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from .train import NEVER

LADDER = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999]
PARAMS = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
              early_stopping=False, random_state=1)


@dataclass
class TreeCheck:
    name: str
    dims: int
    base: float
    # per tree: feature (-1 = leaf), threshold, missing goes left, left, right, value (leaves)
    trees: list[np.ndarray] = field(default_factory=list)
    sure: float = NEVER

    def raw(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        out = np.full(len(x), self.base)
        rows = np.arange(len(x))
        for t in self.trees:
            node = np.zeros(len(x), dtype=int)
            while True:
                f = t[node, 0].astype(int)
                inner = f >= 0
                if not inner.any():
                    break
                i = rows[inner]
                v = x[i, f[inner]]
                left = np.where(np.isnan(v), t[node[inner], 2] > 0, v <= t[node[inner], 1])
                node[inner] = np.where(left, t[node[inner], 3], t[node[inner], 4]).astype(int)
            out = out + t[node, 5]
        return out

    def prob(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.raw(x)))

    def to_text(self) -> str:
        lines = [f"trees {self.name} {self.dims}", f"base {bits(self.base)}"]
        for t in self.trees:
            lines.append(f"tree {len(t)}")
            for f, th, miss, l, r, v in t:
                lines.append(f"leaf {bits(v)}" if f < 0 else f"{int(f)} {bits(th)} {int(miss)} {int(l)} {int(r)}")
        lines.append(f"sure {float(self.sure)!r}")
        return "\n".join(lines) + "\n"


def bits(v: float) -> str:
    """A number as its 64 bits, 16 hex digits (the app: BitConverter.Int64BitsToDouble)."""
    return struct.pack(">d", float(v)).hex()


def unbits(s: str) -> float:
    return struct.unpack(">d", bytes.fromhex(s))[0]


def fit(name: str, x: np.ndarray, y: np.ndarray) -> TreeCheck:
    """Trees for one check: scikit-learn's own numbers, exactly (written as bits)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    y = np.asarray(y, dtype=bool)
    x = np.asarray(x, dtype=float)
    gone = np.isnan(x).all(axis=0)
    if gone.any():  # (a column missing in every row, e.g. no hint at all: scikit-learn can't bin it; a constant is never split on)
        x = x.copy()
        x[:, gone] = 0.0
    w = np.where(y, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    m = HistGradientBoostingClassifier(**PARAMS).fit(x, y, sample_weight=w)
    trees = []
    for it in m._predictors:
        nodes = it[0].nodes
        t = np.zeros((len(nodes), 6))
        for k, n in enumerate(nodes):
            if n["is_leaf"]:
                t[k] = (-1, 0, 0, 0, 0, float(n["value"]))
            else:
                t[k] = (int(n["feature_idx"]), float(n["num_threshold"]), 1 if n["missing_go_to_left"] else 0,
                        int(n["left"]), int(n["right"]), 0)
        trees.append(t)
    return TreeCheck(name, x.shape[1], float(np.ravel(m._baseline_prediction)[0]), trees)


def read(text: str) -> dict[str, TreeCheck]:
    """The tree checks of a model's text (its logistic ones: train.read_model)."""
    lines = text.replace("\r", "").split("\n")
    out: dict[str, TreeCheck] = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("trees "):
            i += 1
            continue
        _, name, dims = lines[i].split(" ")
        c = TreeCheck(name, int(dims), unbits(lines[i + 1].split(" ")[1]))
        i += 2
        while i < len(lines) and lines[i].startswith("tree "):
            n = int(lines[i].split(" ")[1])
            t = np.zeros((n, 6))
            for k in range(n):
                p = lines[i + 1 + k].split(" ")
                t[k] = (-1, 0, 0, 0, 0, unbits(p[1])) if p[0] == "leaf" else (int(p[0]), unbits(p[1]), int(p[2]), int(p[3]), int(p[4]), 0)
            c.trees.append(t)
            i += 1 + n
        if i < len(lines) and lines[i].startswith("sure "):
            c.sure = float(lines[i].split(" ")[1])
            i += 1
        out[name] = c
    return out


def bar(p: np.ndarray, y: np.ndarray) -> tuple[float, int, int]:
    """The lowest confidence at which held-out decisions were at most 0.3% wrong (at least 20): (bar, decided, wrong)."""
    for th in LADDER:
        dec = (p >= th) | (p <= 1 - th)
        d = int(dec.sum())
        w = int(((p[dec] >= th) != y[dec]).sum())
        if d >= 20 and w <= 0.003 * d:
            return th, d, w
    return NEVER, 0, 0


def train_and_test(name: str, x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> tuple[TreeCheck, np.ndarray]:
    """Held-out packs (8 groups) set the bar, then the check is trained on everything. Also: the held-out P of each row."""
    y = np.asarray(y, dtype=bool)
    held = np.full(len(y), np.nan)
    for f in range(8):
        hold, tr = fold == f, fold != f
        if tr.sum() < 20 or y[tr].all() or not y[tr].any() or not hold.any():
            continue
        c = read(fit(name, x[tr], y[tr]).to_text())[name]
        held[hold] = c.prob(x[hold])
    has = ~np.isnan(held)
    final = read(fit(name, x, y).to_text())[name]
    final.sure = bar(held[has], y[has])[0]
    return final, held
