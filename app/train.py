"""The sorter's checks, trained on the base examples + everyone's shared answers.

A port of Sample Manager's own trainer (src/App/Sorter.cs `Check`, src/App/SortingEval.cs `TrainAndTest`): each check
is a logistic regression on standardised features (400 steps of gradient descent, both answers weighing the same, a
little L2). Its "sure" bar is set on packs held out in 8 groups: the lowest confidence at which it was wrong at most
0.3 % of the times it decided by itself (the app's "no wrong moves" rule). The app picks the examples and their features
(SortingEval.Rows); this file only trains on them, so the two can never pick differently. One difference: a pack's
group comes from its hash (folds_of), so packs that join don't move the others to other groups.

The model text is the app's sortmodel.txt format (SortModel.Read):
    # <about>
    check <name> <d>
    mean <d numbers>
    sd <d numbers>
    w <d numbers> <bias>
    sure <number>
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np

THRESHOLDS = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]
NEVER = 1.01  # a "sure" bar above 1: the check only suggests, it never decides by itself


@dataclass
class Row:
    """One example of one check: its pack (a hash: the same pack groups the same way everywhere), its answer, its features."""

    check: str
    pack: str
    label: bool
    x: list[float]
    source: str = "cloud"
    path: str = ""
    fold: int = -1  # the base's own fold (only for checking that this trains as the app does)


@dataclass
class Check:
    name: str
    mean: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    bias: float
    sure: float = NEVER

    def prob(self, x: np.ndarray) -> np.ndarray:
        """P(yes) for each row of x."""
        z = self.bias + ((np.asarray(x, dtype=float) - self.mean) / self.sd) @ self.w
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + np.exp(-z))

    def to_text(self) -> str:
        def nums(v) -> str:
            return " ".join(_num(float(t)) for t in v)

        return (f"check {self.name} {len(self.w)}\nmean {nums(self.mean)}\nsd {nums(self.sd)}\n"
                f"w {nums(self.w)} {_num(self.bias)}\nsure {_num(self.sure)}\n")


def _num(v: float) -> str:
    """A number as C# reads it back exactly (shortest round-trip; never inf / nan)."""
    if not math.isfinite(v):
        raise ValueError("a check's number is not finite")
    return repr(float(v))


def train_check(name: str, x: np.ndarray, y: np.ndarray) -> Check:
    """Sorter.cs Check.Train: both answers weigh the same however many examples each has."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=bool)
    n, d = x.shape
    mean = x.mean(axis=0)
    sd = np.maximum(1e-6, np.sqrt(((x - mean) ** 2).mean(axis=0)))
    pos = int(y.sum())
    neg = n - pos
    weight = np.where(y, n / (2 * max(1, pos)), n / (2 * max(1, neg)))
    z0 = (x - mean) / sd
    target = y.astype(float)
    w = np.zeros(d)
    bias = 0.0
    rate, l2 = 0.5, 0.002
    with np.errstate(over="ignore"):
        for _ in range(400):
            p = 1.0 / (1.0 + np.exp(-(bias + z0 @ w)))
            err = (p - target) * weight
            g = z0.T @ err
            w = w - rate * (g / n + l2 * w)
            bias -= rate * float(err.sum()) / n
    return Check(name, mean, sd, w, bias)


def folds_of(rows: list[Row], use_base_folds: bool = False) -> np.ndarray:
    """
    Which of the 8 groups each row is held out in: by its pack, the pack's own group (from its hash). The app numbers
    the packs in order instead; here a new pack must not move every other pack to another group (the "sure" bars would
    wobble from night to night with no new knowledge). use_base_folds = the app's own groups (the port's test).
    """
    if use_base_folds:
        return np.array([r.fold for r in rows])
    return np.array([_group(r.pack) for r in rows])


def _group(pack: str) -> int:
    try:
        return int(pack[:8], 16) % 8
    except ValueError:
        return zlib.crc32(pack.encode("utf-8")) % 8


def train_and_test(name: str, rows: list[Row], report: list[str], use_base_folds: bool = False) -> Check | None:
    """SortingEval.cs TrainAndTest: held-out packs set the "sure" bar, then the check is trained on everything."""
    y = np.array([r.label for r in rows], dtype=bool)
    if len(rows) < 40 or y.all() or not y.any():
        report.append(f"   {name}: too few examples ({len(rows)})")
        return None
    x = np.array([r.x for r in rows], dtype=float)
    fold = folds_of(rows, use_base_folds)
    held = np.full(len(rows), np.nan)
    for f in range(8):
        hold = fold == f
        train = ~hold
        if train.sum() < 20 or y[train].all() or not y[train].any():
            continue
        c = train_check(name, x[train], y[train])
        if hold.any():
            held[hold] = c.prob(x[hold])
    has = ~np.isnan(held)
    hp, hy = held[has], y[has]
    chosen = NEVER
    line = []
    for th in THRESHOLDS:
        dec = (hp >= th) | (hp <= 1 - th)
        decided = int(dec.sum())
        wrong = int(((hp[dec] >= th) != hy[dec]).sum())
        line.append(f"  {th:g}: {100.0 * decided / max(1, int(has.sum())):.0f}% / {wrong}")
        if chosen > 1 and decided >= 20 and wrong <= 0.003 * decided:
            chosen = th
    report.append(f"   {name:<22} {int(y.sum()):5} yes / {int((~y).sum()):5} no | decides on its own at "
                  f"{'never' if chosen > 1 else f'{chosen:g}'} | (threshold: % decided / wrong){''.join(line)}")
    # the most confident "mistakes" (often the label is what's wrong: worth a look)
    idx = np.where(has)[0]
    miss = [i for i in idx if (held[i] >= 0.5) != y[i]]
    miss.sort(key=lambda i: -abs(held[i] - 0.5))
    for i in miss[:8]:
        r = rows[i]
        report.append(f"        sure but wrong: P {held[i]:.4f} labelled {'yes' if r.label else 'no'} ({r.source}) {r.path}")
    final = train_check(name, x, y)
    final.sure = chosen
    return final


def train_model(rows: list[Row], about: str, report: list[str], use_base_folds: bool = False) -> tuple[str, dict[str, Check]]:
    """Every check (in the order the rows name them first), as the model's text, and the checks themselves."""
    by_check: dict[str, list[Row]] = {}
    for r in rows:
        by_check.setdefault(r.check, []).append(r)
    report.append("== the checks: packs held out in 8 groups (how many each decides on its own, and how many of those are wrong)")
    checks: dict[str, Check] = {}
    text = [f"# {about}\n"]
    for name, ex in by_check.items():
        c = train_and_test(name, ex, report, use_base_folds)
        if c is not None:
            checks[name] = c
            text.append(c.to_text())
    return "".join(text), checks


def read_model(text: str) -> dict[str, Check]:
    """SortModel.Read: a model's checks by name."""
    lines = text.replace("\r", "").split("\n")
    checks: dict[str, Check] = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("check ") or i + 3 >= len(lines):
            i += 1
            continue
        name = lines[i].split(" ")[1]

        def nums(s: str) -> np.ndarray:
            return np.array([float(t) for t in s.split(" ")[1:] if t])

        w = nums(lines[i + 3])
        c = Check(name, nums(lines[i + 1]), nums(lines[i + 2]), w[:-1], float(w[-1]))
        if i + 4 < len(lines) and lines[i + 4].startswith("sure "):
            c.sure = float(lines[i + 4].split(" ")[1])
            i += 1
        checks[name] = c
        i += 4
    return checks


@dataclass
class Score:
    decided: int = 0
    wrong: int = 0
    rows: int = 0


def score(check: Check | None, rows: list[Row]) -> Score:
    """How many rows a check decides by itself (at its own "sure" bar), and how many of those wrongly."""
    s = Score(rows=len(rows))
    if check is None or not rows or check.sure > 1:
        return s
    x = np.array([r.x for r in rows], dtype=float)
    y = np.array([r.label for r in rows], dtype=bool)
    p = check.prob(x)
    dec = (p >= check.sure) | (p <= 1 - check.sure)
    s.decided = int(dec.sum())
    s.wrong = int(((p[dec] >= check.sure) != y[dec]).sum())
    return s


@dataclass
class Gate:
    passed: bool
    checks: dict[str, dict] = field(default_factory=dict)
    why: list[str] = field(default_factory=list)


def gate(candidate: dict[str, Check], current: dict[str, Check], golden: list[Row]) -> Gate:
    """
    May the candidate replace the current model? Only if every check the current one has is there, and on the golden
    examples (the developer's base: answers can't change what counts as right) no check of it decides more of them
    wrongly by itself than the current one does (never more wrong moves than before), nor gives up deciding: under half
    as many decided by itself as now (shared answers made it unsure: bad answers, or too few yet).
    """
    by_check: dict[str, list[Row]] = {}
    for r in golden:
        by_check.setdefault(r.check, []).append(r)
    g = Gate(passed=True)
    for name in sorted(set(current) | set(candidate)):
        rows = by_check.get(name, [])
        now, new = score(current.get(name), rows), score(candidate.get(name), rows)
        g.checks[name] = {"rows": len(rows), "current": {"decided": now.decided, "wrong": now.wrong},
                          "candidate": {"decided": new.decided, "wrong": new.wrong}}
        if name in current and name not in candidate:
            g.passed = False
            g.why.append(f"{name}: missing (too few examples?)")
        elif new.wrong > now.wrong:
            g.passed = False
            g.why.append(f"{name}: {new.wrong} wrong decisions on the golden examples (now {now.wrong})")
        elif now.decided > 0 and new.decided * 2 < now.decided:
            g.passed = False
            g.why.append(f"{name}: decides {new.decided} by itself (now {now.decided}): much less")
    return g
