"""The base examples: the developer's sorted library and downloads, as the app's test bench picks them
(SampleManager --lab-cloudbase: SortingEval.CloudBase). One row per example of each check:
    check, pack (hash), label (1 / 0), source, path, fold (the bench's own), features (space between)
A first line "# scheme N" says which features they are (the app's Cloud.Scheme)."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from .train import Row


def load(path: str | Path, arrays: bool = False) -> tuple[int, list[Row]]:
    """(scheme, rows) of a base file (.tsv or .tsv.gz). arrays = each row's features as a numpy array (scheme 2's base
    is 4 million numbers: as Python floats it would take ~130 MB)."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    scheme = 0
    rows: list[Row] = []
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].split()
                if len(parts) >= 2 and parts[0] == "scheme":
                    scheme = int(parts[1])
                continue
            c = line.split("\t")
            if len(c) < 7:
                continue
            x = [float(t) for t in c[6].split(" ") if t]
            rows.append(Row(check=c[0], pack=c[1], label=c[2] == "1", source=c[3], path=c[4], fold=int(c[5]),
                            x=np.array(x) if arrays else x))
    return scheme, rows
