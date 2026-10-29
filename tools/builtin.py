"""The app's built-in sorter for scheme 2: the same trainer the server runs every night (app/trees.py train_model), on
the base alone (data/base2.tsv.gz: the developer's sorted library + downloads, made by SampleManagerDev --lab-strongbase).
It writes the model and each check's held-out score on the base (what the server compares its first candidate with):

    python tools/builtin.py data/base2.tsv.gz data/builtin2.txt data/builtin2-heldout.json [report.txt]
    copy data\\builtin2.txt ..\\src\\sortmodel.txt                    (then build.bat: it is built into the exe)

(from cloud/; needs scikit-learn; ~5 minutes). With --small: a fifth of the packs (the tests' base, tests/fixtures).
"""
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from app import base, trees  # noqa: E402


def small(base_file: str, out_file: str) -> None:
    """The packs whose hash is a multiple of 5 (a fifth of them, each whole), as a base file of their own."""
    opener = gzip.open if base_file.endswith(".gz") else open
    with opener(base_file, "rt", encoding="utf-8") as f, gzip.open(out_file, "wt", encoding="utf-8", newline="\n") as out:
        for line in f:
            c = line.split("\t")
            if line.startswith("#") or (len(c) > 1 and int(c[1][:8], 16) % 5 == 0):
                out.write(line)


def main(base_file: str, model_out: str, heldout_out: str, report_file: str | None = None) -> None:
    t0 = time.time()
    scheme, rows = base.load(base_file, arrays=True)
    if scheme != 2:
        raise SystemExit(f"{base_file}: scheme {scheme}, not a scheme 2 base (--lab-strongbase)")
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    about = f"built in: trained {when} UTC on {len(rows)} base examples (the developer's library + downloads), scheme 2"
    report: list[str] = [about]
    text, checks, heldout = trees.train_model(rows, about, report)
    Path(model_out).write_text(text, encoding="utf-8", newline="\n")
    Path(heldout_out).write_text(json.dumps(heldout, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    report.append(f"{len(checks)} checks, {len(text) / 1e6:.2f} MB, {time.time() - t0:.0f} s")
    if report_file:
        Path(report_file).write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(line for line in report if "sure but wrong" not in line))


if __name__ == "__main__":
    if sys.argv[1] == "--small":
        small(sys.argv[2], sys.argv[3])
        main(sys.argv[3], *sys.argv[4:])
    else:
        main(*sys.argv[1:])
