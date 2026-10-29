"""(research, 2026-10-29; HANDOVER 1zq) Two --lab-policy2 reports compared sound by sound: what B newly sorts that A
asked about (and how often the bench's truth agrees), what B asks about that A sorted, and where both sorted but
differently.   python research/compare_policy.py policy-A.txt policy-B.txt"""
import sys
from collections import Counter


def each(path):
    out, on = {}, False
    for line in open(path, encoding="utf-8"):
        if line.startswith("== each"):
            on = True
            continue
        if on and line.strip():
            src, truth, got, why, p = line.rstrip("\n").split("\t")
            out[(src, p)] = (truth, got, why)
    return out


def coarse(k):
    return "loop" if k.startswith("Loop") or k in ("Fill", "Stem") else "fx" if k.startswith("Fx") else "hit"



def main():
    a, b = each(sys.argv[1]), each(sys.argv[2])
    for src in sorted({s for s, _ in a}):
        keys = [k for k in a if k[0] == src and k in b]
        newly = [k for k in keys if a[k][1] == "ASK" and b[k][1] != "ASK"]
        lost = [k for k in keys if a[k][1] != "ASK" and b[k][1] == "ASK"]
        changed = [k for k in keys if "ASK" not in (a[k][1], b[k][1]) and a[k][1] != b[k][1]]
        right = sum(b[k][1] == b[k][0] for k in newly)
        print(f"== {src}: {len(keys)} sounds")
        print(f"   B sorts {len(newly)} that A asked about: {right} as the bench's truth says ({100 * right / max(1, len(newly)):.1f}%)")
        for (t, g), n in Counter((b[k][0], b[k][1]) for k in newly if b[k][1] != b[k][0]).most_common(8):
            print(f"        {n:4}  truth {t:<12} B sorted {g}")
        print(f"   B asks about {len(lost)} that A sorted ({sum(a[k][1] == a[k][0] for k in lost)} of them A had right)")
        for w, n in Counter(b[k][2] for k in lost).most_common(5):
            print(f"        {n:4}  {w}")
        print(f"   both sorted, differently: {len(changed)} (A right {sum(a[k][1] == a[k][0] for k in changed)}, B right {sum(b[k][1] == b[k][0] for k in changed)})")
        for k in changed[:6]:
            print(f"        truth {a[k][0]:<12} A {a[k][1]:<12} B {b[k][1]:<12} {k[1][-70:]}")


if __name__ == "__main__":
    main()
