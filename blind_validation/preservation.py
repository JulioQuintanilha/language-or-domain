"""Blind validation of translations: does the translated text still carry the gold label?
Preservation among decided items (Wilson 95% CI), conservative reading (undecidable = failure),
by class, Fisher tests, undecidability by class, Kazakh directions against the others, and
agreement with the automatic triage. Run for each annotator."""
import csv, sys
import numpy as np
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score

key = {r["id"]: r for r in csv.DictReader(open("key.csv"))}


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m), min(1.0, c + m)


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.2f} [{lo:.2f}-{hi:.2f}]"


def analisar(arquivo):
    resp = {r["id"]: r["problematic"] for r in csv.DictReader(open(arquivo)) if r["problematic"]}
    ids = sorted(resp)
    ouro = {i: int(key[i]["gold_label"]) for i in ids}
    dec = [i for i in ids if resp[i] != "undecidable"]
    acerto = {i: int((resp[i] == "yes") == (ouro[i] == 1)) for i in dec}
    out = {"answered": len(ids), "undecidable": 1 - len(dec) / len(ids)}
    print(f"\n=== {arquivo}: {len(ids)} items answered, {len(ids) - len(dec)} undecidable ({1 - len(dec) / len(ids):.2f})")
    k = sum(acerto.values())
    print("  preservation, decided items:      ", fmt(k, len(dec)))
    print("  preservation, undecidable = fail: ", fmt(k, len(ids)))
    tab = {}
    for v, nome in ((1, "positive"), (0, "negative")):
        d = [i for i in dec if ouro[i] == v]; a = [i for i in ids if ouro[i] == v]
        kk = sum(acerto[i] for i in d)
        tab[v] = (kk, len(d), len(a))
        print(f"  {nome}: decided {fmt(kk, len(d))}   conservative {fmt(kk, len(a))}")
    (kp, dp, ap), (kn, dn, an) = tab[1], tab[0]
    o1, p1 = fisher_exact([[kp, dp - kp], [kn, dn - kn]])
    o2, p2 = fisher_exact([[kp, ap - kp], [kn, an - kn]])
    print(f"  Fisher, decided: OR={o1:.3f} p={p1:.1e}   conservative: OR={o2:.3f} p={p2:.1e}")
    up = sum(resp[i] == "undecidable" for i in ids if ouro[i] == 1); un = sum(resp[i] == "undecidable" for i in ids if ouro[i] == 0)
    o3, p3 = fisher_exact([[up, ap - up], [un, an - un]])
    print(f"  undecidable: positives {up}/{ap} = {up / ap:.2f}, negatives {un}/{an} = {un / an:.2f}, OR={o3:.2f} p={p3:.3f}")
    kz = [i for i in ids if key[i]["direction"].startswith("KZ")]; ot = [i for i in ids if not key[i]["direction"].startswith("KZ")]
    for nome, s in (("Kazakh", kz), ("others", ot)):
        d = [i for i in s if i in acerto]; kk = sum(acerto[i] for i in d)
        print(f"  {nome}: decided {fmt(kk, len(d))}  undecidable {1 - len(d) / len(s):.2f}  conservative {fmt(kk, len(s))}")
    ukz = sum(resp[i] == "undecidable" for i in kz); uot = sum(resp[i] == "undecidable" for i in ot)
    print(f"  Fisher on undecidable rate, Kazakh vs others: p={fisher_exact([[ukz, len(kz) - ukz], [uot, len(ot) - uot]])[1]:.2f}")
    print("  by direction (decided preservation, undecidable rate):")
    for dr in sorted({key[i]["direction"] for i in ids}):
        s = [i for i in ids if key[i]["direction"] == dr]; d = [i for i in s if i in acerto]
        print(f"    {dr:8} n={len(s):2}  undecidable={1 - len(d) / len(s):.2f}  {fmt(sum(acerto[i] for i in d), len(d)) if d else '-'}")
    b1 = [i for i in dec if key[i]["block"] == "1" and key[i]["automatic_triage"]]
    h = [acerto[i] for i in b1]
    for nome, pos in (("partial = preserved", {"preserved", "partial"}), ("partial = lost", {"preserved"})):
        a = [int(key[i]["automatic_triage"] in pos) for i in b1]
        print(f"  vs automatic triage ({nome}): n={len(b1)} agreement={np.mean([x == y for x, y in zip(h, a)]):.2f} kappa={cohen_kappa_score(h, a):.2f}")
    a = [int(key[i]["automatic_triage"] == "preserved") for i in b1]
    print(f"  triage lost / human preserved: {sum(1 for x, y in zip(h, a) if x == 1 and y == 0)}; reverse: {sum(1 for x, y in zip(h, a) if x == 0 and y == 1)}")


for f in sys.argv[1:] or ["annotator1_answers.csv", "annotator2_answers.csv"]:
    analisar(f)
