"""Agreement between the two blind annotators of the translation validation."""
import csv
import numpy as np
from sklearn.metrics import cohen_kappa_score

ler = lambda f: {r["id"]: r["problematic"] for r in csv.DictReader(open(f))}
a1, a2 = ler("annotator1_answers.csv"), ler("annotator2_answers.csv")
ids = sorted(k for k in a1 if a1[k] and a2.get(k))
x, y = [a1[k] for k in ids], [a2[k] for k in ids]
rng = np.random.default_rng(20261030)
boot = []
for _ in range(5000):
    s = rng.integers(0, len(ids), len(ids))
    boot.append(cohen_kappa_score([x[i] for i in s], [y[i] for i in s]))
dec = [k for k in ids if "undecidable" not in (a1[k], a2[k])]
u1 = [v == "undecidable" for v in x]; u2 = [v == "undecidable" for v in y]
print(f"items answered by both: {len(ids)}")
print(f"raw agreement (3 categories): {np.mean([i == j for i, j in zip(x, y)]):.3f}")
print(f"Cohen's kappa (3 categories): {cohen_kappa_score(x, y):.3f}  95% CI {np.percentile(boot, [2.5, 97.5]).round(2)}")
print(f"items decided by both: {len(dec)}, agreement {np.mean([a1[k] == a2[k] for k in dec]):.3f}")
print(f"undecidable: annotator 1 {np.mean(u1):.3f}, annotator 2 {np.mean(u2):.3f}, kappa {cohen_kappa_score(u1, u2):.3f}")
