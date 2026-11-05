"""Per-cell bootstrap with macro-F1 per seed (M3 and M4) for the paired XLM-R vs mBERT
test, 10,000 replicates by default, resampling the seeds as well.
Writes figuras_bootstrap_sementes_B10000.npz."""
import os
import numpy as np
from nucleo import CORPORA, carregar, macro_f1, pred_prev

B = int(os.environ.get("B", 10000)); SEED = int(os.environ.get("SEMENTE", 20261030))
M = ["M3", "M4"]; K = len(CORPORA)
CEL = {m: carregar(m) for m in M}
N = {t: len(CEL["M3"][(t, t)][0]) for t in CORPORA}
rng = np.random.default_rng(SEED)
F = {m: np.empty((B, K, K, 3)) for m in M}
for r in range(B):
    idx = {t: rng.integers(0, N[t], N[t]) for t in CORPORA}
    for m in M:
        for i, s in enumerate(CORPORA):
            for j, t in enumerate(CORPORA):
                y, P = CEL[m][(s, t)]; ys = y[idx[t]]
                F[m][r, i, j] = [macro_f1(ys, pred_prev(p[idx[t]], ys.mean())) for p in P]
    if (r + 1) % 250 == 0:
        print("replicate", r + 1, flush=True)
np.savez_compressed(os.environ.get("SAIDA", "figuras_bootstrap_sementes_B10000.npz"), **F)
print("ok")
