"""Per-cell bootstrap (macro-F1, prevalence-matched threshold) for the per-cell and
per-group intervals. Indices are drawn once per target corpus and replicate, shared by all
cells of that target and by the four models. Writes F[model][replicate, source, target]
to figuras_bootstrap.npz."""
import os
import numpy as np
from nucleo import RODADAS, CORPORA, carregar, macro_f1, pred_prev

B = int(os.environ.get("B", 1500)); SEED = int(os.environ.get("SEMENTE", 20261030))
M = list(RODADAS); K = len(CORPORA)
CEL = {m: carregar(m) for m in M}
N = {t: len(CEL["M1"][(t, t)][0]) for t in CORPORA}

def F_cel(y, P, idx):
    ys = y[idx]
    return float(np.mean([macro_f1(ys, pred_prev(p[idx], ys.mean())) for p in P]))

def matriz(m, idx):
    out = np.empty((K, K))
    for i, s in enumerate(CORPORA):
        for j, t in enumerate(CORPORA):
            out[i, j] = F_cel(*CEL[m][(s, t)], idx[t])
    return out

rng = np.random.default_rng(SEED)
F = {m: np.empty((B, K, K)) for m in M}
for r in range(B):
    idx = {t: rng.integers(0, N[t], N[t]) for t in CORPORA}
    for m in M:
        F[m][r] = matriz(m, idx)
    if (r + 1) % 250 == 0:
        print("replicate", r + 1, flush=True)
Y = {t: CEL["M1"][(t, t)][0] for t in CORPORA}
np.savez_compressed("figuras_bootstrap.npz", **{m: F[m] for m in M},
                    ponto=np.stack([matriz(m, {t: np.arange(N[t]) for t in CORPORA}) for m in M]))
print("ok")
