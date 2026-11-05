"""Shared core of the re-analyses: reading predictions, prevalence-matched threshold,
macro-F1, trivial floor, retention and costs."""
import glob, os, re
import numpy as np

BASE = os.environ.get("PIBIC_RES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "predictions"))
RODADAS = {  # runs used in the paper
    "M1": "predicoes_M1_20260806_0117",
    "M2": "predicoes_M2_20260806_0646",
    "M3": "predicoes_M3_20260806_0327",
    "M4": "predicoes_M4_20260806_0416",
}
# 19 Aug runs: repeat the matrix and add the ablations (EN_ESTRITO, PT1_noacc)
RODADAS_ABLACAO = {"M3": "predicoes_M3_20260819_1634", "M4": "predicoes_M4_20260819_2103"}
CORPORA = ["PT1", "PT2", "PT3", "EN", "ID", "KZ"]
PT = ["PT1", "PT2", "PT3"]
SEEDS = [42, 43, 44]


def carregar(modelo, rodada=None):
    """dict[(source, target)] -> (y, probs[n_seeds, n])"""
    d = os.path.join(BASE, rodada or RODADAS[modelo])
    out = {}
    for f in glob.glob(os.path.join(d, "*.npz")):
        m = re.match(r"(.+)_to_(.+)_s(\d+)\.npz$", os.path.basename(f))
        s, t, seed = m.group(1), m.group(2), int(m.group(3))
        z = np.load(f, allow_pickle=False)
        out.setdefault((s, t), {})[seed] = (z["y"].astype(int), z["probs"].astype(float))
    cel = {}
    for k, v in out.items():
        ys = [v[s][0] for s in sorted(v)]
        assert all(np.array_equal(ys[0], y) for y in ys), f"rótulos diferentes entre sementes em {k}"
        cel[k] = (ys[0], np.stack([v[s][1] for s in sorted(v)]))
    return cel


def macro_f1(y, pred):
    f = []
    for c in (1, 0):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def pred_prev(p, prev):
    """Threshold at the quantile that matches the predicted positive rate to the target prevalence."""
    k = int(round(prev * len(p)))
    if k <= 0:
        return np.zeros_like(p, dtype=int)
    if k >= len(p):
        return np.ones_like(p, dtype=int)
    thr = np.sort(p)[::-1][k - 1]
    return (p >= thr).astype(int)


def f1_celula(y, P, regime="prev"):
    """Macro-F1 averaged over the seeds."""
    vals = []
    for p in P:
        pred = pred_prev(p, y.mean()) if regime == "prev" else (p >= 0.5).astype(int)
        vals.append(macro_f1(y, pred))
    return float(np.mean(vals))


def piso(y):
    q = max(y.mean(), 1 - y.mean())
    return q / (1 + q)


def custos(F, b):
    """F[(s,t)] and b[t] -> (domain cost, language cost without KZ, with KZ)."""
    R = lambda s, t: (F[(s, t)] - b[t]) / (F[(s, s)] - b[s])
    dom = [R(s, t) for s in PT for t in PT if s != t]
    lin = [R(s, t) for s in PT for t in ("EN", "ID")]
    lin_kz = [R(s, t) for s in PT for t in ("EN", "ID", "KZ")]
    return 1 - np.mean(dom), 1 - np.mean(lin), 1 - np.mean(lin_kz)


def auc(y, p):
    """ROC-AUC from ranks (Mann-Whitney), ties at the mean rank."""
    from scipy.stats import rankdata
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
