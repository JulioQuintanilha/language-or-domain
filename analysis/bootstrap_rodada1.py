"""Paired bootstrap for the transfer costs: domain cost, language-associated cost
(Portuguese source with English/Indonesian targets, and English/Indonesian source with
Portuguese targets), cost with Kazakh, their differences, the coefficient gamma and the
PT3<->EN control. Indices are drawn once per target corpus and replicate and shared by
all cells of that target and by the four models. The prevalence-matched threshold is
recomputed on each resample; the source diagonal comes from the same replicate."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from nucleo import RODADAS, CORPORA, PT, carregar, macro_f1, pred_prev, piso

import os
B = int(os.environ.get("B", 1500)); SEED = int(os.environ.get("SEMENTE", 20261030)); DIAG_FIXA = os.environ.get("DIAG_FIXA") == "1"; SAIDA = os.environ.get("SAIDA", "bootstrap_rodada1.json"); REGIME = os.environ.get("REGIME", "prev")
MODELOS = list(RODADAS)
CEL = {m: carregar(m) for m in MODELOS}
N = {t: len(CEL["M1"][(t, t)][0]) for t in CORPORA}

def F_amostra(y, P, idx, diagonal=False):
    if DIAG_FIXA and diagonal:
        idx = np.arange(len(y))
    ys = y[idx]
    return float(np.mean([macro_f1(ys, pred_prev(p[idx], ys.mean()) if REGIME == "prev" else (p[idx] >= 0.5).astype(int)) for p in P]))

def medidas(F, b):
    R = {(s, t): (F[(s, t)] - b[t]) / (F[(s, s)] - b[s]) for s in CORPORA for t in CORPORA}
    dom = 1 - np.mean([R[(s, t)] for s in PT for t in PT if s != t])
    lin = 1 - np.mean([R[(s, t)] for s in PT for t in ("EN", "ID")])
    lin_kz = 1 - np.mean([R[(s, t)] for s in PT for t in ("EN", "ID", "KZ")])
    sem_kz = ["PT1", "PT2", "PT3", "EN", "ID"]
    def lang(c): return "PT" if c in PT else c
    cross = [(s, t) for s in sem_kz for t in sem_kz if lang(s) != lang(t)]
    ctrl = [("PT3", "EN"), ("EN", "PT3")]
    demais = [c for c in cross if c not in ctrl]
    r_ctrl = np.mean([R[c] for c in ctrl]); r_dem = np.mean([R[c] for c in demais])
    # language cost with Portuguese targets (same targets as the domain cost)
    lin_rev = 1 - np.mean([R[(s, t)] for s in ("EN", "ID") for t in PT])
    # language-switch penalty with source and target fixed effects (20 cells without KZ)
    cel = [(s, t) for s in sem_kz for t in sem_kz if s != t]
    X = np.array([[1.0] + [1.0 * (s == c) for c in sem_kz[1:]] + [1.0 * (t == c) for c in sem_kz[1:]]
                  + [1.0 * (lang(s) != lang(t))] for s, t in cel])
    gama = float(np.linalg.lstsq(X, np.array([R[c] for c in cel]), rcond=None)[0][-1])
    return dict(dom=dom, lin=lin, lin_kz=lin_kz, dif=lin - dom, razao=dom / lin,
                ctrl=r_ctrl, demais=r_dem, ctrl_dif=r_ctrl - r_dem, n_demais=len(demais),
                lin_rev=lin_rev, dif_rev=lin_rev - dom, gama=gama)

def calcula(idx_por_alvo):
    out = {}
    for m in MODELOS:
        F = {k: F_amostra(*CEL[m][k], idx_por_alvo[k[1]], k[0] == k[1]) for k in CEL[m] if k[0] in CORPORA and k[1] in CORPORA}
        b = {t: piso(CEL[m][(t, t)][0][idx_por_alvo[t]]) for t in CORPORA}
        out[m] = medidas(F, b)
    return out

ponto = calcula({t: np.arange(N[t]) for t in CORPORA})
rng = np.random.default_rng(SEED)
reps = []
for i in range(B):
    reps.append(calcula({t: rng.integers(0, N[t], N[t]) for t in CORPORA}))
    if (i + 1) % 300 == 0: print("replicate", i + 1, flush=True)

def serie(m, k): return np.array([r[m][k] for r in reps])
def ic(x): return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
def p_bilateral(x): return float(min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0))))

res = {"B": B, "ponto": ponto, "ic": {}, "pares_dominio": {}, "pares_lingua": {}, "pares_lingua_rev": {}, "pares_gama": {}}
for m in MODELOS:
    res["ic"][m] = {k: ic(serie(m, k)) for k in ("dom", "lin", "lin_kz", "dif", "razao", "ctrl", "demais", "ctrl_dif", "lin_rev", "dif_rev", "gama")}
    res["ic"][m]["p_dif"] = p_bilateral(serie(m, "dif"))
    res["ic"][m]["p_ctrl_dif"] = p_bilateral(serie(m, "ctrl_dif"))
    res["ic"][m]["p_dif_rev"] = p_bilateral(serie(m, "dif_rev"))
    res["ic"][m]["p_gama"] = p_bilateral(serie(m, "gama"))
    res["ic"][m]["P_razao_maior_1"] = float(np.mean(serie(m, "razao") > 1))
    res["ic"][m]["mediana_ctrl_dif"] = float(np.median(serie(m, "ctrl_dif")))
for a, c in [("M1", "M2"), ("M1", "M3"), ("M1", "M4"), ("M2", "M3"), ("M2", "M4"), ("M3", "M4")]:
    for k, dest in (("dom", "pares_dominio"), ("lin", "pares_lingua"), ("lin_rev", "pares_lingua_rev"), ("gama", "pares_gama")):
        d = serie(c, k) - serie(a, k)
        res[dest][f"{c}-{a}"] = {"ponto": float(ponto[c][k] - ponto[a][k]), "ic": ic(d), "p": p_bilateral(d)}
for m in MODELOS:
    d = serie(m, "dif"); res["ic"][m]["dif_ic90"] = [float(np.percentile(d, 5)), float(np.percentile(d, 95))]; res["ic"][m]["p_unilateral"] = float(np.mean(d >= 0))
json.dump(res, open(SAIDA, "w"), indent=1, default=float)
print("ok")
