"""Transfer costs with the target prevalence estimated without labels by the EM
algorithm of Saerens, Latinne and Decaestecker (2002), from the model's own probabilities
and the source prevalence. Point estimates; writes em_saerens.json."""
import json
import numpy as np
from nucleo import RODADAS, CORPORA, PT, carregar, macro_f1, pred_prev, piso

def em_prev(p, pi_s, it=500, tol=1e-7):
    pi = pi_s
    for _ in range(it):
        a = pi / pi_s * p; b = (1 - pi) / (1 - pi_s) * (1 - p)
        novo = float(np.mean(a / (a + b)))
        if abs(novo - pi) < tol:
            break
        pi = novo
    return pi

out = {}
for m in RODADAS:
    C = carregar(m); F = {}; prev_est = {}
    for (s, t), (y, P) in C.items():
        if s not in CORPORA or t not in CORPORA:
            continue
        pi_s = C[(s, s)][0].mean(); v = []; est = []
        for p in P:
            pr = pi_s if s == t else em_prev(p, pi_s)
            est.append(pr); v.append(macro_f1(y, pred_prev(p, pr)))
        F[(s, t)] = float(np.mean(v)); prev_est[f"{s}->{t}"] = [float(np.mean(est)), float(y.mean())]
    b = {t: piso(C[(t, t)][0]) for t in CORPORA}
    R = {(s, t): (F[(s, t)] - b[t]) / (F[(s, s)] - b[s]) for s in CORPORA for t in CORPORA}
    lang = lambda c: "PT" if c in PT else c
    grupo = lambda s_, t_: "C" if "KZ" in (s_, t_) else ("A" if lang(s_) == lang(t_) else "B")
    erros = {"A": [], "B": [], "C": []}
    for k, (a_, b2) in prev_est.items():
        s_, t_ = k.split("->")
        if s_ != t_:
            erros[grupo(s_, t_)].append(abs(a_ - b2))
    out[m] = {"prev_estimada_e_real_por_celula": prev_est, "erro_prev_grupo": {g: float(np.mean(v)) for g, v in erros.items()}, "dom": 1 - np.mean([R[(s, t)] for s in PT for t in PT if s != t]),
              "lin": 1 - np.mean([R[(s, t)] for s in PT for t in ("EN", "ID")]),
              "lin_rev": 1 - np.mean([R[(s, t)] for s in ("EN", "ID") for t in PT]),
              "erro_prev_medio": float(np.mean([abs(a - b_) for k, (a, b_) in prev_est.items() if k.split("->")[0] != k.split("->")[1]]))}
json.dump(out, open("em_saerens.json", "w"), indent=1, default=float)
for m, v in out.items():
    print(m, {k: (round(x, 3) if not isinstance(x, dict) else {g: round(y, 3) for g, y in x.items()}) for k, x in v.items() if k != "prev_estimada_e_real_por_celula"})
