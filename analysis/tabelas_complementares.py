"""Point estimates reported in the paper that are not produced by the bootstrap scripts:
diagonal table, retention by group, cells below the floor, calibration cost, ECE and its
correlation with the calibration cost, per-class performance (mean and per cell), translate-test/translate-train
table, alternative retention formulations, oracle gain and seed variance.
Writes tabelas_complementares.json and prints a summary."""
import json, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, ".")
from nucleo import RODADAS, CORPORA, PT, carregar, macro_f1, pred_prev, piso, auc

MODELOS = list(RODADAS)
CEL = {m: carregar(m) for m in MODELOS}
SEM_KZ = ["PT1", "PT2", "PT3", "EN", "ID"]
lang = lambda c: "PT" if c in PT else c


def grupo(s, t):
    if s == t: return "diag"
    if lang(s) == lang(t): return "A"
    return "C" if "KZ" in (s, t) else "B"


def f1(y, P, regime):
    out = []
    for p in P:
        if regime == "prev": pred = pred_prev(p, y.mean())
        elif regime == "fixo": pred = (p >= 0.5).astype(int)
        else:  # oracle: best threshold on the target itself (quantile grid, midpoints between unique values, 0.5)
            u = np.unique(p)
            meio = (u[:-1] + u[1:]) / 2 if len(u) < 5000 else np.array([])
            q = np.unique(np.concatenate([np.quantile(p, np.linspace(0.0005, 0.9995, 512)), meio, [0.5, u[0] - 1e-9]]))
            out.append(max(macro_f1(y, (p >= c).astype(int)) for c in q)); continue
        out.append(macro_f1(y, pred))
    return float(np.mean(out))


def ece(y, p, nbins=10):
    """Expected calibration error with equal-frequency bins, on the positive-class probability."""
    ordem = np.argsort(p); y, p = y[ordem], p[ordem]
    tot = 0.0
    for b in np.array_split(np.arange(len(p)), nbins):
        if len(b): tot += len(b) / len(p) * abs(y[b].mean() - p[b].mean())
    return float(tot)


def por_classe(y, P):
    r = []
    for p in P:
        pr = pred_prev(p, y.mean())
        tp = np.sum((pr == 1) & (y == 1)); fp = np.sum((pr == 1) & (y == 0)); fn = np.sum((pr == 0) & (y == 1)); tn = np.sum((pr == 0) & (y == 0))
        f1p = 2 * tp / (2 * tp + fp + fn) if tp else 0.0; f1n = 2 * tn / (2 * tn + fn + fp) if tn else 0.0
        r.append((f1p, f1n, tp / (tp + fn) if tp + fn else 0.0, tp / (tp + fp) if tp + fp else 0.0))
    return np.mean(r, axis=0)


def gama(R, corpora, excluir=()):
    cel = [(s, t) for s in corpora for t in corpora if s != t and (s, t) not in excluir]
    X = np.array([[1.0] + [1.0 * (s == c) for c in corpora[1:]] + [1.0 * (t == c) for c in corpora[1:]]
                  + [1.0 * (lang(s) != lang(t))] for s, t in cel])
    return float(np.linalg.lstsq(X, np.array([R[c] for c in cel]), rcond=None)[0][-1])


def gama_sens(R):
    return dict(principal=gama(R, SEM_KZ), com_kz=gama(R, CORPORA),
                sem_en_id=gama(R, SEM_KZ, [("EN", "ID")]), sem_id_en=gama(R, SEM_KZ, [("ID", "EN")]))


saida = {}
for m in MODELOS:
    C = {k: v for k, v in CEL[m].items() if k[0] in CORPORA and k[1] in CORPORA}
    b = {t: piso(C[(t, t)][0]) for t in CORPORA}
    Fp = {k: f1(*v, "prev") for k, v in C.items()}
    Ff = {k: f1(*v, "fixo") for k, v in C.items()}
    Fo = {k: f1(*v, "oraculo") for k, v in C.items()}
    A = {k: float(np.mean([auc(v[0], p) for p in v[1]])) for k, v in C.items()}
    sd = {k: float(np.std([macro_f1(v[0], pred_prev(p, v[0].mean())) for p in v[1]], ddof=1)) for k, v in C.items()}
    E = {k: float(np.mean([ece(v[0], p) for p in v[1]])) for k, v in C.items()}
    R = {(s, t): (Fp[(s, t)] - b[t]) / (Fp[(s, s)] - b[s]) for (s, t) in C}
    transf = [k for k in C if k[0] != k[1]]
    g = {k: grupo(*k) for k in C}
    med = lambda d, gr: float(np.mean([d[k] for k in C if g[k] == gr]))
    calib = {k: Fp[k] - Ff[k] for k in C}
    pc_transf = np.mean([por_classe(*C[k]) for k in transf], axis=0)
    pc_diag = {t: por_classe(*C[(t, t)]).tolist() for t in CORPORA}
    rho, p_rho = spearmanr([E[k] for k in transf], [calib[k] for k in transf])
    # alternative retention formulations; ratio = domain cost / language cost (EN and ID targets)
    def razao(Rf):
        dom = 1 - np.mean([Rf[(s, t)] for s in PT for t in PT if s != t])
        lin = 1 - np.mean([Rf[(s, t)] for s in PT for t in ("EN", "ID")])
        return float(dom / lin)
    RB = {(s, t): (Fp[(s, t)] - b[t]) / (Fp[(t, t)] - b[t]) for (s, t) in C}
    RC = {(s, t): Fp[(s, t)] / Fp[(s, s)] for (s, t) in C}
    RD = {(s, t): (A[(s, t)] - 0.5) / (A[(s, s)] - 0.5) for (s, t) in C}
    saida[m] = dict(
        diagonal={t: dict(macro_f1=Fp[(t, t)], dp=sd[(t, t)], auc=A[(t, t)]) for t in CORPORA},
        retencao_grupo={gr: med(R, gr) for gr in ("A", "B", "C")},
        abaixo_do_piso=int(sum(Fp[k] < b[k[1]] - 1e-6 for k in transf)),  # KZ->PT1 in M1 sits exactly on the floor
        custo_calibracao={gr: med(calib, gr) for gr in ("diag", "A", "B", "C")},
        ece={gr: med(E, gr) for gr in ("diag", "A", "B", "C")},
        spearman_ece_calibracao=dict(rho=float(rho), p=float(p_rho)),
        por_classe_transf=dict(f1_pos=pc_transf[0], f1_neg=pc_transf[1], rev_pos=pc_transf[2], prec_pos=pc_transf[3]),
        por_classe_diag=pc_diag,
        por_classe_celula={f"{s}->{t}": dict(zip(("f1_pos", "f1_neg", "rev_pos", "prec_pos"), por_classe(*C[(s, t)]).tolist())) for (s, t) in sorted(C)},
        por_classe_diag_media=dict(f1_pos=float(np.mean([v[0] for v in pc_diag.values()])), f1_neg=float(np.mean([v[1] for v in pc_diag.values()]))),
        razao_formulacoes=dict(A=razao(R), B=razao(RB), C=razao(RC), D=razao(RD)),
        ganho_oraculo=float(np.mean([Fo[k] - Fp[k] for k in C])),
        violacoes_oraculo=int(sum(Fo[k] < max(Fp[k], Ff[k]) - 1e-9 for k in C)),
        dp_medio_sementes=float(np.mean(list(sd.values()))),
        gama_sensibilidade=gama_sens(R),
    )

# translate-test (mt) and translate-train (tt): mean macro-F1 over cross-lingual cells
trad = {}
for m in ("M1", "M2"):
    C = CEL[m]
    for nome, alvos in (("sem_kz", SEM_KZ), ("com_kz", CORPORA)):
        cel = [(s, t) for s in alvos for t in alvos if lang(s) != lang(t)]
        if nome == "com_kz": cel = [c for c in cel if "KZ" in c]
        z = [f1(*C[c], "prev") for c in cel]
        mt = [f1(*C[(c[0] + "_mt", c[1])], "prev") for c in cel]
        tt = [f1(*C[(c[0] + "_tt", c[1])], "prev") for c in cel]
        trad[f"{m}_{nome}"] = dict(zero_shot=float(np.mean(z)), translate_test=float(np.mean(mt)),
                                   translate_train=float(np.mean(tt)), melhor_por_celula=float(np.mean(np.maximum(mt, tt))))
for m in ("M3", "M4"):
    for nome, alvos in (("sem_kz", SEM_KZ), ("com_kz", CORPORA)):
        cel = [(s, t) for s in alvos for t in alvos if lang(s) != lang(t)]
        if nome == "com_kz": cel = [c for c in cel if "KZ" in c]
        trad[f"{m}_{nome}"] = dict(zero_shot=float(np.mean([f1(*CEL[m][c], "prev") for c in cel])))
saida["traducao"] = trad

# uncertainty decomposition per cell: bootstrap over test indices (shared by the three seeds)
# against the standard deviation across seeds; total margin = 1.96 * sqrt(sd_boot^2 + sd_seed^2)
rng = np.random.default_rng(20261030); B_MARGEM = 1000
for m in MODELOS:
    C = {k: v for k, v in CEL[m].items() if k[0] in CORPORA and k[1] in CORPORA}
    idx = {t: [rng.integers(0, len(C[(t, t)][0]), len(C[(t, t)][0])) for _ in range(B_MARGEM)] for t in CORPORA}
    dom, mg, acima = 0, [], {}
    for (s, t), (y, P) in sorted(C.items()):
        sds = np.std([macro_f1(y, pred_prev(p, y.mean())) for p in P], ddof=1)
        sdb = np.std([np.mean([macro_f1(y[i], pred_prev(p[i], y[i].mean())) for p in P]) for i in idx[t]], ddof=1)
        dom += int(sds > sdb); g = 1.96 * np.sqrt(sdb ** 2 + sds ** 2); mg.append(g)
        if g > 0.08: acima[t] = acima.get(t, 0) + 1
    saida[m]["incerteza"] = dict(semente_domina=dom, margem_media=float(np.mean(mg)), celulas_margem_acima_008=acima)

json.dump(saida, open("tabelas_complementares.json", "w"), indent=1, default=float)
for m in MODELOS:
    s = saida[m]
    print(m, "diag", {t: round(v["macro_f1"], 3) for t, v in s["diagonal"].items()})
    print("   retention", {k: round(v, 3) for k, v in s["retencao_grupo"].items()}, "below the floor", s["abaixo_do_piso"])
    print("   calibration", {k: round(v, 3) for k, v in s["custo_calibracao"].items()}, "ECE", {k: round(v, 3) for k, v in s["ece"].items()})
    print("   rho ECE x calib", round(s["spearman_ece_calibracao"]["rho"], 3), "ratios", {k: round(v, 2) for k, v in s["razao_formulacoes"].items()})
    print("   gama", {k: round(v, 3) for k, v in s["gama_sensibilidade"].items()})
    print("   oracle", round(s["ganho_oraculo"], 3), "violations", s["violacoes_oraculo"], "mean sd", round(s["dp_medio_sementes"], 4))
for m in MODELOS: print(m, "incerteza", saida[m]["incerteza"])
print({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in trad.items()})
print("M1 per class (transfer)", {k: round(v, 3) for k, v in saida["M1"]["por_classe_transf"].items()})
