"""Paper figures as vector PDF, one panel per file, no inner title.
Palette: ordinal blue for the M1-M4 hierarchy; categorical (blue, orange, aqua, yellow)
for unordered identities; diverging blue-red with grey centre; sequential blue.
Requires figuras_bootstrap.npz (figuras_bootstrap.py), figuras_bootstrap_sementes_B10000.npz
(figuras_bootstrap_sementes.py) and final_fixo_B10000_v2.json (bootstrap_rodada1.py)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.ticker import FuncFormatter
from nucleo import RODADAS, CORPORA, PT, carregar, macro_f1, pred_prev, piso, auc

SAIDA = "../figures/"
M = list(RODADAS)
ROTM = {"M1": "M1 · Bag-of-Words", "M2": "M2 · BiLSTM", "M3": "M3 · mBERT", "M4": "M4 · XLM-RoBERTa"}
ROT2 = ["M1\nBag-of-Words", "M2\nBiLSTM", "M3\nmBERT", "M4\nXLM-RoBERTa"]
ORD = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]                 # ordinal blue ramp (M1 to M4)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]                 # validated categorical palette
TINTA, CINZA, GRADE = "#2b2b2b", "#8a8a85", "#e4e4e0"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#a3221f", "#e34948", "#f3b3a9", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])
K = len(CORPORA)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.edgecolor": CINZA, "axes.labelcolor": TINTA, "xtick.color": TINTA, "ytick.color": TINTA,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
virg = lambda v, d=2: f"{v:.{d}f}".replace("-", "−")
TICK = FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:.2f}".rstrip("0").rstrip(".").replace("-", "−"))

def salvar(fig, nome):
    fig.savefig(SAIDA + nome + ".pdf")
    plt.close(fig)

# ---------------------------------------------------------------- point estimates
CEL = {m: carregar(m) for m in M}
B_ = {t: piso(CEL["M1"][(t, t)][0]) for t in CORPORA}
def F(m, s, t, regime="prev"):
    y, P = CEL[m][(s, t)]
    return float(np.mean([macro_f1(y, pred_prev(p, y.mean()) if regime == "prev" else (p >= 0.5).astype(int)) for p in P]))
def A(m, s, t):
    y, P = CEL[m][(s, t)]
    return float(np.mean([auc(y, p) for p in P]))
FP = {m: np.array([[F(m, s, t) for t in CORPORA] for s in CORPORA]) for m in M}
FX = {m: np.array([[F(m, s, t, "fixo") for t in CORPORA] for s in CORPORA]) for m in M}
AU = {m: np.array([[A(m, s, t) for t in CORPORA] for s in CORPORA]) for m in M}
b = np.array([B_[t] for t in CORPORA])
def retencao(Fm):
    return (Fm - b[None, :]) / (np.diag(Fm) - b)[:, None]
iPT = [CORPORA.index(c) for c in PT]; lang = lambda c: "PT" if c in PT else c
GRUPO = {}
for i, s in enumerate(CORPORA):
    for j, t in enumerate(CORPORA):
        if i == j: GRUPO[(i, j)] = "D"
        elif "KZ" in (s, t): GRUPO[(i, j)] = "C"
        elif lang(s) == lang(t): GRUPO[(i, j)] = "A"
        else: GRUPO[(i, j)] = "B"

# ---------------------------------------------------------------- Group A, gain over the floor
pares = [(s, t) for s in PT for t in PT if s != t]
fig, ax = plt.subplots(figsize=(6.6, 2.7))
w = 0.2
for k, m in enumerate(M):
    g = [FP[m][CORPORA.index(s), CORPORA.index(t)] - B_[t] for s, t in pares]
    ax.bar(np.arange(6) + (k - 1.5) * w, g, w * 0.92, color=ORD[k], label=ROTM[m])
ax.set_xticks(range(6)); ax.set_xticklabels([f"{s}→{t}" for s, t in pares])
ax.set_ylabel("Gain over the trivial floor"); ax.yaxis.set_major_formatter(TICK)
ax.grid(axis="y", color=GRADE, lw=0.6); ax.set_axisbelow(True)
ax.legend(ncol=4, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.12))
fig.tight_layout(); salvar(fig, "fig2_grupoA")

# ---------------------------------------------------------------- macro-F1 and AUC matrices, one file per model
def heatmap(Mx, nome, cmap, norm, rot_cb, fmt=lambda v: virg(v), marca=None, tam=(3.3, 2.9)):
    fig, ax = plt.subplots(figsize=tam)
    im = ax.pcolormesh(np.arange(K + 1) - 0.5, np.arange(K + 1) - 0.5, Mx, cmap=cmap, norm=norm)
    ax.set_xlim(-0.5, K - 0.5); ax.set_ylim(K - 0.5, -0.5); ax.set_aspect('equal')
    for i in range(K):
        for j in range(K):
            v = Mx[i, j]; rgba = cmap(norm(v)); lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt = fmt(v) + ("*" if marca is not None and marca[i, j] else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.6, color="white" if lum < 0.5 else TINTA)
    ax.set_xticks(range(K)); ax.set_xticklabels(CORPORA); ax.set_yticks(range(K)); ax.set_yticklabels(CORPORA)
    ax.set_xlabel("Evaluation corpus"); ax.set_ylabel("Training corpus")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, K), minor=True); ax.set_yticks(np.arange(-.5, K), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0); ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    cb.ax.yaxis.set_major_formatter(TICK); cb.set_label(rot_cb, color=TINTA)
    fig.tight_layout(); salvar(fig, nome)

for m in M:
    heatmap(FP[m], f"fig3_macrof1_{m}", SEQ, Normalize(0.1, 1.0), "Macro-F1")
    heatmap(AU[m], f"fig4_auc_{m}", DIV, TwoSlopeNorm(0.5, 0.2, 1.0), "ROC-AUC (0.5 = chance)")

# ---------------------------------------------------------------- per-cell bootstrap
Z = np.load("figuras_bootstrap.npz")
FB = {m: Z[m] for m in M}
assert all(np.allclose(Z["ponto"][k], FP[m]) for k, m in enumerate(M))
def p_bil(x):
    return min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0)))

# ---------------------------------------------------------------- XLM-R minus mBERT, Holm
# paired test resampling the target indices and, independently per model,
# the three seeds; 10,000 replicates (figuras_bootstrap_sementes.py with B=10000)
dif = FP["M4"] - FP["M3"]
ZS = np.load("figuras_bootstrap_sementes_B10000.npz"); S3, S4 = ZS["M3"], ZS["M4"]; rs = np.random.default_rng(20261030)
nb = S3.shape[0]
D = np.stack([S4[r][..., rs.integers(0, 3, 3)].mean(-1) - S3[r][..., rs.integers(0, 3, 3)].mean(-1) for r in range(nb)])
pv = np.minimum(1, 2 * np.minimum((D <= 0).mean(0), (D >= 0).mean(0)))
ordem = np.argsort(pv.ravel()); n = pv.size; adj = np.empty(n); corre = 0.0
for r, k in enumerate(ordem):
    corre = max(corre, min(1.0, (n - r) * pv.ravel()[k])); adj[k] = corre
pholm = adj.reshape(K, K); holm = pholm <= 0.05
heatmap(dif, "fig5_xlmr_menos_mbert", DIV, TwoSlopeNorm(0, -0.2, 0.2), "Macro-F1 difference",
        fmt=lambda v: f"{v:+.3f}".replace("-", "−"), marca=holm, tam=(4.8, 3.7))
json.dump({"n_holm": int(holm.sum()), "celulas": [f"{CORPORA[i]}->{CORPORA[j]}" for i in range(K) for j in range(K) if holm[i, j]],
           "dif": {f"{CORPORA[i]}->{CORPORA[j]}": round(float(dif[i, j]), 3) for i in range(K) for j in range(K)},
           "p": {f"{CORPORA[i]}->{CORPORA[j]}": round(float(pv[i, j]), 4) for i in range(K) for j in range(K)},
           "p_holm": {f"{CORPORA[i]}->{CORPORA[j]}": round(float(pholm[i, j]), 3) for i in range(K) for j in range(K)}},
          open("figura_holm.json", "w"), indent=1)

# ---------------------------------------------------------------- retention by group
def ret_grupo(Fm, g):
    R = retencao(Fm)
    return np.mean([R[i, j] for (i, j), gg in GRUPO.items() if gg == g])
fig, ax = plt.subplots(figsize=(6.6, 2.8))
GR = [("A", "A · same language,\nother domain"), ("B", "B · other language,\nno Kazakh"), ("C", "C · pairs with\nKazakh")]
saida_grupos = {}
for k, m in enumerate(M):
    pts = [ret_grupo(FP[m], g) for g, _ in GR]
    bs = np.array([[ret_grupo(FB[m][r], g) for g, _ in GR] for r in range(FB[m].shape[0])])
    lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
    saida_grupos[m] = {g: [round(float(p), 3), round(float(l), 3), round(float(h), 3)] for (g, _), p, l, h in zip(GR, pts, lo, hi)}
    xs = np.arange(3) + (k - 1.5) * w
    ax.bar(xs, pts, w * 0.92, color=ORD[k], label=ROTM[m])
    ax.errorbar(xs, pts, yerr=[np.array(pts) - lo, hi - np.array(pts)], fmt="none", ecolor=TINTA, elinewidth=0.8, capsize=2)
json.dump(saida_grupos, open("figura_grupos.json", "w"), indent=1)
ax.axhline(0, color=CINZA, lw=0.8)
ax.set_xticks(range(3)); ax.set_xticklabels([r for _, r in GR])
ax.set_ylabel("Retention of the gain over the floor"); ax.yaxis.set_major_formatter(TICK)
ax.grid(axis="y", color=GRADE, lw=0.6); ax.set_axisbelow(True)
ax.legend(ncol=2, fontsize=7, loc="upper right")
fig.tight_layout(); salvar(fig, "fig7_grupos")

# ---------------------------------------------------------------- calibration x representation (M4)
cal = FP["M4"] - FX["M4"]
fig, ax = plt.subplots(figsize=(6.6, 3.3))
ESTILO = {"A": (CAT[0], "o", "Group A"), "B": (CAT[1], "s", "Group B"), "C": (CAT[2], "^", "Group C"), "D": (CAT[3], "D", "Diagonal")}
for g, (cor, mk, nome) in ESTILO.items():
    ij = [k for k, gg in GRUPO.items() if gg == g]
    ax.scatter([AU["M4"][i, j] for i, j in ij], [cal[i, j] for i, j in ij], s=26, c=cor, marker=mk,
               edgecolors="white", linewidths=0.8, label=nome, zorder=3)
    if g == "C":
        rotular = [(f"{CORPORA[i]}→{CORPORA[j]}", AU["M4"][i, j], cal[i, j]) for i, j in ij]
ax.axvline(0.5, color=CINZA, lw=0.8, ls="--"); ax.axhline(0, color=CINZA, lw=0.8, ls="--")
ax.set_xlabel("Cell ROC-AUC (quality of the transferred ranking)")
ax.set_ylabel("Calibration cost (macro-F1)")
ax.xaxis.set_major_formatter(TICK); ax.yaxis.set_major_formatter(TICK)
ax.grid(color=GRADE, lw=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=7, loc="upper right")
# Group C labels: for each one, the first position that covers no points or labels already placed
fig.tight_layout(); fig.canvas.draw(); rend = fig.canvas.get_renderer()
pts_px = ax.transData.transform(np.array([[AU["M4"][i, j], cal[i, j]] for (i, j) in GRUPO]))
caixas = []
for txt, xv, yv in rotular:
    for dx, dy, ha, va in ((5, 4, "left", "bottom"), (5, -4, "left", "top"), (-5, 4, "right", "bottom"),
                           (-5, -4, "right", "top"), (0, 7, "center", "bottom"), (0, -7, "center", "top")):
        an = ax.annotate(txt, (xv, yv), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=6, color=TINTA)
        bb = an.get_window_extent(rend).expanded(1.05, 1.15)
        pad = 4.5
        livre = (not any(bb.x0 - pad <= px <= bb.x1 + pad and bb.y0 - pad <= py <= bb.y1 + pad for px, py in pts_px)
                 and not any(bb.overlaps(c) for c in caixas)
                 and ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1))
        if livre:
            caixas.append(bb); break
        an.remove()
    else:
        caixas.append(ax.annotate(txt, (xv, yv), xytext=(5, 4), textcoords="offset points", fontsize=6, color=TINTA).get_window_extent(rend))
salvar(fig, "fig8_calibracao")

# ---------------------------------------------------------------- Portuguese sources, one file per criterion
FONTES = [("PT1", "PT1"), ("PT2", "PT2"), ("PT3", "PT3")]
def crit(m, s, c):
    i = CORPORA.index(s); alvos = [CORPORA.index(t) for t in PT if t != s]
    if c == "f1": return np.mean([FP[m][i, j] for j in alvos])
    if c == "ganho": return np.mean([FP[m][i, j] - b[j] for j in alvos])
    return np.mean([retencao(FP[m])[i, j] for j in alvos])
saida_fontes = {}
for c, rot, y0 in (("f1", "Macro-F1 (PT targets)", 0.45), ("ganho", "Gain over floor (PT targets)", 0.0),
                   ("ret", "Retention (PT targets)", 0.0)):
    fig, ax = plt.subplots(figsize=(2.5, 2.4)); w3 = 0.26
    for k, (s, nome) in enumerate(FONTES):
        v = [crit(m, s, c) for m in M]
        saida_fontes.setdefault(c, {})[s] = [round(float(x), 3) for x in v]
        ax.bar(np.arange(4) + (k - 1) * w3, v, w3 * 0.92, color=CAT[k], label=nome)
    if c == "ret": ax.axhline(1, color=CINZA, lw=0.8, ls="--")
    ax.set_ylim(bottom=y0)
    ax.set_xticks(range(4)); ax.set_xticklabels(M); ax.set_ylabel(rot); ax.yaxis.set_major_formatter(TICK)
    ax.grid(axis="y", color=GRADE, lw=0.6); ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5, handlelength=1, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); salvar(fig, f"fig9_fontes_{c}")
json.dump(saida_fontes, open("figura_fontes.json", "w"), indent=1)

# ---------------------------------------------------------------- costs under the fixed threshold
R = json.load(open("final_fixo_B10000_v2.json"))
fig, ax = plt.subplots(figsize=(3.6, 2.9)); x = range(4)
for k, cor, mk, ls, nome in (("lin", CAT[0], "o", "-", "Language, EN and ID targets"), ("lin_rev", CAT[2], "^", "--", "Language, PT targets"),
                              ("dom", CAT[1], "s", "-", "Domain, PT targets")):
    y = [R["ponto"][m][k] for m in M]; lo = [R["ic"][m][k][0] for m in M]; hi = [R["ic"][m][k][1] for m in M]
    ax.fill_between(x, lo, hi, color=cor, alpha=0.12, linewidth=0)
    ax.plot(x, y, color=cor, lw=2, ls=ls, marker=mk, ms=6, mec="white", mew=1.2, label=nome)
    ax.annotate(virg(y[3], 3), (3, y[3]), xytext=(7, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5, color=TINTA)
ax.set_xticks(list(x)); ax.set_xticklabels(ROT2, fontsize=7.5); ax.set_xlim(-0.25, 3.75)
ax.set_ylabel("Transfer cost (1 − retention)"); ax.yaxis.set_major_formatter(TICK)
ax.axhline(0, color=CINZA, lw=0.6, ls=":"); ax.grid(axis="y", color=GRADE, lw=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout(); salvar(fig, "fig10_limiar_fixo")
print("ok")
