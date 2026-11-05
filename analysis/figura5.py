"""Cost figure from the 10,000-replicate bootstrap (final_prev_B10000_v2.json):
(a) language and domain costs along the model hierarchy, with 95% CI;
(b) difference (language - domain) with 95% CI. Vector PDF, one panel per file."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open("final_prev_B10000_v2.json"))
M = ["M1", "M2", "M3", "M4"]
ROT = ["M1\nBag-of-Words", "M2\nBiLSTM", "M3\nmBERT", "M4\nXLM-RoBERTa"]
AZUL, LARANJA, AQUA, TINTA, CINZA = "#2a78d6", "#eb6834", "#1baf7a", "#2b2b2b", "#8a8a85"
virg = lambda v: f"{v:.3f}".replace(".", ",").replace("-", "−")
from matplotlib.ticker import FuncFormatter
TICK = FuncFormatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".").replace(".", ",").replace("-", "−") if v != 0 else "0")

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": CINZA,
                     "axes.labelcolor": TINTA, "xtick.color": TINTA, "ytick.color": TINTA,
                     "axes.spines.top": False, "axes.spines.right": False})
x = range(4)

# (a)
fig, ax = plt.subplots(figsize=(3.6, 2.9))
for k, cor, mk, ls, nome in (("lin", AZUL, "o", "-", "Língua, alvos EN e ID"), ("lin_rev", AQUA, "^", "--", "Língua, alvos PT"), ("dom", LARANJA, "s", "-", "Domínio, alvos PT")):
    y = [r["ponto"][m][k] for m in M]
    lo = [r["ic"][m][k][0] for m in M]; hi = [r["ic"][m][k][1] for m in M]
    ax.fill_between(x, lo, hi, color=cor, alpha=0.12, linewidth=0)
    ax.plot(x, y, color=cor, lw=2, ls=ls, marker=mk, ms=6, mec="white", mew=1.2, label=nome)
    ax.annotate(virg(y[3]), (3, y[3]), xytext=(7, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5, color=TINTA)
ax.set_xticks(list(x)); ax.set_xticklabels(ROT, fontsize=7.5)
ax.set_xlim(-0.25, 3.75)
ax.set_ylabel("Custo de transferência (1 − retenção)")
ax.axhline(0, color=CINZA, lw=0.6, ls=":")
ax.grid(axis="y", color="#e4e4e0", lw=0.6); ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(TICK)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout()
fig.savefig("../figures/fig6_custos_a.pdf")

# (b)
fig, ax = plt.subplots(figsize=(3.6, 2.9))
ax.axhline(0, color=CINZA, lw=1)
todos_lo, todos_hi = [], []
for k, pk, cor, mk, dx, nome in (("dif", "p_dif", AZUL, "o", -0.13, "Língua com alvos EN e ID"), ("dif_rev", "p_dif_rev", AQUA, "^", 0.13, "Língua com alvos PT")):
    y = [r["ponto"][m][k] for m in M]
    lo = [r["ic"][m][k][0] for m in M]; hi = [r["ic"][m][k][1] for m in M]
    todos_lo += lo; todos_hi += hi
    xs = [i + dx for i in x]
    ax.errorbar(xs, y, yerr=[[a - b for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)]], fmt=mk, color=cor,
                ms=6, mec="white", mew=1.2, elinewidth=1.6, capsize=3, label=nome)
    if k == "dif":
        p = r["ic"]["M4"][pk]
        ax.annotate(f"{virg(y[3])}\np = " + f"{p:.3f}".replace(".", ","), (xs[3], lo[3]), xytext=(0, -5), textcoords="offset points", ha="center", va="top", fontsize=7, color=TINTA)
ax.set_xticks(list(x)); ax.set_xticklabels(ROT, fontsize=7.5)
ax.set_xlim(-0.5, 3.5); ax.set_ylim(min(todos_lo) - 0.3, max(todos_hi) + 0.08)
ax.legend(frameon=False, fontsize=7, loc="upper right")
ax.set_ylabel("Custo de língua − custo de domínio")
ax.yaxis.set_major_formatter(TICK)
ax.grid(axis="y", color="#e4e4e0", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("../figures/fig6_custos_b.pdf")
print("ok")
