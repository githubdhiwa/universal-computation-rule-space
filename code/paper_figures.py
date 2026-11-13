"""Vector figures sized for a two-column paper (column = 3.4 in, full width = 7.0 in)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eca_classify import run

INK, INK2, MUTED, GRID, AXIS = "#1B2030", "#4A5060", "#6E7380", "#E4E3DE", "#9C9DA0"
C = {1: "#C9C8C2", 2: "#8E8D87", 3: "#E0662F", 4: "#2A6FC9"}
NAMES = {1: "Fixed (1)", 2: "Periodic (2)", 3: "Chaotic (3)", 4: "Complex (4)"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
                     "legend.frameon": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
lbl = pd.read_csv("eca_classes.csv", index_col=0)["cls_auto"].values.astype(int)

# ---------------- Fig 1: example spacetime diagrams (full width)
rng = np.random.default_rng(3)
x0 = rng.integers(0, 2, 120, dtype=np.uint8)
fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.95))
for ax, (r, c) in zip(axs, [(4, 2), (30, 3), (90, 3), (110, 4)]):
    ax.imshow(run(r, x0, 120), cmap=matplotlib.colors.ListedColormap(["#FFFFFF", INK]), interpolation="nearest")
    ax.set_title(f"({'abcd'[[4,30,90,110].index(r)]}) Rule {r}: {NAMES[c].split(' (')[0].lower()}", loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(GRID)
fig.tight_layout(pad=0.4); fig.savefig("figures/fig_examples.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------- Fig 2: rule-space map (full width)
fig, ax = plt.subplots(figsize=(7.0, 3.5))
pos = {}
for k in range(9):
    rules = [r for r in range(256) if bin(r).count("1") == k]
    for i, r in enumerate(rules):
        pos[r] = ((i - (len(rules) - 1) / 2), k)
for r in range(256):
    for b in range(8):
        s = r ^ (1 << b)
        if s > r:
            ax.plot(*zip(pos[r], pos[s]), color=GRID, lw=0.25, zorder=1)
for r in np.where(lbl == 4)[0]:
    for b in range(8):
        ax.plot(*zip(pos[r], pos[r ^ (1 << b)]), color=C[4], lw=0.8, alpha=0.7, zorder=2)
for c in (1, 2, 3, 4):
    rs = [r for r in range(256) if lbl[r] == c]
    ax.scatter([pos[r][0] for r in rs], [pos[r][1] for r in rs], s=34 if c == 4 else 11, c=C[c],
               edgecolors="white", linewidths=0.6, zorder=3, label=NAMES[c], marker="D" if c == 4 else "o")
for r in (110, 124, 137, 193, 30):
    ax.annotate(str(r), pos[r], xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.5, zorder=4)
ax.set_yticks(range(9)); ax.set_yticklabels([f"{k}/8" for k in range(9)])
ax.set_ylabel("Langton’s λ (share of 1s in the rule table)")
ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
ax.legend(loc="upper left", ncol=1, handletextpad=0.3)
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_rulespace.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------- Fig 3: neighbour composition of complex rules (column)
counts = np.bincount(lbl, minlength=5)[1:] / 256
cx = np.where(lbl == 4)[0]
nbf = np.bincount([lbl[r ^ (1 << k)] for r in cx for k in range(8)], minlength=5)[1:] / (8 * len(cx))
fig, ax = plt.subplots(figsize=(3.4, 1.9))
y = np.arange(4)
ax.barh(y + 0.19, counts, height=0.36, color="#C9C8C2", label="All 256 rules")
ax.barh(y - 0.19, nbf, height=0.36, color=C[4], label="Neighbours of complex rules (n = 32)")
for i in range(4):
    ax.text(counts[i] + 0.012, i + 0.19, f"{counts[i]:.1%}", va="center", fontsize=6.5, color=INK2)
    ax.text(nbf[i] + 0.012, i - 0.19, f"{nbf[i]:.1%}", va="center", fontsize=6.5, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([NAMES[c] for c in (1, 2, 3, 4)]); ax.invert_yaxis()
ax.set_xlim(0, 0.95)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc="lower right")
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_neighbours.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------- Fig 4: percolation (column)
d = pd.read_csv("percolation.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.fill_between(d.f, d.random_mean - 2 * d.random_sd, d.random_mean + 2 * d.random_sd, color="#C9C8C2", alpha=0.5, lw=0)
ax.plot(d.f, d.random_mean, color="#8E8D87", lw=1.4, label="Random order (mean ± 2 s.d., 200 orders)")
ax.plot(d.f, d.complexity_order, color=C[4], lw=1.5, label="Most irregular first (chaotic, then complex)")
ax.annotate("13 → 33 rules", (0.16, 0.129), xytext=(0.28, 0.05), fontsize=6.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6)
ax.set_xlabel("Fraction of the 256 rules switched on")
ax.set_ylabel("Largest connected cluster\n(fraction of all rules)")
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc="upper left")
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_percolation.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------- Fig 5: search application, share of runs finished after k tests (column)
raw = np.load("search_application_raw.npz")
style = [("random", "Random testing", "#8E8D87", "--"),
         ("lambda", "λ-guided (middle rows)", "#B07A2A", ":"),
         ("edge_chaos", "Edge-of-chaos walk", "#E0662F", "-"),
         ("combined", "Combined (λ, then chaos)", C[4], "-")]
fig, ax = plt.subplots(figsize=(3.4, 2.4))
k = np.arange(1, 257)
for key, lab, col, ls in style:
    x = np.sort(raw[key])
    ax.plot(k, np.searchsorted(x, k, side="right") / len(x), color=col, ls=ls, lw=1.4, label=lab)
ax.axvline(20, color=GRID, lw=0.8, zorder=0)
ax.set_xlim(0, 160); ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Rules tested (evaluations)")
ax.set_ylabel("Searches that found a\ncomplex rule")
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc="lower right")
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_search.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
print("figures done")
