# -*- coding: utf-8 -*-
"""Figure for Proposition 3: share of a pair's optimal contribution lost when the
model prices with elasticity beta_hat instead of the true beta (exact formula
and second-order approximation). Writes figs/fig_theory_regret.png."""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts in PDF figures
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

V2 = Path(os.environ.get("KT_PAPER_DIR", str(Path(__file__).resolve().parents[1] / "results" / "paper")))
(V2 / "figs").mkdir(parents=True, exist_ok=True)   # output folder of the tables, figures, and text
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False})


def loss(bh, b):
    r = bh * (b - 1) / (b * (bh - 1))
    return 1 - r ** (-b) * (1 + b * (r - 1))


fig, ax = plt.subplots(figsize=(3.5, 2.75))
out = {}
handles, labels = [], []
for b, c in ((1.5, "#d94801"), (2.0, "#2171b5"), (3.0, "#238b45")):
    bh = np.linspace(1.02, 5.0, 600)
    ax.plot(bh - b, 100 * loss(bh, b), color=c, lw=1.4)
    ax.plot(bh - b, 100 * (bh - b) ** 2 / (2 * b * (b - 1)), color=c, lw=0.9, ls=":")
    handles.append(Line2D([], [], color=c, lw=1.4))
    labels.append(fr"$\beta={b:g}$")
    out[str(b)] = {f"{d:+.1f}": float(100 * loss(b + d, b)) for d in (-0.3, 0.3, 0.5, 1.0) if b + d > 1}
handles += [Line2D([], [], color="0.3", lw=1.4), Line2D([], [], color="0.3", lw=0.9, ls=":")]
labels += ["exact", "second order"]
ax.set_xlim(-1.0, 1.5)
ax.set_ylim(0, 40)
ax.set_xlabel(r"Elasticity error $\hat\beta-\beta$")
ax.set_ylabel("Contribution lost (%)")
# legend above the axes, so that it covers no curve
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=3, frameon=False,
           fontsize=7.2, columnspacing=1.0, handlelength=2.0, handletextpad=0.4)
fig.tight_layout(rect=(0, 0, 1, 0.84))
(V2 / "figs").mkdir(exist_ok=True)
fig.savefig(V2 / "figs" / "fig_theory_regret.png", dpi=300, bbox_inches="tight")
fig.savefig(V2 / "figs" / "fig_theory_regret.pdf", bbox_inches="tight")  # vector copy for the journal artwork
(Path(__file__).resolve().parents[1] / "results" / "known_truth" / "summary" / "theory_regret.json").write_text(
    json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
