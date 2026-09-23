"""
Manuscript-ready dataset figures for the SIDN-CRO benchmark suite.

Instead of a single dense 4x4 grid that does not fit a journal page, this
module produces four focused figures, each one-concept-per-figure and each
sized for direct drop-in to an Elsevier / Springer / IEEE manuscript:

    fig02_coverage_timeline.{pdf,png}      double-column,  7.0 x 2.4 in
        Gantt-style temporal coverage of the four panels with shaded
        Train / Validation / Test segments.

    fig03_demand_panels.{pdf,png}          double-column,  7.0 x 4.6 in
        2 x 2 grid of weekly augmented demand per dataset, lines colour
        coded by product.

    fig04_circular_fractions.{pdf,png}     double-column,  7.0 x 2.4 in
        1 x 4 strip showing u_s / u_m / u_r evolution per dataset on a
        common percent-recycled axis.

    fig05_distribution_shift.{pdf,png}     double-column,  7.0 x 2.6 in
        1 x 2 panel comparing train vs test KDEs for demand and price
        across the four benchmarks.

A combined cross-replication summary table is also written to
results/tables/cross_replication_summary.csv.
"""

from __future__ import annotations
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.utils.paths import (TRAIN_DIR,        VAL_DIR,        TEST_DIR,
                             OLIST_TRAIN_DIR,  OLIST_VAL_DIR,  OLIST_TEST_DIR,
                             HM_TRAIN_DIR,     HM_VAL_DIR,     HM_TEST_DIR,
                             SYNTH26_TRAIN_DIR, SYNTH26_VAL_DIR, SYNTH26_TEST_DIR,
                             FIGURES_DIR, TABLES_DIR)
from src.utils.plotting import set_style, save, COLORS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("comparative")


PRODUCT_PALETTE = [COLORS["blue"], COLORS["orange"], COLORS["green"],
                   COLORS["red"], COLORS["purple"]]

DATASET_LABELS = {
    "DataCo":      "(a) DataCo (USA, 2015-17)",
    "Olist":       "(b) Olist (Brazil, 2016-18)",
    "H&M":         "(c) H&M (global, 2018-20)",
    "Synth-2026":  "(d) Synth-2024-26 (current)",
}
DATASET_ORDER = ["DataCo", "Olist", "H&M", "Synth-2026"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_panel(train_dir, val_dir, test_dir, ds_name):
    parts = []
    for name, sub in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        df = pd.read_parquet(sub / f"{name}_full.parquet")
        df["split"]   = name
        df["dataset"] = ds_name
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def cross_replication_summary(panels):
    rows = []
    for ds_name, panel in panels.items():
        for split in ("train", "val", "test"):
            sub = panel[panel["split"] == split]
            rows.append({
                "dataset":        ds_name,
                "split":          split,
                "n_obs":          len(sub),
                "n_active":       int(sub["had_orders"].sum()),
                "weeks":          sub["week_start"].nunique(),
                "products":       sub["product"].nunique(),
                "retailers":      sub["retailer"].nunique(),
                "qty_aug_mean":   round(sub["qty_aug"].mean(), 2),
                "qty_aug_sd":     round(sub["qty_aug"].std(),  2),
                "p_mean":         round(sub["p"].mean(),       2),
                "ad_total_mean":  round((sub["b_s"] + sub["b_m"] + sub["b_r"]).mean(), 2),
                "avg_circ_mean":  round(sub["avg_circ"].mean(), 4),
                "first_week":     str(sub["week_start"].min().date()),
                "last_week":      str(sub["week_start"].max().date()),
            })
    df = pd.DataFrame(rows)
    out = TABLES_DIR / "cross_replication_summary.csv"
    df.to_csv(out, index=False)
    log.info("wrote %s", out)
    print(df.to_string(index=False))


def _split_boundaries(panel):
    return (panel.loc[panel["split"] == "train", "week_start"].max(),
            panel.loc[panel["split"] == "val",   "week_start"].max())


def _shade_splits(ax, train_end, val_end, panel, alpha_t=0.06, alpha_v=0.07,
                  alpha_te=0.08):
    xmin = panel["week_start"].min()
    xmax = panel["week_start"].max()
    ax.axvspan(xmin,      train_end, color=COLORS["blue"],   alpha=alpha_t,  lw=0)
    ax.axvspan(train_end, val_end,   color=COLORS["orange"], alpha=alpha_v,  lw=0)
    ax.axvspan(val_end,   xmax,      color=COLORS["green"],  alpha=alpha_te, lw=0)
    for x in (train_end, val_end):
        ax.axvline(x, color="black", linestyle=":", linewidth=0.5, alpha=0.7)


# --------------------------------------------------------------------------- #
# Fig 1 -- Coverage timeline (Gantt)
# --------------------------------------------------------------------------- #
def plot_coverage_timeline(panels):
    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.16, right=0.985)

    y_labels = []
    for i, name in enumerate(DATASET_ORDER):
        panel = panels[name]
        train = panel[panel["split"] == "train"]["week_start"]
        val   = panel[panel["split"] == "val"]["week_start"]
        test  = panel[panel["split"] == "test"]["week_start"]
        y = i

        ax.barh(y, mdates.date2num(train.max()) - mdates.date2num(train.min()),
                left=mdates.date2num(train.min()), height=0.55,
                color=COLORS["blue"], alpha=0.85, edgecolor="white", lw=0.4)
        ax.barh(y, mdates.date2num(val.max()) - mdates.date2num(val.min()),
                left=mdates.date2num(val.min()), height=0.55,
                color=COLORS["orange"], alpha=0.9, edgecolor="white", lw=0.4)
        ax.barh(y, mdates.date2num(test.max()) - mdates.date2num(test.min()),
                left=mdates.date2num(test.min()), height=0.55,
                color=COLORS["green"], alpha=0.9, edgecolor="white", lw=0.4)

        ax.text(mdates.date2num(panel["week_start"].max()) + 25, y,
                f"  n = {len(panel):,}",
                va="center", ha="left", fontsize=7.5)
        y_labels.append(name)

    ax.set_yticks(range(len(DATASET_ORDER)))
    ax.set_yticklabels(y_labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(mdates.date2num(pd.Timestamp("2014-09-01")),
                mdates.date2num(pd.Timestamp("2026-09-01")))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis='x', labelsize=7.5)
    ax.grid(axis='x', linestyle="--", alpha=0.4)
    ax.set_xlabel("Year", fontsize=8.5)

    handles = [Patch(facecolor=COLORS[c], alpha=0.85, label=lab)
               for c, lab in [("blue", "Train"),
                              ("orange", "Validation"),
                              ("green", "Test")]]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=7.5,
              frameon=False, handlelength=1.6, columnspacing=2.0)

    save(fig, FIGURES_DIR, "fig02_coverage_timeline")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 2 -- Demand panels (2x2)
# --------------------------------------------------------------------------- #
def plot_demand_panels(panels):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6),
                             gridspec_kw=dict(hspace=0.55, wspace=0.30))
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.10, right=0.985)

    for ax, name in zip(axes.ravel(), DATASET_ORDER):
        panel = panels[name]
        products = sorted(panel["product"].unique())
        cmap = dict(zip(products, PRODUCT_PALETTE[:len(products)]))
        train_end, val_end = _split_boundaries(panel)

        for p in products:
            ts = (panel[panel["product"] == p]
                  .groupby("week_start")["qty_aug"].mean().reset_index())
            ax.plot(ts["week_start"], ts["qty_aug"], color=cmap[p], lw=0.85)
        _shade_splits(ax, train_end, val_end, panel)

        ax.set_title(DATASET_LABELS[name], fontsize=8.5, loc="left")
        ax.set_ylabel("$D^{aug}$ (units / week)", fontsize=7.5)
        ax.tick_params(axis='x', rotation=30, labelsize=6.5)
        ax.tick_params(axis='y', labelsize=7.0)

    # global legends
    prod_handles = [Line2D([0], [0], color=PRODUCT_PALETTE[i], lw=1.5,
                           label=f"product $i = {i+1}$") for i in range(5)]
    split_handles = [Patch(facecolor=COLORS[c], alpha=0.55, label=lab)
                     for c, lab in [("blue", "Train"),
                                    ("orange", "Validation"),
                                    ("green", "Test")]]
    leg1 = fig.legend(handles=prod_handles, loc="upper left",
                      bbox_to_anchor=(0.10, 0.985), ncol=5,
                      fontsize=7, frameon=False, handlelength=1.3,
                      columnspacing=1.1)
    fig.legend(handles=split_handles, loc="upper right",
               bbox_to_anchor=(0.985, 0.985), ncol=3,
               fontsize=7, frameon=False, handlelength=1.3,
               columnspacing=1.1)
    fig.add_artist(leg1)

    save(fig, FIGURES_DIR, "fig03_demand_panels")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 3 -- Circular fraction strip (1x4)
# --------------------------------------------------------------------------- #
def plot_circular_fractions(panels):
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.4),
                             gridspec_kw=dict(wspace=0.35))
    fig.subplots_adjust(top=0.78, bottom=0.27, left=0.07, right=0.985)

    # determine common y-max for visual comparability
    ymax = 0
    for name in DATASET_ORDER:
        panel = panels[name]
        ymax = max(ymax, 100 * panel[["u_s", "u_m", "u_r"]].mean(axis=1).max())
    ymax = np.ceil(ymax * 1.1)

    for ax, name in zip(axes, DATASET_ORDER):
        panel = panels[name]
        train_end, val_end = _split_boundaries(panel)
        weekly = panel.groupby("week_start")[["u_s", "u_m", "u_r"]].mean().reset_index()
        ax.plot(weekly["week_start"], 100 * weekly["u_s"],
                color=COLORS["blue"],   lw=0.9)
        ax.plot(weekly["week_start"], 100 * weekly["u_m"],
                color=COLORS["orange"], lw=0.9)
        ax.plot(weekly["week_start"], 100 * weekly["u_r"],
                color=COLORS["green"],  lw=0.9)
        _shade_splits(ax, train_end, val_end, panel)
        ax.set_title(DATASET_LABELS[name], fontsize=8.0, loc="left")
        ax.set_ylim(0, ymax)
        ax.tick_params(axis='x', rotation=30, labelsize=6.0)
        ax.tick_params(axis='y', labelsize=7.0)

    axes[0].set_ylabel("circular fraction (% recycled)", fontsize=7.5)

    handles = [Line2D([0], [0], color=COLORS[c], lw=1.4, label=lab)
               for c, lab in [("blue",   r"$u_s$ supplier"),
                              ("orange", r"$u_m$ manufacturer"),
                              ("green",  r"$u_r$ retailer")]]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=7.5,
               frameon=False, handlelength=1.6, columnspacing=2.0)

    save(fig, FIGURES_DIR, "fig04_circular_fractions")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 4 -- Distribution shift (1x2)
# --------------------------------------------------------------------------- #
def _kde_panel(ax, train_vals, test_vals, color):
    """Light KDE using numpy histogram + smoothing.  Avoids scipy dependency."""
    bins = np.linspace(0, max(train_vals.max(), test_vals.max()) * 1.02, 60)
    h_tr, edges = np.histogram(train_vals, bins=bins, density=True)
    h_te, _     = np.histogram(test_vals,  bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, h_tr, color=color, lw=1.1, alpha=0.85)
    ax.fill_between(centers, h_te, color=color, alpha=0.18, lw=0)


def plot_distribution_shift(panels):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7),
                             gridspec_kw=dict(wspace=0.30))
    fig.subplots_adjust(top=0.86, bottom=0.20, left=0.10, right=0.985)

    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"],
              COLORS["red"]]

    # left -- demand (qty_aug); normalise within each dataset for visibility
    for c, name in zip(colors, DATASET_ORDER):
        panel = panels[name]
        scale = panel["qty_aug"].mean()
        if scale <= 0:
            continue
        tr = panel.loc[panel["split"] == "train", "qty_aug"].values / scale
        te = panel.loc[panel["split"] == "test",  "qty_aug"].values / scale
        bins = np.linspace(0, 4.0, 50)
        h_tr, edges = np.histogram(tr, bins=bins, density=True)
        h_te, _     = np.histogram(te, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axes[0].plot(centers, h_tr, color=c, lw=1.1, label=name)
        axes[0].plot(centers, h_te, color=c, lw=0.9, linestyle="--", alpha=0.85)
    axes[0].set_xlabel(r"$D^{aug}$ / mean$(D^{aug}_{\rm train})$", fontsize=7.5)
    axes[0].set_ylabel("density", fontsize=7.5)
    axes[0].set_title("(a) Augmented demand", fontsize=8.5, loc="left")
    axes[0].tick_params(labelsize=7.0)

    # right -- price (p)
    for c, name in zip(colors, DATASET_ORDER):
        panel = panels[name]
        scale = panel["p"].mean()
        if scale <= 0:
            continue
        tr = panel.loc[panel["split"] == "train", "p"].values / scale
        te = panel.loc[panel["split"] == "test",  "p"].values / scale
        bins = np.linspace(0, 4.0, 50)
        h_tr, edges = np.histogram(tr, bins=bins, density=True)
        h_te, _     = np.histogram(te, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axes[1].plot(centers, h_tr, color=c, lw=1.1, label=name)
        axes[1].plot(centers, h_te, color=c, lw=0.9, linestyle="--", alpha=0.85)
    axes[1].set_xlabel(r"$p$ / mean$(p_{\rm train})$", fontsize=7.5)
    axes[1].set_ylabel("density", fontsize=7.5)
    axes[1].set_title("(b) Unit price", fontsize=8.5, loc="left")
    axes[1].tick_params(labelsize=7.0)

    handles_ds = [Line2D([0], [0], color=c, lw=1.4, label=lab)
                  for c, lab in zip(colors, DATASET_ORDER)]
    handles_split = [Line2D([0], [0], color="black", lw=1.1, label="Train"),
                     Line2D([0], [0], color="black", lw=0.9, linestyle="--",
                            label="Test")]
    leg1 = fig.legend(handles=handles_ds, loc="upper left",
                      bbox_to_anchor=(0.10, 0.985), ncol=4,
                      fontsize=7, frameon=False, handlelength=1.3,
                      columnspacing=1.0)
    fig.legend(handles=handles_split, loc="upper right",
               bbox_to_anchor=(0.985, 0.985), ncol=2,
               fontsize=7, frameon=False, handlelength=1.6,
               columnspacing=1.0)
    fig.add_artist(leg1)

    save(fig, FIGURES_DIR, "fig05_distribution_shift")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    panels = {
        "DataCo":     load_panel(TRAIN_DIR,         VAL_DIR,         TEST_DIR,         "DataCo"),
        "Olist":      load_panel(OLIST_TRAIN_DIR,   OLIST_VAL_DIR,   OLIST_TEST_DIR,   "Olist"),
        "H&M":        load_panel(HM_TRAIN_DIR,      HM_VAL_DIR,      HM_TEST_DIR,      "H&M"),
        "Synth-2026": load_panel(SYNTH26_TRAIN_DIR, SYNTH26_VAL_DIR, SYNTH26_TEST_DIR, "Synth-2026"),
    }
    log.info("Loaded panels: %s", {k: v.shape for k, v in panels.items()})
    cross_replication_summary(panels)

    log.info("Rendering manuscript-ready figures ...")
    plot_coverage_timeline(panels)
    plot_demand_panels(panels)
    plot_circular_fractions(panels)
    plot_distribution_shift(panels)


if __name__ == "__main__":
    sys.exit(main())
