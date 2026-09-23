"""
Produce publication-quality data-summary artefacts.

Outputs
-------
results/figures/fig_dataset_overview.{pdf,png}   2 x 3 panel:
    (a) weekly augmented vs observed demand per retailer
    (b) cooperative advertising spend per echelon (averaged across retailers)
    (c) circular-fraction trajectory u_s, u_m, u_r per retailer
    (d) selling-price time series per retailer
    (e) demand histogram (qty_aug, train split)
    (f) feature correlation heatmap

results/tables/dataset_summary.csv               Per-split row counts and means.
results/tables/feature_descriptions.csv          Plain-language data dictionary.
"""

from __future__ import annotations
import logging
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.paths import (PREPROCESSED_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
                             FIGURES_DIR, TABLES_DIR)
from src.utils.plotting import (set_style, figsize, save,
                                COLORS, RETAILER_COLOR, PRODUCT_COLOR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("data_summary")


def load_full_panel() -> pd.DataFrame:
    parts = []
    for name, sub in [("train", TRAIN_DIR),
                      ("val",   VAL_DIR),
                      ("test",  TEST_DIR)]:
        df = pd.read_parquet(sub / f"{name}_full.parquet")
        df["split"] = name
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def write_summary_table(panel: pd.DataFrame) -> None:
    rows = []
    for split, sub in panel.groupby("split"):
        rows.append({
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
    df = pd.DataFrame(rows).set_index("split").loc[["train", "val", "test"]]
    out = TABLES_DIR / "dataset_summary.csv"
    df.to_csv(out)
    log.info("wrote %s", out)
    print(df.to_string())

    # per-(product, retailer) breakdown across all splits
    rows = []
    for (prod, ret), sub in panel.groupby(["product", "retailer"]):
        rows.append({
            "product":        prod,
            "retailer":       ret,
            "n_obs":          len(sub),
            "qty_aug_mean":   round(sub["qty_aug"].mean(), 2),
            "p_mean":         round(sub["p"].mean(),       2),
            "ad_total_mean":  round((sub["b_s"] + sub["b_m"] + sub["b_r"]).mean(), 2),
            "avg_circ_mean":  round(sub["avg_circ"].mean(), 4),
        })
    df2 = pd.DataFrame(rows)
    df2.to_csv(TABLES_DIR / "panel_breakdown.csv", index=False)
    log.info("wrote panel_breakdown.csv (%d rows)", len(df2))


def write_data_dictionary() -> None:
    rows = [
        ("product",     "categorical",  "Product line: one of {Fan Shop, Apparel, Golf, Footwear, Outdoors} -- top-5 DataCo departments by volume (~96% coverage)"),
        ("retailer",    "categorical",  "B2C/B2B segment (Consumer / Corporate / Home Office) used as the three retailers"),
        ("week_start",  "timestamp",    "ISO week start (Monday)"),
        ("p",           "USD/unit",     "Mean Order Item Product Price for the (retailer, week)"),
        ("disc_rate",   "fraction",     "Mean Order Item Discount Rate"),
        ("qty_obs",     "units/week",   "Sum of Order Item Quantity (raw DataCo demand signal)"),
        ("qty_aug",     "units/week",   "Augmented target: qty_obs * Ad_eff(b) * Cir_eff(u) * exp(eps), eps~N(0,0.05^2)"),
        ("qty_baseline", "units/week",  "qty_obs (renamed for diagnostics)"),
        ("sales",       "USD/week",     "Sum of Sales"),
        ("profit",      "USD/week",     "Sum of Order Profit Per Order"),
        ("n_orders",    "count",        "Distinct Order Id count"),
        ("had_orders",  "0/1",          "Indicator: any order recorded that week"),
        ("share_eu",    "fraction",     "Share of weekly orders shipped to Europe"),
        ("share_latam", "fraction",     "Share to LATAM"),
        ("share_pac",   "fraction",     "Share to Pacific Asia"),
        ("share_usca",  "fraction",     "Share to US/Canada"),
        ("share_africa","fraction",     "Share to Africa"),
        ("b_s",         "USD/week",     "Calibrated supplier ad spend (cooperative split 25%)"),
        ("b_m",         "USD/week",     "Calibrated manufacturer ad spend (35%)"),
        ("b_r",         "USD/week",     "Calibrated retailer ad spend (40%)"),
        ("u_s",         "[0,1]",        "Calibrated supplier circular fraction"),
        ("u_m",         "[0,1]",        "Calibrated manufacturer circular fraction"),
        ("u_r",         "[0,1]",        "Calibrated retailer circular fraction"),
        ("avg_circ",    "[0,1]",        "Mean of (u_s, u_m, u_r)"),
        ("ad_eff",      "multiplier",   "1 + k1*ln(1+b_s) + k2*ln(1+b_m) + k3*ln(1+b_r)"),
        ("cir_eff",     "multiplier",   "1 + alpha_inc * avg_circ"),
        ("sin_woy",     "feature",      "sin(2*pi*weekofyear/52)"),
        ("cos_woy",     "feature",      "cos(2*pi*weekofyear/52)"),
        ("year_norm",   "feature",      "(year - 2015)/3"),
        ("product_Fan_Shop",     "0/1", "Product one-hot encoding"),
        ("product_Apparel",      "0/1", "Product one-hot encoding"),
        ("product_Golf",         "0/1", "Product one-hot encoding"),
        ("product_Footwear",     "0/1", "Product one-hot encoding"),
        ("product_Outdoors",     "0/1", "Product one-hot encoding"),
        ("retailer_Consumer",    "0/1", "Retailer one-hot encoding"),
        ("retailer_Corporate",   "0/1", "Retailer one-hot encoding"),
        ("retailer_Home_Office", "0/1", "Retailer one-hot encoding"),
    ]
    df = pd.DataFrame(rows, columns=["variable", "type/unit", "description"])
    out = TABLES_DIR / "feature_descriptions.csv"
    df.to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(df))


# --------------------------------------------------------------------------- #
# Main figure
# --------------------------------------------------------------------------- #
def _split_boundaries(panel: pd.DataFrame):
    train_end = panel.loc[panel["split"] == "train", "week_start"].max()
    val_end   = panel.loc[panel["split"] == "val",   "week_start"].max()
    return train_end, val_end


def _add_split_lines(ax, train_end, val_end):
    for x in (train_end, val_end):
        ax.axvline(x, color="black", linestyle=":", linewidth=0.6, alpha=0.7)


def _label_split_regions(ax, train_end, val_end, panel) -> None:
    """Shade Train / Val / Test bands lightly behind the lines."""
    xmin = panel["week_start"].min()
    xmax = panel["week_start"].max()
    ax.axvspan(xmin,      train_end, color="#0072B2", alpha=0.04, lw=0)
    ax.axvspan(train_end, val_end,   color="#E69F00", alpha=0.05, lw=0)
    ax.axvspan(val_end,   xmax,      color="#009E73", alpha=0.06, lw=0)
    for x in (train_end, val_end):
        ax.axvline(x, color="black", linestyle=":", linewidth=0.6, alpha=0.7)


def plot_overview(panel: pd.DataFrame) -> None:
    set_style()
    train_end, val_end = _split_boundaries(panel)

    fig = plt.figure(figsize=(7.2, 9.4))
    gs = fig.add_gridspec(3, 2, hspace=0.80, wspace=0.34,
                          top=0.91, bottom=0.08, left=0.09, right=0.985)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, 0])
    axF = fig.add_subplot(gs[2, 1])

    # ---- (A) augmented demand per product (averaged across retailers)
    for prod, sub in panel.groupby("product"):
        ts = (sub.groupby("week_start")["qty_aug"].mean().reset_index())
        axA.plot(ts["week_start"], ts["qty_aug"],
                 color=PRODUCT_COLOR[prod], label=prod, lw=0.9)
    _label_split_regions(axA, train_end, val_end, panel)
    axA.set_title("(a) Augmented weekly demand $D^{aug}_{i,j,t}$ per product")
    axA.set_ylabel("units / week (mean over retailers)")
    axA.tick_params(axis='x', rotation=30)

    # ---- (B) cooperative ad spend
    weekly = panel.groupby("week_start")[["b_s", "b_m", "b_r"]].mean().reset_index()
    axB.plot(weekly["week_start"], weekly["b_s"], color=COLORS["blue"],
             label=r"$b_{s}$ supplier",     lw=0.9)
    axB.plot(weekly["week_start"], weekly["b_m"], color=COLORS["orange"],
             label=r"$b_{m}$ manufacturer", lw=0.9)
    axB.plot(weekly["week_start"], weekly["b_r"], color=COLORS["green"],
             label=r"$b_{r}$ retailer",     lw=0.9)
    _label_split_regions(axB, train_end, val_end, panel)
    axB.set_title("(b) Cooperative ad spend (cross-retailer mean)")
    axB.set_ylabel("USD / week")
    axB.legend(loc="upper left", framealpha=0.9, fontsize=7)
    axB.tick_params(axis='x', rotation=30)

    # ---- (C) circular fractions
    cuts = panel.groupby("week_start")[["u_s", "u_m", "u_r"]].mean().reset_index()
    axC.plot(cuts["week_start"], 100 * cuts["u_s"], color=COLORS["blue"],
             label=r"$u_s$ supplier",     lw=0.9)
    axC.plot(cuts["week_start"], 100 * cuts["u_m"], color=COLORS["orange"],
             label=r"$u_m$ manufacturer", lw=0.9)
    axC.plot(cuts["week_start"], 100 * cuts["u_r"], color=COLORS["green"],
             label=r"$u_r$ retailer",     lw=0.9)
    _label_split_regions(axC, train_end, val_end, panel)
    axC.set_title("(c) Circular-input fractions (cross-retailer mean)")
    axC.set_ylabel("% recycled / circular")
    axC.legend(loc="upper left", framealpha=0.9, fontsize=7)
    axC.tick_params(axis='x', rotation=30)

    # ---- (D) selling price per product (averaged across retailers)
    for prod, sub in panel.groupby("product"):
        ts = sub.groupby("week_start")["p"].mean().reset_index()
        axD.plot(ts["week_start"], ts["p"],
                 color=PRODUCT_COLOR[prod], label=prod, lw=0.9)
    _label_split_regions(axD, train_end, val_end, panel)
    axD.set_title("(d) Mean unit selling price $p_{i,j,t}$ per product")
    axD.set_ylabel("USD / unit")
    axD.tick_params(axis='x', rotation=30)

    # ---- (E) target distribution by split
    train = panel[panel["split"] == "train"]["qty_aug"]
    val   = panel[panel["split"] == "val"  ]["qty_aug"]
    test  = panel[panel["split"] == "test" ]["qty_aug"]
    bins = np.linspace(0, max(train.max(), val.max(), test.max()) * 1.02, 30)
    axE.hist(train, bins=bins, alpha=0.55, color=COLORS["blue"],   label="Train")
    axE.hist(val,   bins=bins, alpha=0.55, color=COLORS["orange"], label="Val")
    axE.hist(test,  bins=bins, alpha=0.55, color=COLORS["green"],  label="Test")
    axE.set_title("(e) Augmented demand distribution by split")
    axE.set_xlabel("units / week")
    axE.set_ylabel("# observations")
    axE.legend(loc="upper right", framealpha=0.9, fontsize=7)

    # ---- (F) Pearson correlation
    corr_cols = ["p", "disc_rate", "b_s", "b_m", "b_r",
                 "u_s", "u_m", "u_r", "qty_aug"]
    corr = panel[corr_cols].corr().loc["qty_aug"].drop("qty_aug")
    bar_colors = [COLORS["red"] if v < 0 else COLORS["blue"] for v in corr.values]
    axF.barh(range(len(corr)), corr.values, color=bar_colors)
    axF.set_yticks(range(len(corr)))
    axF.set_yticklabels(corr.index)
    axF.invert_yaxis()
    axF.axvline(0, color="black", lw=0.5)
    axF.set_title("(f) Pearson correlation with $D^{aug}$")
    axF.set_xlabel("correlation coefficient")

    # ---- top-of-figure legends
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    product_handles = [Line2D([0], [0], color=PRODUCT_COLOR[p], lw=1.4, label=p)
                       for p in ["Fan Shop", "Apparel", "Golf",
                                 "Footwear", "Outdoors"]]
    split_handles = [
        Patch(facecolor="#0072B2", alpha=0.18, label="Train"),
        Patch(facecolor="#E69F00", alpha=0.22, label="Validation"),
        Patch(facecolor="#009E73", alpha=0.24, label="Test"),
    ]
    leg1 = fig.legend(handles=product_handles, loc="upper left",
                      bbox_to_anchor=(0.09, 0.985),
                      ncol=5, title="Product $i$ (panels a, d)",
                      frameon=False, fontsize=8, title_fontsize=8,
                      handlelength=1.6, columnspacing=1.0)
    fig.add_artist(leg1)
    fig.legend(handles=split_handles, loc="upper right",
               bbox_to_anchor=(0.985, 0.985),
               ncol=3, title="Temporal split (shaded bands)",
               frameon=False, fontsize=8, title_fontsize=8)

    save(fig, FIGURES_DIR, "fig01_dataset_overview")
    plt.close(fig)


def main() -> None:
    panel = load_full_panel()
    log.info("Loaded full panel: %s", panel.shape)
    write_summary_table(panel)
    write_data_dictionary()
    plot_overview(panel)


if __name__ == "__main__":
    sys.exit(main())
