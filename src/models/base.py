"""
Common interface for every demand model in the SIDN-CRO benchmark.

A demand model is any callable:

    D = model(p, b_s, b_m, b_r, u_s, u_m, u_r, context=...)

returning the predicted demand for each (product i, retailer j) cell at the
given decision-variable values, conditioned on the context features.

Two consumers expect this interface:

    1. SCM optimisation         (src/optim/sqp_layer.py)
    2. Training / evaluation     (src/train/, src/eval/)

The two adapters in this module bridge between

    *  the row-format representation used by tabular ML models, where each
       sample is a single (i, j, t) cell with all features in one vector,

    *  and the matrix-format representation expected by the SQP solver,
       where decisions are I x J arrays.

The transformation is fully invertible and uses the schema written by
make_train_val_test*.py:

        cont_feats        list of names in the order rows are produced
        cat_feats         retailer + product one-hot column names
        x_scaler          StandardScaler fit on training continuous features

Subclasses must implement :py:meth:`fit` (training-data based) and
:py:meth:`raw_predict` (returns demand on the *original* unit scale, given
the un-scaled feature matrix).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import json
import numpy as np
import joblib


# --------------------------------------------------------------------------- #
# Schema container
# --------------------------------------------------------------------------- #
@dataclass
class PanelSchema:
    cont_feats: list[str]
    cat_feats:  list[str]
    target:     str
    products:   list[str]
    retailers:  list[str]

    @classmethod
    def load(cls, schema_path: Path) -> "PanelSchema":
        with open(schema_path) as f:
            d = json.load(f)
        return cls(cont_feats=d["cont_feats"],
                   cat_feats=d["cat_feats"],
                   target=d["target"],
                   products=d["products"],
                   retailers=d["retailers"])

    @property
    def n_cont(self) -> int:
        return len(self.cont_feats)

    @property
    def n_cat(self) -> int:
        return len(self.cat_feats)


# --------------------------------------------------------------------------- #
# Decision-variable indexing helpers
# --------------------------------------------------------------------------- #
DECISION_KEYS = ("p", "b_s", "b_m", "b_r", "u_s", "u_m", "u_r")


def decision_columns(schema: PanelSchema) -> dict[str, int]:
    """Map each decision variable name to its column index in cont_feats."""
    out = {}
    for k in DECISION_KEYS:
        if k not in schema.cont_feats:
            raise ValueError(f"decision key {k!r} missing from cont_feats")
        out[k] = schema.cont_feats.index(k)
    return out


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class DemandModel:
    """Abstract interface for demand prediction.

    Concrete subclasses must implement
        fit(X_cont, X_cat, y_scaled, y_scaler)        and
        raw_predict(X_cont_unscaled, X_cat)            -> predicted demand.

    The methods below provide the optimisation-time API.
    """
    schema: PanelSchema
    name: str = "demand_model"

    def __init__(self, schema: PanelSchema, x_scaler, y_scaler):
        self.schema = schema
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    # --- Mandatory hooks ----------------------------------------------------
    def fit(self, X_cont_scaled, X_cat, y_scaled, **kw) -> "DemandModel":
        raise NotImplementedError

    def raw_predict(self, X_cont_unscaled: np.ndarray,
                    X_cat: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # --- Tabular interface (predict on a precomputed feature matrix) -------
    def predict_rows(self, X_cont_unscaled: np.ndarray,
                     X_cat: np.ndarray) -> np.ndarray:
        return self.raw_predict(X_cont_unscaled, X_cat)

    # --- Optimisation interface --------------------------------------------
    def __call__(self, p, b_s, b_m, b_r, u_s, u_m, u_r,
                 context: dict) -> np.ndarray:
        """Return demand D of shape (I, J) given a context dictionary.

        ``context`` must contain:
            * ``base_cont``   ndarray (I*J, n_cont) -- the *unscaled*
                              continuous-feature matrix for the (i, j) cells
                              we want to predict, with the decision-variable
                              columns set to *current* values.  We will
                              overwrite the decision columns with the
                              passed-in p, b, u arrays.
            * ``X_cat``       ndarray (I*J, n_cat) -- one-hot encodings.
            * ``shape``       tuple (I, J) for the reshape.
        """
        I, J = context["shape"]
        cont = context["base_cont"].copy()
        cat  = context["X_cat"]
        idx  = decision_columns(self.schema)
        cont[:, idx["p"]]   = p.ravel()
        cont[:, idx["b_s"]] = b_s.ravel()
        cont[:, idx["b_m"]] = b_m.ravel()
        cont[:, idx["b_r"]] = b_r.ravel()
        # u_*, broadcast over retailers per product
        u_s_full = np.repeat(u_s, J)
        u_m_full = np.repeat(u_m, J)
        u_r_full = np.repeat(u_r, J)
        cont[:, idx["u_s"]] = u_s_full
        cont[:, idx["u_m"]] = u_m_full
        cont[:, idx["u_r"]] = u_r_full

        D_flat = self.raw_predict(cont, cat)
        return np.maximum(0.0, D_flat).reshape(I, J)

    # --- Persistence --------------------------------------------------------
    def save(self, path: Path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "DemandModel":
        return joblib.load(path)


# --------------------------------------------------------------------------- #
# Helper -- build context for a target (product, retailer, week)
# --------------------------------------------------------------------------- #
def build_optimisation_context(panel_row, schema: PanelSchema, x_scaler,
                               n_products: int, n_retailers: int,
                               product_to_idx: dict, retailer_to_idx: dict,
                               cont_template: np.ndarray) -> dict:
    """Construct the (base_cont, X_cat, shape) context dict for one (week)
    optimisation call.  The decision-variable columns of base_cont are set
    to placeholder zeros; ``solve_sqp`` will overwrite them.

    ``cont_template`` is an (n_products * n_retailers, n_cont) matrix where
    every row corresponds to a specific (i, j) cell, with all *non-decision*
    features set to the values observed for that (week, product, retailer).
    """
    return {
        "base_cont": cont_template,
        "X_cat":     None,    # filled at construction site
        "shape":     (n_products, n_retailers),
        "panel_row": panel_row,
    }
