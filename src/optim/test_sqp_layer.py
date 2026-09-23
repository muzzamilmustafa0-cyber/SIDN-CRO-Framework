"""Stand-alone smoke test for the SQP layer.

Builds a tiny 2 product x 3 retailer instance with the analytical demand
model, solves it, and prints summary statistics.  Run as

    python -m src.optim.test_sqp_layer
"""

from __future__ import annotations
import numpy as np

from src.optim.sqp_layer import (SCMParameters, AnalyticalDemandModel,
                                 solve_sqp)


def main():
    np.random.seed(0)
    I, J = 2, 3

    params = SCMParameters(
        I=I, J=J,
        C_s   = np.array([12.0, 10.0]),
        C_m   = np.array([ 8.0,  7.0]),
        C_r   = np.array([[ 4.0, 4.5, 4.2],
                          [ 3.5, 3.8, 3.6]]),
        CrCAd = np.array([-5.0, -3.0]),
        h_s   = np.array([1.5, 1.2]),
        h_m   = np.array([1.5, 1.2]),
        h_r   = np.array([[1.0, 1.0, 1.0],
                          [0.9, 0.9, 0.9]]),
        A_s   = np.array([200.0, 200.0]),
        A_m   = np.array([150.0, 150.0]),
        A_r   = np.array([[80.0, 80.0, 80.0],
                          [80.0, 80.0, 80.0]]),
        c2_s  = np.array([40.0, 40.0]),
        c2_m  = np.array([30.0, 30.0]),
        c2_r  = np.array([[10.0, 10.0, 10.0],
                          [10.0, 10.0, 10.0]]),
        inc_rate  = 1.5,
        alpha_inc = 0.8,
        A_R       = np.array([900.0, 900.0, 900.0]),
        AdBudget  = np.array([4000.0, 4000.0]),
        Capacity  = 6000.0,
        p_lo      = np.full((I, J),  20.0),
        p_hi      = np.full((I, J), 120.0),
        b_hi      = np.full((I, J), 1500.0),
        u_hi      = 0.50,
    )

    demand_fn = AnalyticalDemandModel(
        D0   = np.array([[300., 250., 200.],
                         [220., 200., 180.]]),
        p0   = np.array([[ 60.,  60.,  60.],
                         [ 50.,  50.,  50.]]),
        beta = 0.40, k1 = 0.030, k2 = 0.030, k3 = 0.040,
        alpha_inc = 0.80,
    )

    sol = solve_sqp(params, demand_fn, verbose=False)

    print("=" * 70)
    print(f"SQP layer smoke test  --  I={I} products, J={J} retailers")
    print("=" * 70)
    print(f"Solver success    : {sol.raw.success}")
    print(f"Iterations        : {sol.raw.nit}")
    print(f"Feasible          : {sol.feasible}")
    print(f"Total profit      : ${sol.profit:,.2f}")
    print()
    print("Decision summary (rounded):")
    print("  prices        :", np.round(sol.p, 2).tolist())
    print("  ad b_s        :", np.round(sol.b_s, 1).tolist())
    print("  ad b_m        :", np.round(sol.b_m, 1).tolist())
    print("  ad b_r        :", np.round(sol.b_r, 1).tolist())
    print("  u_s, u_m, u_r :", np.round(sol.u_s, 3).tolist(),
                               np.round(sol.u_m, 3).tolist(),
                               np.round(sol.u_r, 3).tolist())
    print()
    print("Demand realised :", np.round(sol.demand, 1).tolist())
    print(f"Total demand    : {sol.demand.sum():.1f} units")
    print()

    # sanity checks
    assert sol.raw.success or sol.feasible, "Solver did not return a feasible point"
    assert sol.profit > 0, "Profit should be positive on this instance"
    assert (sol.demand >= 0).all(), "Demand must be non-negative"
    assert (sol.u_s >= 0).all() and (sol.u_s <= params.u_hi).all()
    assert (sol.u_m >= 0).all() and (sol.u_m <= params.u_hi).all()
    assert (sol.u_r >= 0).all() and (sol.u_r <= params.u_hi).all()
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
