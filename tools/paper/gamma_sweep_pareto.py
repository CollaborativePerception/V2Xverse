#!/usr/bin/env python3
"""Combine the 8-point fixed-CBF gamma sweep + adaptive-CBF reference
into a single Pareto table (score vs. collision rate) for the paper's
sensitivity-sweep figure (IROS review R1/R8 W3/Q1)."""
import csv
import os

GAMMA_SWEEP_CSV = "paper_outputs/gamma_sweep/method_summary.csv"
MAIN_CSV = "paper_outputs/final_eval3/method_summary.csv"
OUT_CSV = "paper_outputs/gamma_sweep/gamma_sweep_pareto.csv"
OUT_FIG = "paper_outputs/gamma_sweep/figures/gamma_sweep_pareto.png"

METHOD_TO_GAMMA = {
    "fixed_cbf_low_gamma": 0.1,
    "fixed_cbf_g02": 0.2,
    "fixed_cbf_g03": 0.3,
    "fixed_cbf_g04": 0.4,
    "fixed_cbf": 0.5,
    "fixed_cbf_g06": 0.6,
    "fixed_cbf_g07": 0.7,
    "fixed_cbf_high_gamma": 0.8,
}


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows_sweep = {r["method"]: r for r in _read_csv(GAMMA_SWEEP_CSV)}
    rows_main = {r["method"]: r for r in _read_csv(MAIN_CSV)}

    combined = []
    for method, gamma in METHOD_TO_GAMMA.items():
        src = rows_main if method == "fixed_cbf" else rows_sweep
        r = src[method]
        combined.append(
            {
                "setting": method,
                "gamma": gamma,
                "n_samples": int(r["n_samples"]),
                "score_composed_mean": float(r["score_composed_mean"]),
                "score_composed_ci95": float(r["score_composed_ci95"]),
                "completion_rate": float(r["completion_rate"]),
                "collisions_per_km": float(r["collisions_per_km"]),
            }
        )
    combined.sort(key=lambda r: r["gamma"])

    ref = rows_main["rl_cbf_adaptive"]
    adaptive_row = {
        "setting": "rl_cbf_adaptive",
        "gamma": "adaptive",
        "n_samples": int(ref["n_samples"]),
        "score_composed_mean": float(ref["score_composed_mean"]),
        "score_composed_ci95": float(ref["score_composed_ci95"]),
        "completion_rate": float(ref["completion_rate"]),
        "collisions_per_km": float(ref["collisions_per_km"]),
    }
    combined.append(adaptive_row)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
        writer.writeheader()
        writer.writerows(combined)
    print("Wrote:", OUT_CSV)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure")
        return

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fixed_pts = [r for r in combined if r["setting"] != "rl_cbf_adaptive"]
    xs = [r["collisions_per_km"] for r in fixed_pts]
    ys = [r["score_composed_mean"] for r in fixed_pts]
    yerr = [r["score_composed_ci95"] for r in fixed_pts]
    ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color="tab:blue", label="Fixed CBF (gamma sweep)", capsize=3)
    for r in fixed_pts:
        ax.annotate("g={:.1f}".format(r["gamma"]), (r["collisions_per_km"], r["score_composed_mean"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.errorbar([adaptive_row["collisions_per_km"]], [adaptive_row["score_composed_mean"]],
                yerr=[adaptive_row["score_composed_ci95"]], fmt="*", color="tab:red",
                markersize=14, label="Adaptive CBF (RL)", capsize=3)

    ax.set_xlabel("Collisions per km (lower is better)")
    ax.set_ylabel("Driving score (higher is better)")
    ax.set_title("Fixed-CBF gamma sensitivity sweep, Town05 (n=105 per point, n=315 adaptive)")
    ax.legend()
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print("Wrote:", OUT_FIG)


if __name__ == "__main__":
    main()
