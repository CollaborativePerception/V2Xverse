"""
Descriptive proxy for the class-specific-vs-shared-gamma ablation (R1 / R8 W4 Q5)
when a real retrain-under-constraint ablation isn't feasible.

Reads the per-step 'gammas' dict (route_info['gammas'], written by
pnp_infer_action_e2e.py only when simulation.save_step_outputs=True and RL is
active) and reports how much the trained class-specific policy's three learned
gains actually diverge from each other in practice. This does NOT answer "would
performance change under a shared-gamma constraint" -- only training under that
constraint answers that. It answers the weaker question "is the policy actually
using its per-class freedom, or converging to a de facto shared value anyway."

Usage:
    python3 tools/paper/gamma_divergence_analysis.py \
        --runs-root results/paper_eval/town05_gamma_divergence_probe \
        --method rl_cbf_adaptive \
        --output-csv paper_outputs/gamma_divergence/gamma_divergence_summary.csv
"""
import argparse
import glob
import json
import os

import numpy as np


def _iter_step_jsons(method_root):
    pattern = os.path.join(method_root, "**", "ego_vehicle_*", "*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    method_root = os.path.join(args.runs_root, args.method)
    veh, ped, bike = [], [], []
    num_steps = 0
    num_missing = 0

    for path in _iter_step_jsons(method_root):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        gammas = data.get("gammas")
        if not isinstance(gammas, dict):
            num_missing += 1
            continue
        try:
            v = float(gammas["gamma_vehicle"])
            p = float(gammas["gamma_ped"])
            b = float(gammas["gamma_bike"])
        except (KeyError, TypeError, ValueError):
            num_missing += 1
            continue
        veh.append(v)
        ped.append(p)
        bike.append(b)
        num_steps += 1

    if num_steps == 0:
        raise SystemExit("No steps with logged gammas found under {}".format(method_root))

    veh = np.array(veh)
    ped = np.array(ped)
    bike = np.array(bike)

    pair_vp = np.abs(veh - ped)
    pair_vb = np.abs(veh - bike)
    pair_pb = np.abs(ped - bike)
    max_pairwise = np.maximum(np.maximum(pair_vp, pair_vb), pair_pb)

    thresholds = [0.02, 0.05, 0.10]
    row = {
        "num_steps": num_steps,
        "num_missing": num_missing,
        "gamma_vehicle_mean": float(veh.mean()),
        "gamma_vehicle_std": float(veh.std()),
        "gamma_ped_mean": float(ped.mean()),
        "gamma_ped_std": float(ped.std()),
        "gamma_bike_mean": float(bike.mean()),
        "gamma_bike_std": float(bike.std()),
        "mean_abs_diff_vehicle_ped": float(pair_vp.mean()),
        "mean_abs_diff_vehicle_bike": float(pair_vb.mean()),
        "mean_abs_diff_ped_bike": float(pair_pb.mean()),
        "mean_max_pairwise_diff": float(max_pairwise.mean()),
        "corr_vehicle_ped": float(np.corrcoef(veh, ped)[0, 1]),
        "corr_vehicle_bike": float(np.corrcoef(veh, bike)[0, 1]),
        "corr_ped_bike": float(np.corrcoef(ped, bike)[0, 1]),
    }
    for t in thresholds:
        row["pct_steps_all_within_{:.2f}".format(t)] = 100.0 * float(np.mean(max_pairwise <= t))

    print("Steps analyzed: {} (missing/skipped: {})".format(num_steps, num_missing))
    print("gamma_vehicle: mean={:.4f} std={:.4f}".format(row["gamma_vehicle_mean"], row["gamma_vehicle_std"]))
    print("gamma_ped:     mean={:.4f} std={:.4f}".format(row["gamma_ped_mean"], row["gamma_ped_std"]))
    print("gamma_bike:    mean={:.4f} std={:.4f}".format(row["gamma_bike_mean"], row["gamma_bike_std"]))
    print("mean |veh-ped|  = {:.4f}".format(row["mean_abs_diff_vehicle_ped"]))
    print("mean |veh-bike| = {:.4f}".format(row["mean_abs_diff_vehicle_bike"]))
    print("mean |ped-bike| = {:.4f}".format(row["mean_abs_diff_ped_bike"]))
    for t in thresholds:
        print("pct steps all 3 gammas within {:.2f}: {:.1f}%".format(t, row["pct_steps_all_within_{:.2f}".format(t)]))

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w") as f:
            f.write(",".join(row.keys()) + "\n")
            f.write(",".join(str(v) for v in row.values()) + "\n")
        print("Wrote {}".format(args.output_csv))


if __name__ == "__main__":
    main()
