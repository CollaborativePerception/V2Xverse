"""
Summarize CBF-QP solve-time and slack-activation statistics from the per-step
dashboard JSONs saved by pnp_infer_action_e2e.py (route_info['cbf_info']).

Addresses two reporting gaps flagged in IROS 2026 review:
  - Reviewer 1: "Execution time is omitted... must document OSQP solve times."
  - Reviewer 1 / Reviewer 8 Q4: "The frequency of nonzero slack activation... is
    never reported" / "How often does the QP rely on slack or fail?"

This does NOT overlap with tools/paper/aggregate_eval.py, which only reads the
route-level results.json files and never opens these per-step JSONs.

Usage:
    python3 tools/paper/aggregate_cbf_stats.py \
        --runs-root results/paper_eval/<experiment_name> \
        --output-csv paper_outputs/cbf_stats.csv
"""
import argparse
import glob
import json
import os

import numpy as np


FALLBACK_STATUSES = {"qp_fail", "missing_solver", "no_ego", "bad_dt"}


def _iter_step_jsons(method_root):
    # results/<experiment>/<method>/pass*/images*/<route>/ego_vehicle_0/NNNN.json
    pattern = os.path.join(method_root, "**", "ego_vehicle_*", "*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        yield path


def summarize_method(method_name, method_root):
    solve_times = []
    slack_flags = []
    fallback_flags = []
    num_steps = 0
    num_missing_cbf_info = 0

    for path in _iter_step_jsons(method_root):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        cbf_info = data.get("cbf_info")
        if not isinstance(cbf_info, dict):
            num_missing_cbf_info += 1
            continue
        num_steps += 1
        status = cbf_info.get("status", "")
        fallback_flags.append(1 if status in FALLBACK_STATUSES else 0)
        solve_times.append(float(cbf_info.get("solve_time_ms", 0.0)))
        slack_flags.append(1 if int(cbf_info.get("num_active_slack", 0)) > 0 else 0)

    if num_steps == 0:
        return {
            "method": method_name,
            "num_steps": 0,
            "num_missing_cbf_info": num_missing_cbf_info,
            "solve_time_ms_mean": None,
            "solve_time_ms_p95": None,
            "solve_time_ms_max": None,
            "pct_steps_active_slack": None,
            "pct_steps_solver_fallback": None,
        }

    solve_times_arr = np.array(solve_times, dtype=np.float64)
    return {
        "method": method_name,
        "num_steps": num_steps,
        "num_missing_cbf_info": num_missing_cbf_info,
        "solve_time_ms_mean": float(np.mean(solve_times_arr)),
        "solve_time_ms_p95": float(np.percentile(solve_times_arr, 95)),
        "solve_time_ms_max": float(np.max(solve_times_arr)),
        "pct_steps_active_slack": 100.0 * float(np.mean(slack_flags)),
        "pct_steps_solver_fallback": 100.0 * float(np.mean(fallback_flags)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True,
                         help="Experiment output root, e.g. results/paper_eval/town05_main_10ms")
    parser.add_argument("--methods", nargs="*", default=None,
                         help="Method subfolder names to summarize (default: all found)")
    parser.add_argument("--output-csv", default=None,
                         help="Optional path to write a CSV summary")
    args = parser.parse_args()

    if args.methods:
        method_names = args.methods
    else:
        method_names = sorted(
            d for d in os.listdir(args.runs_root)
            if os.path.isdir(os.path.join(args.runs_root, d))
        )

    rows = []
    for method_name in method_names:
        method_root = os.path.join(args.runs_root, method_name)
        row = summarize_method(method_name, method_root)
        rows.append(row)
        print(
            "{method:24s} steps={num_steps:6d}  "
            "solve_ms(mean/p95/max)={solve_time_ms_mean}/{solve_time_ms_p95}/{solve_time_ms_max}  "
            "active_slack={pct_steps_active_slack}%  "
            "solver_fallback={pct_steps_solver_fallback}%".format(**row)
        )

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with open(args.output_csv, "w") as f:
            f.write(",".join(fieldnames) + "\n")
            for row in rows:
                f.write(",".join(str(row[k]) for k in fieldnames) + "\n")
        print("Wrote {}".format(args.output_csv))


if __name__ == "__main__":
    main()
