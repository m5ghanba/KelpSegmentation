"""
scripts/compile_ablation_tables.py
------------------------------------
Reads the MLflow runs produced by scripts/run_ablation_suite.py (tagged
`ablation_group` / `variant` by that script, plus `training_seed` /
`data_random_seed` auto-tagged by src/train.py) and compiles one
mean ± std table per group — i.e. the actual tables for the manuscript
(indices, substrate encoding, augmentation, loss, pretrained-vs-scratch).

Usage
-----
    python scripts/compile_ablation_tables.py \\
        --tracking-uri mlruns \\
        --experiment-name KelpSegmentation-Ablation \\
        --out-dir outputs/ablation_tables
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List

import mlflow
import pandas as pd

METRIC_KEYS = [
    "test_kelp_dataset_iou", "test_kelp_precision", "test_kelp_recall", "test_kelp_f1",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--experiment-name", default="KelpSegmentation-Ablation")
    parser.add_argument("--out-dir", default="outputs/ablation_tables")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named {args.experiment_name!r} "
                         f"at tracking_uri={args.tracking_uri!r}.")

    runs = client.search_runs([experiment.experiment_id], max_results=5000)
    print(f"Found {len(runs)} runs in experiment {args.experiment_name!r}.")

    by_group: Dict[str, List] = defaultdict(list)
    skipped = 0
    for run in runs:
        tags = run.data.tags
        group = tags.get("ablation_group")
        variant = tags.get("variant")
        if group is None or variant is None:
            skipped += 1
            continue

        row = {"group": group, "variant": variant,
               "seed": tags.get("training_seed", "?"),
               "run_id": run.info.run_id}
        missing_metric = False
        for key in METRIC_KEYS:
            val = run.data.metrics.get(key)
            if val is None:
                missing_metric = True
            row[key] = val
        if missing_metric:
            print(f"  [WARN] run {run.info.run_id} ({group}/{variant}, seed={row['seed']}) "
                  f"is missing one or more metrics — likely still running or crashed. Included as-is.")
        by_group[group].append(row)

    if skipped:
        print(f"  ({skipped} runs had no ablation_group/variant tag — not from "
              f"run_ablation_suite.py, or predate this tagging; skipped.)")

    os.makedirs(args.out_dir, exist_ok=True)
    for group, rows in by_group.items():
        df = pd.DataFrame(rows)
        summary = (
            df.groupby("variant")[METRIC_KEYS]
              .agg(["mean", "std", "count"])
        )
        csv_path = os.path.join(args.out_dir, f"{group}.csv")
        md_path  = os.path.join(args.out_dir, f"{group}.md")
        summary.to_csv(csv_path)
        with open(md_path, "w") as f:
            f.write(f"# Ablation group: {group}\n\n")
            f.write(summary.to_markdown())
            f.write("\n")
        print(f"\n=== {group} ===")
        print(summary)
        print(f"  -> {csv_path}\n  -> {md_path}")


if __name__ == "__main__":
    main()
