"""Combine the metrics of every run into one table (mean ± spread over seeds).

    python -m captioning.summarize --runs-dir runs --split test --beam-size 3
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--beam-size", type=int, default=3)
    args = parser.parse_args()

    by_model = defaultdict(list)
    for path in sorted(Path(args.runs_dir).glob(f"*/metrics_{args.split}_beam{args.beam_size}.json")):
        metrics = json.loads(path.read_text())
        by_model[metrics["model"]].append(metrics)

    columns = [m for m in METRICS if any(m in r for runs in by_model.values() for r in runs)]
    print("| Model | Seeds | " + " | ".join(columns) + " |")
    print("|---|---|" + "---|" * len(columns))
    for model, runs in sorted(by_model.items()):
        cells = []
        for metric in columns:
            values = [r[metric] for r in runs if metric in r]
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{statistics.mean(values):.3f} ± {spread:.3f}")
        print(f"| {model} | {len(runs)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
