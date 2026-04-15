"""
regenerate_figures.py
---------------------
Utility to regenerate experiment figures without rerunning training.

Usage:
    python -m app.regenerate_figures <experiment_id>
"""

import json
import sys
from pathlib import Path

from app.core.config import RESULTADOS_ROOT
from app.training.orchestrator import orchestrator


def _load_history(history_file: Path) -> list[dict]:
    lines = [line for line in history_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split(",")
    history: list[dict] = []
    for line in lines[1:]:
        cols = line.split(",")
        row = {header[i]: cols[i] for i in range(min(len(header), len(cols)))}
        history.append(
            {
                "epoch": int(row.get("epoch", 0)),
                "loss": float(row.get("loss", 0.0)),
                "accuracy": float(row.get("accuracy", 0.0)),
            }
        )
    return history


def regenerate_figures(experiment_id: str) -> dict:
    exp_dir = RESULTADOS_ROOT / experiment_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {experiment_id}")

    summary_file = exp_dir / "metrics" / "final_summary.json"
    history_file = exp_dir / "metrics" / "round_metrics.csv"
    if not summary_file.exists() or not history_file.exists():
        raise FileNotFoundError("Missing experiment metrics files")

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    dataset = summary.get("dataset")
    model = summary.get("model", "cnn_vae")
    if not dataset:
        raise ValueError("Dataset not found in summary")

    history = _load_history(history_file)
    orchestrator._save_figures(exp_dir, history, dataset, model)

    return {
        "status": "ok",
        "experiment_id": experiment_id,
        "dataset": dataset,
        "model": model,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.regenerate_figures <experiment_id>")
        sys.exit(1)
    payload = regenerate_figures(sys.argv[1])
    print(json.dumps(payload, indent=2))
