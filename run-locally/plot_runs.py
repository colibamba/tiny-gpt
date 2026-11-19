"""
Plot the loss curves of your training runs, to compare experiments.

    py plot_runs.py                      # every run in runs/
    py plot_runs.py default big tiny     # only these runs

Solid line = validation loss, dashed = training loss, same colour per run.
If a run's dashed line keeps falling while its solid line turns upwards,
that run is overfitting.

Saves the chart as loss_curves.png and opens it.
"""

import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

runs = sys.argv[1:] or sorted(p.name for p in Path("runs").iterdir() if (p / "log.csv").exists())
if not runs:
    raise SystemExit("No runs found. Train one first with step3_gpt.py")

plt.figure(figsize=(9, 5))
for i, name in enumerate(runs):
    rows = list(csv.DictReader(open(Path("runs") / name / "log.csv")))
    steps = [int(r["step"]) for r in rows]
    colour = f"C{i}"
    plt.plot(steps, [float(r["val_loss"]) for r in rows], color=colour, label=f"{name} (val)")
    plt.plot(steps, [float(r["train_loss"]) for r in rows], color=colour, linestyle="--", alpha=0.6)

plt.axhline(2.5, color="grey", linestyle=":", label="bigram (≈2.5)")
plt.xlabel("training step")
plt.ylabel("loss (lower is better)")
plt.title("Loss curves (solid = validation, dashed = training)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=120)
print("Saved loss_curves.png")
if sys.platform == "win32":
    os.startfile("loss_curves.png")
