import numpy as np
import pandas as pd
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo_abm import GeoModel


biases = np.linspace(-0.5, 0.5, 11)
seeds = [7, 21, 42, 60, 90]
rows = []

for seed in seeds:
    for bias in biases:
        df = GeoModel(n_countries=16, cooperation_bias=bias, seed=seed).run(steps=120)
        rows.append(
            {
                "seed": seed,
                "cooperation_bias": bias,
                "final_polarization": df["polarization"].iloc[-1],
                "total_conflicts": int(df["conflicts"].sum()),
            }
        )

results = pd.DataFrame(rows)
summary = (
    results.groupby("cooperation_bias")
    .agg(
        final_polarization_mean=("final_polarization", "mean"),
        final_polarization_min=("final_polarization", "min"),
        final_polarization_max=("final_polarization", "max"),
        total_conflicts_mean=("total_conflicts", "mean"),
        total_conflicts_min=("total_conflicts", "min"),
        total_conflicts_max=("total_conflicts", "max"),
    )
    .reset_index()
)

print(summary.round(2).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

axes[0].plot(
    summary["cooperation_bias"],
    summary["final_polarization_mean"],
    marker="o",
)
axes[0].fill_between(
    summary["cooperation_bias"],
    summary["final_polarization_min"],
    summary["final_polarization_max"],
    alpha=0.2,
)
axes[0].set_title("Final polarization")
axes[0].set_xlabel("cooperation_bias")
axes[0].set_ylabel("polarization")
axes[0].set_ylim(-0.05, 1.05)

axes[1].plot(
    summary["cooperation_bias"],
    summary["total_conflicts_mean"],
    marker="o",
    color="firebrick",
)
axes[1].fill_between(
    summary["cooperation_bias"],
    summary["total_conflicts_min"],
    summary["total_conflicts_max"],
    color="firebrick",
    alpha=0.2,
)
axes[1].set_title("Total conflicts")
axes[1].set_xlabel("cooperation_bias")
axes[1].set_ylabel("conflicts over 120 steps")

fig.suptitle("Cooperation bias experiment across 5 seeds")
fig.tight_layout()
fig.savefig("cooperation_bias_experiment.png", dpi=150)

print("\nSaved plot to cooperation_bias_experiment.png")
