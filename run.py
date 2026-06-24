"""Run the geopolitical ABM across a few scenarios and plot what emerges.

    python run.py
    python run.py --steps 200 --countries 18

Compares a cooperative vs. rivalrous world and shows the emergent structure:
bloc count, polarization, structural balance, and conflict over time.
"""

import argparse

from geo_abm import GeoModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--countries", type=int, default=14)
    args = ap.parse_args()

    scenarios = {
        "cooperative": 0.30,
        "neutral": 0.0,
        "rivalrous": -0.30,
    }

    results = {}
    for name, bias in scenarios.items():
        model = GeoModel(n_countries=args.countries, cooperation_bias=bias, seed=42)
        results[name] = model.run(steps=args.steps)

    # text summary
    print(f"\n{'scenario':12s} {'mean_coop':>10s} {'n_blocs':>8s} {'polariz.':>9s} "
          f"{'balanced':>9s} {'conflicts(tot)':>14s}")
    for name, df in results.items():
        print(f"{name:12s} {df['mean_cooperation'].iloc[-1]:10.2f} "
              f"{df['n_blocs'].iloc[-1]:8.0f} {df['polarization'].iloc[-1]:9.2f} "
              f"{df['balanced_fraction'].iloc[-1]:9.2f} {int(df['conflicts'].sum()):14d}")

    # combined CSV
    import pandas as pd
    combined = pd.concat({k: v for k, v in results.items()}, names=["scenario", "step"])
    combined.to_csv("results.csv")
    print("\nWrote results.csv")

    # optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        panels = ["mean_cooperation", "n_blocs", "polarization", "conflicts"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, col in zip(axes.flat, panels):
            for name, df in results.items():
                series = df[col].rolling(5).mean() if col == "conflicts" else df[col]
                ax.plot(df.index, series, label=name)
            ax.set_title(col.replace("_", " "))
            ax.set_xlabel("step")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle("Emergent geopolitical structure by scenario", fontsize=13)
        fig.tight_layout()
        fig.savefig("results.png", dpi=120)
        print("Wrote results.png")
    except ImportError:
        print("(matplotlib not installed -- skipped plot)")


if __name__ == "__main__":
    main()
