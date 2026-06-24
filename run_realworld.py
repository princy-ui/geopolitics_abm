"""Initialize the ABM from real-world data and see what blocs emerge.

    python run_realworld.py

Seeds power from CINC and alignment from UN ideal points (see geo_abm/realworld.py),
runs the model forward, prints the emergent blocs next to each country's real-world
bloc hint, and draws the signed network (green = ally, red = rival).
"""

from geo_abm import GeoModel


def main(steps: int = 500):
    # Hold alignments at their real (data) values by turning OFF ideological drift
    # (influence=0); a mild rivalrous bias keeps the world from collapsing into one
    # universal bloc, so real cleavages can express themselves.
    model = GeoModel.from_real_data(cooperation_bias=-0.15, influence=0.0, seed=42)
    model.run(steps=steps)

    snap = model.snapshot
    hint = dict(zip(snap["names"], snap["bloc_hint"]))

    print(f"\nEmergent blocs after {steps} steps (real-world hint in parentheses):\n")
    for k, members in enumerate(model.bloc_membership(), 1):
        print(f"  Bloc {k}:")
        for name in members:
            print(f"      {name:18s} ({hint[name]})")

    # network figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np

        g = nx.Graph()
        g.add_nodes_from(range(model.n_countries))
        edges_pos, edges_neg = [], []
        for i in range(model.n_countries):
            for j in range(i + 1, model.n_countries):
                w = model.ties[i, j]
                if w > 0.25:
                    edges_pos.append((i, j, w))
                elif w < -0.25:
                    edges_neg.append((i, j, w))
        g.add_weighted_edges_from([(i, j, w) for i, j, w in edges_pos])
        pos = nx.spring_layout(g, seed=1, weight="weight", k=0.9)

        sizes = 300 + 6000 * (model.power / model.power.sum())
        colors = ["#1f77b4" if a > 0.15 else "#d62728" if a < -0.15 else "#7f7f7f"
                  for a in model.alignment]  # blue=West-leaning, red=East-leaning, grey=neutral

        plt.figure(figsize=(12, 9))
        nx.draw_networkx_edges(g, pos, edgelist=[(i, j) for i, j, _ in edges_pos],
                               edge_color="#2ca02c", width=[2.5 * w for _, _, w in edges_pos], alpha=0.5)
        nx.draw_networkx_edges(g, pos, edgelist=[(i, j) for i, j, _ in edges_neg],
                               edge_color="#d62728", width=[1.5 * -w for _, _, w in edges_neg],
                               style="dashed", alpha=0.35)
        nx.draw_networkx_nodes(g, pos, node_size=sizes, node_color=colors, alpha=0.9)
        nx.draw_networkx_labels(g, pos, labels={i: model.names[i] for i in range(model.n_countries)},
                                font_size=8)
        plt.title("Emergent alliance network from CINC + UN-ideal-point seeding\n"
                  "(node size = power; green solid = alliance, red dashed = rivalry; "
                  "blue=West-leaning, red=East-leaning)", fontsize=11)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("realworld_network.png", dpi=120)
        print("\nWrote realworld_network.png")
    except ImportError:
        print("\n(matplotlib/networkx not installed -- skipped network figure)")


if __name__ == "__main__":
    main()
