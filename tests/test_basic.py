"""Per-module tests. Run:  python tests/test_basic.py   (or pytest)"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from geo_abm import GeoModel
from geo_abm import rules, metrics


def test_homophily_sign():
    align = np.array([1.0, 1.0, -1.0])
    t = rules.homophily_target(align)
    assert t[0, 1] > 0.9          # identical -> allies
    assert t[0, 2] < -0.9         # opposite -> rivals


def test_balance_pressure_enemy_of_enemy():
    # i-k rival, j-k rival  => i,j should feel positive (balanced) pressure
    ties = np.array([[0, 0, -1.0],
                     [0, 0, -1.0],
                     [-1.0, -1.0, 0]])
    P = rules.balance_pressure(ties)
    assert P[0, 1] > 0


def test_update_ties_is_symmetric_and_bounded():
    align = np.linspace(-1, 1, 6)
    ties = np.zeros((6, 6))
    new = rules.update_ties(ties, align)
    assert np.allclose(new, new.T)
    assert new.max() <= 1.0 and new.min() >= -1.0
    assert np.allclose(np.diag(new), 0.0)


def test_balanced_fraction_perfectly_balanced():
    # two blocs: {0,1} friends, {2,3} friends, cross ties hostile -> all balanced
    t = np.array([[0, 1, -1, -1],
                  [1, 0, -1, -1],
                  [-1, -1, 0, 1],
                  [-1, -1, 1, 0.0]])
    assert metrics.balanced_triad_fraction(t) == 1.0


def test_model_runs_and_collects():
    df = GeoModel(n_countries=12, seed=1).run(steps=30)
    assert len(df) == 30
    for col in ["mean_cooperation", "n_blocs", "polarization", "conflicts"]:
        assert col in df.columns


def test_cooperative_world_more_cooperative():
    coop = GeoModel(n_countries=14, cooperation_bias=0.4, seed=3).run(60)
    rival = GeoModel(n_countries=14, cooperation_bias=-0.4, seed=3).run(60)
    assert coop["mean_cooperation"].iloc[-1] > rival["mean_cooperation"].iloc[-1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All tests passed.")
