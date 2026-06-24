"""Emergence metrics: how to *measure* the patterns the model produces.

These read the model's signed tie matrix and report system-level structure --
the things your literature review calls 'emergent': blocs, polarization,
structural balance, and conflict. Kept separate so they're easy to test and extend.
"""

from __future__ import annotations

import itertools

import numpy as np
import networkx as nx


def mean_cooperation(ties: np.ndarray) -> float:
    """Average signed tie across all pairs. >0 = broadly cooperative system."""
    n = ties.shape[0]
    if n < 2:
        return 0.0
    return float(ties[np.triu_indices(n, k=1)].mean())


def balanced_triad_fraction(ties: np.ndarray, threshold: float = 0.0) -> float:
    """Fraction of triangles that are structurally balanced.

    A triad is balanced if the product of its three (signed) ties is positive
    ('the enemy of my enemy is my friend'). Near 1.0 = a tidy, balanced world
    (often two camps); lower = a tense, frustrated system.
    """
    n = ties.shape[0]
    s = np.sign(ties - 0)  # +/-/0
    bal, total = 0, 0
    for i, j, k in itertools.combinations(range(n), 3):
        prod = s[i, j] * s[j, k] * s[i, k]
        if prod == 0:
            continue
        total += 1
        if prod > 0:
            bal += 1
    return bal / total if total else 1.0


def _positive_graph(ties: np.ndarray, threshold: float = 0.1) -> nx.Graph:
    g = nx.Graph()
    n = ties.shape[0]
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if ties[i, j] > threshold:
                g.add_edge(i, j, weight=ties[i, j])
    return g


def blocs(ties: np.ndarray, threshold: float = 0.1):
    """Detect blocs as communities in the positive-tie graph (greedy modularity).

    Returns (n_blocs, modularity). More/sharper blocs => more polarized system.
    (Review: community detection / Kantian fractionalization.)
    """
    g = _positive_graph(ties, threshold)
    if g.number_of_edges() == 0:
        return g.number_of_nodes(), 0.0
    comms = list(nx.community.greedy_modularity_communities(g, weight="weight"))
    try:
        mod = nx.community.modularity(g, comms, weight="weight")
    except Exception:
        mod = 0.0
    return len(comms), float(mod)


def polarization(ties: np.ndarray) -> float:
    """A simple 0..1 polarization score: how bimodal the tie distribution is.

    Defined as the share of ties that are strongly signed (|tie| > 0.5). A
    polarized system has many firm friends and firm enemies, few neutrals.
    """
    n = ties.shape[0]
    if n < 2:
        return 0.0
    off = ties[np.triu_indices(n, k=1)]
    return float(np.mean(np.abs(off) > 0.5))
