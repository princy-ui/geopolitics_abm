"""The geopolitical 'rules', as small, testable, vectorized functions.

Each function maps onto a mechanism documented in the literature review. They
operate on plain NumPy arrays so they can be unit-tested in isolation, with no
dependence on Mesa. The Mesa model (model.py) just calls them.

Conventions
-----------
* `ties`       : N x N signed matrix in [-1, 1]. +1 = strong ally, -1 = strong
                 rival, 0 = indifferent. Symmetric, zero diagonal.
* `alignment`  : length-N vector in [-1, 1]. A 1-D ideological / strategic axis.
* `power`      : length-N positive vector. A capability score (think CINC).
"""

from __future__ import annotations

import numpy as np


def homophily_target(alignment: np.ndarray) -> np.ndarray:
    """Desired tie sign from ideological similarity (HOMOPHILY).

    Similar alignments 'want' a positive tie; opposite alignments a negative one.
    Returns an N x N matrix in [-1, 1].  (Review: homophily / nodematch terms.)
    """
    diff = np.abs(alignment[:, None] - alignment[None, :])  # 0 (identical) .. 2 (opposite)
    target = 1.0 - diff                                     # +1 .. -1
    np.fill_diagonal(target, 0.0)
    return target


def balance_pressure(ties: np.ndarray) -> np.ndarray:
    """Structural-balance pressure on each tie (HEIDER / 'enemy of my enemy').

    For pair (i, j), look at every third country k: if i and j relate to k the
    same way (both friends or both enemies of k), the product tie_ik * tie_kj is
    positive and pushes i, j toward friendship; if they relate oppositely it
    pushes them toward rivalry. Summing over k and normalizing gives the pressure.
    (Review: Antal-Krapivsky-Redner balance dynamics; signed ERGM balance terms.)
    """
    n = ties.shape[0]
    P = ties @ ties                 # P[i, j] = sum_k tie_ik * tie_kj
    np.fill_diagonal(P, 0.0)
    return P / max(1, n - 2)        # average over the n-2 possible third parties


def power_parity(power: np.ndarray) -> np.ndarray:
    """Parity matrix in [0, 1]: 1.0 when two states are equally powerful.

    Power-transition theory says war risk peaks near parity, so this feeds the
    conflict rule. (Review: Organski & Kugler power-transition.)
    """
    pi, pj = power[:, None], power[None, :]
    mn = np.minimum(pi, pj)
    mx = np.maximum(pi, pj) + 1e-12
    parity = mn / mx
    np.fill_diagonal(parity, 0.0)
    return parity


def gravity_weights(power: np.ndarray, distance: np.ndarray | None = None) -> np.ndarray:
    """Interaction intensity ~ product of masses / distance^2 (GRAVITY model).

    Bigger states interact more; (optional) distance damps it. Used to weight how
    much a partner matters economically. (Review: gravity models of trade/conflict.)
    """
    g = power[:, None] * power[None, :]
    if distance is not None:
        g = g / (distance ** 2 + 1e-9)
    np.fill_diagonal(g, 0.0)
    return g


def update_ties(ties, alignment, *, w_homophily=0.6, w_balance=0.4, learning_rate=0.15, bias=0.0):
    """One step of relationship dynamics: blend homophily + balance, relax toward it.

    `bias` is a global cooperation lever added to every tie target: positive nudges
    the whole world toward cooperation, negative toward rivalry. (Note: it must act
    on the tie target, not on alignments -- homophily depends on alignment *gaps*,
    which a uniform shift leaves unchanged.)

    Returns a new (symmetric, clipped) ties matrix. The learning_rate controls how
    fast relationships adjust. This single function is where blocs emerge.
    """
    target = w_homophily * homophily_target(alignment) + w_balance * balance_pressure(ties) + bias
    target = np.clip(target, -1.0, 1.0)
    new = ties + learning_rate * (target - ties)
    new = np.clip(new, -1.0, 1.0)
    new = 0.5 * (new + new.T)        # enforce symmetry
    np.fill_diagonal(new, 0.0)
    return new
