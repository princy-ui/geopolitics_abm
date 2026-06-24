"""A geopolitical agent-based model, built on Mesa 3.

Countries are agents on a *signed relationship network*. The authoritative state
(power, alignment, ties) lives in NumPy arrays on the Model so the rules can be
vectorized and tested; each Country agent is a thin Mesa wrapper that reads and
writes its own slice. This keeps the code idiomatic Mesa AND fast/testable.

The dynamics each step:
  1. (model)  relationships rewire   -> rules.update_ties  (homophily + balance)
  2. (model)  conflicts may erupt    -> power-transition rule near parity
  3. (agents) each country updates    -> economic growth (gravity-weighted allies)
              its power and drifts     and ideological drift toward its allies
              ideologically

Out of these few rules, blocs, polarization, and episodic conflict emerge --
none of them are hard-coded.
"""

from __future__ import annotations

import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector

from . import rules
from . import metrics


class Country(Agent):
    """One state. Mesa assigns unique_id automatically in Mesa 3."""

    def __init__(self, model, idx: int):
        super().__init__(model)
        self.idx = idx  # row/column in the model's arrays

    # convenient views into the model's arrays
    @property
    def power(self):
        return self.model.power[self.idx]

    @property
    def alignment(self):
        return self.model.alignment[self.idx]

    def step(self):
        """Economic growth + ideological drift for this country."""
        m = self.model
        i = self.idx
        ties_i = m.ties[i]

        # --- economic growth: gravity-weighted benefit of cooperative ties ---
        coop = np.clip(ties_i, 0.0, None)               # only allies help
        power_share = m.power / m.power.sum()
        benefit = float(np.sum(coop * power_share))      # gravity via partner size
        growth = m.base_growth + m.coop_gain * benefit - m.war_damage[i]
        growth = float(np.clip(growth, -0.15, 0.10))
        m.power[i] = max(0.02, m.power[i] * (1.0 + growth))
        m.war_damage[i] = 0.0

        # --- ideological drift toward allies (consolidates blocs) ---
        if coop.sum() > 1e-9:
            target = float(np.sum(coop * m.alignment) / coop.sum())
            m.alignment[i] += m.influence * (target - m.alignment[i])
        # small idiosyncratic drift keeps things alive
        m.alignment[i] += m.rng_np.uniform(-0.02, 0.02)
        m.alignment[i] = float(np.clip(m.alignment[i], -1.0, 1.0))


class GeoModel(Model):
    def __init__(
        self,
        n_countries: int = 14,
        cooperation_bias: float = 0.0,   # global lever: + cooperative world, - rivalrous
        base_growth: float = 0.015,
        coop_gain: float = 0.05,
        influence: float = 0.10,         # how strongly states adopt allies' ideology
        conflict_base: float = 0.05,     # baseline propensity for war near parity
        seed: int | None = 42,
        init_power=None,                 # optional real data: per-country capability
        init_alignment=None,             # optional real data: per-country ideology in [-1,1]
        names=None,                      # optional country names
    ):
        super().__init__(seed=seed)
        if init_alignment is not None:
            init_alignment = np.asarray(init_alignment, dtype=float)
            n_countries = len(init_alignment)
        elif init_power is not None:
            n_countries = len(np.asarray(init_power))

        self.n_countries = n_countries
        self.cooperation_bias = cooperation_bias
        self.base_growth = base_growth
        self.coop_gain = coop_gain
        self.influence = influence
        self.conflict_base = conflict_base
        self.rng_np = np.random.default_rng(seed)
        self.conflicts_this_step = 0
        self.names = list(names) if names is not None else [f"C{i}" for i in range(n_countries)]

        # --- authoritative state as arrays (real data if provided, else random) ---
        if init_power is not None:
            self.power = np.asarray(init_power, dtype=float).copy()
        else:
            self.power = self.rng_np.uniform(0.5, 2.5, size=n_countries)
        if init_alignment is not None:
            self.alignment = init_alignment.copy()
        else:
            self.alignment = self.rng_np.uniform(-1.0, 1.0, size=n_countries)
        self.war_damage = np.zeros(n_countries)
        # initial ties: weak homophily + noise
        self.ties = rules.homophily_target(self.alignment) * 0.2
        self.ties += self.rng_np.uniform(-0.1, 0.1, size=(n_countries, n_countries))
        self.ties = np.clip(0.5 * (self.ties + self.ties.T), -1, 1)
        np.fill_diagonal(self.ties, 0.0)

        # --- agents ---
        for i in range(n_countries):
            Country(self, i)

        self.datacollector = DataCollector(
            model_reporters={
                "mean_cooperation": lambda m: metrics.mean_cooperation(m.ties),
                "balanced_fraction": lambda m: metrics.balanced_triad_fraction(m.ties),
                "n_blocs": lambda m: metrics.blocs(m.ties)[0],
                "modularity": lambda m: metrics.blocs(m.ties)[1],
                "polarization": lambda m: metrics.polarization(m.ties),
                "conflicts": lambda m: m.conflicts_this_step,
                "total_power": lambda m: float(m.power.sum()),
            }
        )

    # ---- model-level dynamics ----
    def _rewire(self):
        self.ties = rules.update_ties(
            self.ties,
            self.alignment,
            w_homophily=0.6,
            w_balance=0.4,
            learning_rate=0.15,
            bias=self.cooperation_bias,   # global cooperation lever (acts on tie target)
        )

    def _resolve_conflicts(self):
        """Rival dyads near power parity may go to war (power-transition rule)."""
        self.conflicts_this_step = 0
        parity = rules.power_parity(self.power)
        n = self.n_countries
        for i in range(n):
            for j in range(i + 1, n):
                rivalry = max(0.0, -self.ties[i, j])      # only rivals
                if rivalry <= 0:
                    continue
                p_war = self.conflict_base * rivalry * parity[i, j]
                if self.rng_np.random() < p_war:
                    self.conflicts_this_step += 1
                    # both lose; the weaker loses relatively more
                    self.war_damage[i] += 0.06
                    self.war_damage[j] += 0.06
                    # war hardens the rivalry
                    self.ties[i, j] = self.ties[j, i] = max(-1.0, self.ties[i, j] - 0.2)

    def step(self):
        self.datacollector.collect(self)   # record state at start of step
        self._rewire()                     # 1. relationships adjust
        self._resolve_conflicts()          # 2. wars near parity
        self.agents.shuffle_do("step")     # 3. each country grows & drifts

    def run(self, steps: int = 120):
        for _ in range(steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()

    # ---- real-world initialization ----
    @classmethod
    def from_real_data(cls, path=None, **kwargs):
        """Build a model seeded from the bundled real-world snapshot (CINC + UN
        ideal points). Pass `path` to use your own CSV. See geo_abm/realworld.py."""
        from .realworld import load_snapshot, to_model_inputs

        snap = load_snapshot(path)
        power, alignment = to_model_inputs(snap)
        model = cls(init_power=power, init_alignment=alignment, names=snap["names"], **kwargs)
        model.snapshot = snap
        return model

    def bloc_membership(self, threshold: float = 0.1):
        """Return the emergent blocs as lists of country *names* (positive-tie
        communities). Handy for eyeballing whether they resemble real blocs."""
        import networkx as nx

        g = metrics._positive_graph(self.ties, threshold)
        if g.number_of_edges() == 0:
            return [[self.names[i]] for i in range(self.n_countries)]
        comms = nx.community.greedy_modularity_communities(g, weight="weight")
        return [sorted(self.names[i] for i in c) for c in comms]

if __name__ == "__main__":
    df = GeoModel(n_countries=14, cooperation_bias=0.0, seed=42).run(steps=120)
    print(df.tail())
