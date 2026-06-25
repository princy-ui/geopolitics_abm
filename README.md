# A geopolitical agent-based model (Mesa-based)

Countries are **agents on a signed relationship network**. A few rules taken
straight from the geopolitical-network literature drive how the network rewires,
and **blocs, polarization, and conflict emerge** — none of them are hard-coded.

This is the Python/Mesa path, and it runs forward in time so it can later be 
coupled to an emissions model.

## What's here

```
geopolitics_abm/
├── geo_abm/
│   ├── rules.py        # the geopolitical rules as small, testable NumPy functions
│   ├── model.py        # the Mesa model: Country agents + signed-network dynamics
│   ├── metrics.py      # how to MEASURE emergence (blocs, balance, polarization)
│   └── realworld.py    # load real CINC + UN-ideal-point data into model inputs
├── data/
│   └── countries_snapshot.csv   # ~24 real countries (approx. CINC + ideal points)
├── run.py              # cooperative/neutral/rivalrous scenarios -> results.csv + .png
├── run_realworld.py    # seed from real data; print + draw the emergent blocs
├── tests/test_basic.py
├── requirements.txt
├── results.png             # sample scenario output
├── realworld_network.png   # sample real-data alliance network
└── README.md
```

## Run it

```bash
pip install -r requirements.txt        # mesa, networkx, numpy, matplotlib, pandas
python3 tests/test_basic.py            # all 6 tests should pass
python3 run.py                         # writes results.csv + results.png
python3 run.py --steps 200 --countries 18
```

```python
from geo_abm import GeoModel
df = GeoModel(n_countries=14, cooperation_bias=0.2, seed=42).run(steps=120)
print(df.tail())   # mean_cooperation, n_blocs, polarization, balanced_fraction, conflicts, ...
```

## Seeding from real-world data

Instead of random countries, initialize from real data — capability and alignment
straight out of the literature review's datasets:

```bash
python3 run_realworld.py        # writes realworld_network.png
```
```python
from geo_abm import GeoModel
model = GeoModel.from_real_data(influence=0.0, cooperation_bias=-0.15, seed=42)
model.run(steps=100)
for bloc in model.bloc_membership():
    print(bloc)
```

- **`power` ← CINC** (Correlates of War National Material Capabilities): so China
  and the US start as the largest nodes, ratios preserved.
- **`alignment` ← UN ideal points** (Bailey-Strezhnev-Voeten): each country's real
  position on the pro-/anti-US-order axis, rescaled to [-1, 1].

With ideological drift turned off (`influence=0`, holding alignments at their real
values) the model **recovers a recognizable Western bloc** (US, UK, France,
Germany, Italy, Canada, Australia, Japan, South Korea, Israel) opposite an
East/Non-Aligned grouping — emerging from the data, not hand-assigned. Borderline
cases are instructive: Turkey and Ukraine drift toward the non-Western side because
their *measured* ideal points sit only mildly positive.

**The bundled CSV numbers are approximate** (hand-entered for a runnable demo). For
research, drop in the authoritative files — the loader takes any CSV with the same
columns:
- CINC v6.0 — https://correlatesofwar.org/data-sets/national-material-capabilities/
- UN ideal points — https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LEJUQZ

## The rules, and where each comes from

| In the code | Rule | Literature |
|-------------|------|-----------|
| `rules.homophily_target` | similar ideology → ally, opposite → rival | homophily / nodematch |
| `rules.balance_pressure` | "enemy of my enemy is my friend" (signed triads) | Heider balance; Antal-Krapivsky-Redner |
| `rules.power_parity` + `model._resolve_conflicts` | war risk peaks near power parity | power-transition theory (Organski & Kugler) |
| `rules.gravity_weights` / growth in `Country.step` | bigger partners matter more | gravity models of interaction |
| ideological drift in `Country.step` | states adopt their allies' positions | social influence → bloc consolidation |
| `cooperation_bias` lever | global cooperative vs. rivalrous climate | SSP-style scenario knob |

## What emerges (try it and see)

With the default rules, the **cooperative** and **neutral** worlds self-organize
into a single cohesive bloc with little conflict, while the **rivalrous** world
freezes into ~2 hostile camps with recurring wars near power parity. That split —
"one camp vs. two camps" — is the classic result of structural-balance dynamics,
reproduced here from the bottom up. The point of the project is to interrogate
and improve these rules, not to trust them.

## A 5-day sprint for this week

1. **Day 1 — Run & read.** Get it running; read `rules.py` and `model.py`; in a
   notebook, plot `mean_cooperation` and `n_blocs` for one scenario. Write two
   sentences on what "emergence" means here.
2. **Day 2 — Poke the rules.** Change `w_balance` vs `w_homophily` and the
   `learning_rate`; find a setting that produces 3 stable blocs. Plot it.
3. **Day 3 — Add a visualization.** Draw the signed network with NetworkX
   (green = ally, red = rival) at steps 0, 40, 120 to *see* blocs form.
4. **Day 4 — A real experiment.** Sweep `cooperation_bias` from −0.5 to +0.5
   (say 11 values, 5 seeds each); plot final polarization and total conflicts vs.
   bias. This is her first genuine result.
5. **Day 5 — Write it up.** A short notebook: question, method, figures, what she
   found, what's unsatisfying about the model (the seed of next week's work).

## Where it goes next (extension hooks)

- **Ground it in data.** Initialize `alignment` from UN-voting ideal points and
  `power` from CINC; compare emergent blocs to real ones (Cold-War, post-1991).
- **Make conflict richer.** Add alliances pulling members into each other's wars;
  add a "dissatisfaction" term to the power-transition rule.
- **Couple to emissions (later).** Each country already has a `power`/output
  signal; map it to emissions and feed a simple climate model (e.g., the `fair`
  package) — the society→emissions wiring, now with *real* geopolitics behind it.
- **Migrate metrics.** Swap the greedy-modularity bloc detector for the multiplex
  / Kantian-fractionalization measure from the review.

## A note on rigor

Fixed seeds make every run reproducible. Each rule lives in its own tested
function. When you change a rule, watch the magnitudes — an early version of this
model had a bias that did nothing because it was applied to the wrong quantity
(homophily depends on alignment *gaps*, which a uniform shift can't change). That
kind of bug is exactly what the tests and sanity-checks are there to catch.
