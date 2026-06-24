"""Load real-world country data and turn it into model inputs.

WHAT THIS IS
------------
A small, bundled snapshot (`data/countries_snapshot.csv`) of ~24 major states
with two real-world anchors from the literature review:

  * `cinc`        -- Composite Index of National Capability (Correlates of War,
                     National Material Capabilities v6.0). Used for each country's
                     `power`. Magnitudes are approximate, ~mid-2010s.
  * `ideal_point` -- UN General Assembly voting ideal point (Bailey, Strezhnev &
                     Voeten). A ~[-3, +3] axis of position toward the US-led
                     liberal order. Used for each country's `alignment`.
  * `bloc_hint`   -- a coarse West / East / NonAligned label, ONLY for eyeballing
                     whether the emergent blocs resemble reality. Not used as input.

IMPORTANT: the bundled numbers are *approximate*, hand-entered for a runnable
demo. For research, replace this CSV with the authoritative data:
  - CINC v6.0:        https://correlatesofwar.org/data-sets/national-material-capabilities/
  - UN ideal points:  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LEJUQZ
The loader below works on any CSV with the same column names, so swapping in the
real files (filtered to a year) is a drop-in change.

HOW IT MAPS TO THE MODEL
------------------------
  power     = cinc / mean(cinc)            # mean ~1, preserves real power ratios
  alignment = ideal_point / max(|ideal|)   # rescaled into [-1, 1] for the rules
"""

from __future__ import annotations

import csv
import os

import numpy as np

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "countries_snapshot.csv")


def load_snapshot(path: str | None = None) -> dict:
    """Read the snapshot CSV into a dict of parallel lists/arrays."""
    path = path or DEFAULT_PATH
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return {
        "names": [r["country"] for r in rows],
        "iso3": [r["iso3"] for r in rows],
        "region": [r["region"] for r in rows],
        "bloc_hint": [r["bloc_hint"] for r in rows],
        "cinc": np.array([float(r["cinc"]) for r in rows]),
        "ideal_point": np.array([float(r["ideal_point"]) for r in rows]),
    }


def to_model_inputs(snap: dict):
    """Return (power, alignment) arrays ready for GeoModel.

    power preserves real capability ratios (mean-normalized); alignment is
    rescaled from the ideal-point axis into the model's [-1, 1] convention.
    """
    cinc = snap["cinc"]
    ideal = snap["ideal_point"]
    power = cinc / cinc.mean()
    alignment = ideal / np.max(np.abs(ideal))
    return power, alignment
