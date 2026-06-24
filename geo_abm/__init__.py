"""A Mesa-based geopolitical agent-based model.

Countries are agents on a signed relationship network whose rewiring follows
literature-grounded rules (homophily, structural balance, gravity, power
transition). Blocs, polarization, and conflict emerge.

    from geo_abm import GeoModel
    df = GeoModel(n_countries=14, cooperation_bias=0.2).run(steps=120)
"""

from .model import GeoModel, Country
from . import rules, metrics

__all__ = ["GeoModel", "Country", "rules", "metrics"]
