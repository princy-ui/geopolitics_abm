from geo_abm import GeoModel
df = GeoModel(n_countries=14, cooperation_bias=0.4, seed=42).run(steps=120)
print(df.tail())
