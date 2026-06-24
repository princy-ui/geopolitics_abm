# Results plot notes

The cooperative scenario quickly reaches nearly complete cooperation, while the neutral scenario rises more gradually and ends a little above 0.9; the rivalrous scenario stays near zero. The cooperative and neutral scenarios both end with one bloc, while the rivalrous scenario ends with the most blocs, at two. The cooperative scenario has no conflicts, the neutral scenario has about two in total, and the rivalrous scenario has recurring conflicts—about 28 in total. Polarization reaches 1 in the cooperative and neutral scenarios but remains near zero in the rivalrous scenario.

## Cooperation bias experiment

Across seeds `7`, `21`, `42`, `60`, and `90`, total conflicts fall sharply as `cooperation_bias` increases: very negative settings average more than 100 conflicts, neutral settings average about 3, and the strongest positive settings produce none. Final polarization is steady at `1.00` once the bias is neutral or positive, which means the model forms strong relationships; in the positive-bias runs, those strong relationships are mostly cooperative rather than hostile. The conflict trend is fairly steady across seeds, although the negative-bias worlds vary more because random starting conditions and war rolls matter more when the model is already rivalry-prone.

## Real-world model: emergent blocs

After 100 steps, the countries formed these blocs:

- **Bloc 1:** Brazil, China, Egypt, India, Indonesia, Iran, Nigeria, North Korea, Pakistan, Russia, Saudi Arabia, South Africa, Turkey, and Ukraine.
- **Bloc 2:** Australia, Canada, France, Germany, Israel, Italy, Japan, South Korea, the United Kingdom, and the United States.

Turkey's placement in Bloc 1 was unexpected because the snapshot labels it as Western. Its `ideal_point` is only `0.30`, so the model reads Turkey as much less strongly aligned with the Western side than countries with larger positive values, which may have pulled it into the East/Non-Aligned bloc.
