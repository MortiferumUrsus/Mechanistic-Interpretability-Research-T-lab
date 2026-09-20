# Feature distribution of the (arm - baseline) effect (scored_expA_r3.csv, baseline = naive)

Anti-steerability check: the fraction of features whose paired delta has the sign opposite the group mean, at fixed c, plus the min/median/max of the delta across features.

| arm | c | n_features | mean_delta_logppl | frac_opposite_logppl | min_delta_logppl | median_delta_logppl | max_delta_logppl | mean_delta_keyword_hit | frac_opposite_keyword_hit | min_delta_keyword_hit | median_delta_keyword_hit | max_delta_keyword_hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antimanifold | 1 | 12 | -0.3546 | 0.0833 | -0.7118 | -0.4109 | 0.3124 | 0.0167 | 0.6667 | -0.1667 | -0.0333 | 0.4333 |
| antimanifold | 1.5 | 12 | -0.3053 | 0.1667 | -1.01 | -0.4043 | 0.7874 | 0.0917 | 0.1667 | -0.2 | 0.1 | 0.3667 |
| antimanifold | 2 | 12 | 0.0025 | 0.5833 | -0.7217 | -0.0843 | 0.8327 | 0.1361 | 0.0833 | -0.0333 | 0.1333 | 0.3333 |
| dirfix | 1 | 12 | -0.7037 | 0.0833 | -1.819 | -0.7041 | 0.0412 | 0.1917 | 0.25 | -0.1667 | 0.1 | 0.7 |
| dirfix | 1.5 | 12 | -1.029 | 0.0833 | -2.898 | -0.8581 | 1.151 | 0.3306 | 0.0833 | -0.0667 | 0.25 | 0.9333 |
| dirfix | 2 | 12 | -1.261 | 0.1667 | -3.855 | -0.6025 | 1.161 | 0.475 | 0 | 0 | 0.5333 | 1 |
| residual | 1 | 12 | -0.0323 | 0.5 | -0.5452 | 0.0088 | 0.4269 | 0.0472 | 0.25 | -0.1333 | 0.0167 | 0.3 |
| residual | 1.5 | 12 | -0.0079 | 0.5 | -0.4967 | 0.0136 | 0.7658 | 0.0639 | 0.25 | -0.1 | 0.05 | 0.5 |
| residual | 2 | 12 | 0.0649 | 0.4167 | -0.312 | 0.0507 | 0.7931 | 0.1056 | 0.0833 | -0.0333 | 0.0333 | 0.4333 |
| shared | 1 | 12 | -1.336 | 0 | -2.244 | -1.257 | -0.4051 | 0.2472 | 0 | 0 | 0.1 | 0.7333 |
| shared | 1.5 | 12 | -2.032 | 0.0833 | -3.772 | -1.941 | 0.376 | 0.4278 | 0 | 0 | 0.3167 | 0.9333 |
| shared | 2 | 12 | -2.364 | 0.0833 | -4.184 | -2.534 | 0.3145 | 0.5861 | 0 | 0 | 0.6833 | 1 |
| shared_only | 1 | 12 | -2.644 | 0 | -3.316 | -2.749 | -1.276 | -0.4528 | 0.0833 | -0.9333 | -0.3833 | 0.0333 |
| shared_only | 1.5 | 12 | -4.005 | 0 | -5.268 | -4.547 | -0.4603 | -0.3472 | 0 | -0.9 | -0.25 | 0 |
| shared_only | 2 | 12 | -4.457 | 0 | -6.639 | -5.358 | -0.3003 | -0.175 | 0 | -0.6667 | -0.1167 | 0 |

