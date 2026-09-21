# Layer profile: residual norm, projection on d_bar, and final entropy by layer

Features: first 4 of split `test_r3`. Prompts: 30 from `data/prompts.json`. Strengths: [1.0, 2.0].

## mean last-position resid_post norm, by layer (6..11)

| arm | c | layer_6 | layer_7 | layer_8 | layer_9 | layer_10 | layer_11 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0 | 94.08 | 107.1 | 124.7 | 153.3 | 233.6 | 414.3 |
| naive | 1 | 107.6 | 118.8 | 133.5 | 158.3 | 230.8 | 401.7 |
| naive | 2 | 139 | 150.9 | 164.3 | 186.9 | 249.7 | 421.9 |
| shared | 1 | 103.8 | 113.1 | 129.9 | 155 | 221.8 | 339.6 |
| shared | 2 | 133 | 137.1 | 149.6 | 171.8 | 227.4 | 269.8 |
| shared_only | 1 | 101.5 | 109.1 | 125.6 | 151.1 | 216.5 | 313.3 |
| shared_only | 2 | 129.4 | 130.2 | 139.7 | 158.4 | 205.2 | 248.5 |

## mean last-position projection onto d_bar, by layer (6..11)

| arm | c | layer_6 | layer_7 | layer_8 | layer_9 | layer_10 | layer_11 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0 | -10.67 | -14.51 | -18.84 | -25 | -42.07 | -35.9 |
| naive | 1 | -10.02 | -14.29 | -18.73 | -24.17 | -40.84 | -36.27 |
| naive | 2 | -9.364 | -14.18 | -19.21 | -24.48 | -40.97 | -40.25 |
| shared | 1 | 24.83 | 19.25 | 14.27 | 8.988 | -7.207 | -4.419 |
| shared | 2 | 60.33 | 55.16 | 50.27 | 45.39 | 29.38 | 30.14 |
| shared_only | 1 | 39.21 | 34.05 | 29.86 | 24.16 | 8.249 | 11.28 |
| shared_only | 2 | 89.09 | 86.16 | 86.68 | 83.39 | 69.68 | 71.16 |

## entropy of the final next-token distribution, by arm x c

| arm | c | entropy |
| --- | --- | --- |
| clean | 0 | 3.758 |
| naive | 1 | 3.876 |
| naive | 2 | 3.564 |
| shared | 1 | 3.32 |
| shared | 2 | 1.925 |
| shared_only | 1 | 2.873 |
| shared_only | 2 | 2.046 |

