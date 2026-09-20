# Matched-repetition comparison (scored_expA_r3.csv, baseline = naive)

Each arm is compared to the baseline at the strength where it reproduces the baseline's value of the matching metric on that (feature, c_n) cell (linear interpolation on the arm's own c-grid), not at the same nominal c.

## Matched on `rep4` (comparing d_logppl, d_keyword_hit)

81 / 480 (feature, c_n, arm) cells skipped overall (baseline's rep4 value fell outside the arm's achievable range).

| arm | n_pairs | n_matched | n_skipped | skip_frac | n_features | mean_d_logppl | d_logppl_lo95 | d_logppl_hi95 | mean_d_keyword_hit | d_keyword_hit_lo95 | d_keyword_hit_hi95 | mean_d_rep4 | d_rep4_lo95 | d_rep4_hi95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antimanifold | 96 | 91 | 5 | 0.0521 | 12 | -0.9576 | -1.511 | -0.3401 | -0.0484 | -0.1244 | 0.0286 | nan | nan | nan |
| dirfix | 96 | 82 | 14 | 0.1458 | 12 | -1.316 | -1.758 | -0.8149 | 0.1121 | -0.017 | 0.2388 | nan | nan | nan |
| residual | 96 | 90 | 6 | 0.0625 | 12 | -0.7189 | -1.228 | -0.1419 | 0.087 | -0.0238 | 0.2035 | nan | nan | nan |
| shared | 96 | 71 | 25 | 0.2604 | 12 | -1.632 | -2.02 | -1.156 | 0.1239 | -0.0282 | 0.2606 | nan | nan | nan |
| shared_only | 96 | 65 | 31 | 0.3229 | 12 | 1.359 | -0.3282 | 2.896 | -0.1537 | -0.2456 | -0.0805 | nan | nan | nan |

## Matched on `keyword_hit` (comparing d_logppl, d_rep4)

64 / 480 (feature, c_n, arm) cells skipped overall (baseline's keyword_hit value fell outside the arm's achievable range).

| arm | n_pairs | n_matched | n_skipped | skip_frac | n_features | mean_d_logppl | d_logppl_lo95 | d_logppl_hi95 | mean_d_keyword_hit | d_keyword_hit_lo95 | d_keyword_hit_hi95 | mean_d_rep4 | d_rep4_lo95 | d_rep4_hi95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| antimanifold | 96 | 91 | 5 | 0.0521 | 12 | -0.7991 | -1.333 | -0.1976 | nan | nan | nan | -0.0452 | -0.1301 | 0.0002 |
| dirfix | 96 | 91 | 5 | 0.0521 | 12 | -0.9432 | -1.507 | -0.2568 | nan | nan | nan | -0.0466 | -0.1368 | 0.0095 |
| residual | 96 | 89 | 7 | 0.0729 | 12 | -0.5405 | -1.19 | 0.1941 | nan | nan | nan | -0.0476 | -0.1413 | 0.0009 |
| shared | 96 | 90 | 6 | 0.0625 | 12 | -1.43 | -1.95 | -0.8021 | nan | nan | nan | 0.056 | -0.0692 | 0.1903 |
| shared_only | 96 | 55 | 41 | 0.4271 | 12 | -1.143 | -1.956 | -0.2186 | nan | nan | nan | 0.0389 | -0.1742 | 0.2285 |

