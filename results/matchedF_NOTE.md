# Matched-repetition comparison (scored_expF_r3.csv, baseline = naive)

Each arm is compared to the baseline at the strength where it reproduces the baseline's value of the matching metric on that (feature, c_n) cell (linear interpolation on the arm's own c-grid), not at the same nominal c.

## Matched on `rep4` (comparing d_logppl, d_keyword_hit)

55 / 504 (feature, c_n, arm) cells skipped overall (baseline's rep4 value fell outside the arm's achievable range).

| arm | n_pairs | n_matched | n_skipped | skip_frac | n_features | mean_d_logppl | d_logppl_lo95 | d_logppl_hi95 | mean_d_keyword_hit | d_keyword_hit_lo95 | d_keyword_hit_hi95 | mean_d_rep4 | d_rep4_lo95 | d_rep4_hi95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| centred | 72 | 71 | 1 | 0.0139 | 12 | -0.8905 | -1.365 | -0.3445 | 0.016 | -0.0624 | 0.0934 | nan | nan | nan |
| diffmeans | 72 | 69 | 3 | 0.0417 | 12 | -0.7606 | -1.235 | -0.2116 | 0.0181 | -0.0711 | 0.1316 | nan | nan | nan |
| diffmeans_purified | 72 | 67 | 5 | 0.0694 | 12 | -0.8286 | -1.295 | -0.2693 | -0.002 | -0.1029 | 0.1196 | nan | nan | nan |
| dirfix | 72 | 63 | 9 | 0.125 | 12 | -0.9532 | -1.402 | -0.4117 | 0.0331 | -0.0596 | 0.1213 | nan | nan | nan |
| purified | 72 | 68 | 4 | 0.0556 | 12 | -0.7923 | -1.278 | -0.2295 | 0.0261 | -0.0509 | 0.1028 | nan | nan | nan |
| rotate | 72 | 65 | 7 | 0.0972 | 12 | -1.124 | -1.538 | -0.5797 | -0.0104 | -0.09 | 0.0601 | nan | nan | nan |
| shared | 72 | 46 | 26 | 0.3611 | 12 | -1.488 | -1.768 | -1.057 | 0.0394 | -0.1158 | 0.1984 | nan | nan | nan |

## Matched on `keyword_hit` (comparing d_logppl, d_rep4)

34 / 504 (feature, c_n, arm) cells skipped overall (baseline's keyword_hit value fell outside the arm's achievable range).

| arm | n_pairs | n_matched | n_skipped | skip_frac | n_features | mean_d_logppl | d_logppl_lo95 | d_logppl_hi95 | mean_d_keyword_hit | d_keyword_hit_lo95 | d_keyword_hit_hi95 | mean_d_rep4 | d_rep4_lo95 | d_rep4_hi95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| centred | 72 | 69 | 3 | 0.0417 | 12 | -0.8391 | -1.406 | -0.2236 | nan | nan | nan | -0.0291 | -0.0845 | 0.0007 |
| diffmeans | 72 | 64 | 8 | 0.1111 | 12 | -0.6217 | -1.193 | -0.0441 | nan | nan | nan | -0.0439 | -0.1106 | 0.0017 |
| diffmeans_purified | 72 | 64 | 8 | 0.1111 | 12 | -0.6047 | -1.17 | -0.0312 | nan | nan | nan | -0.0435 | -0.1079 | 0.002 |
| dirfix | 72 | 70 | 2 | 0.0278 | 12 | -1.142 | -1.635 | -0.5422 | nan | nan | nan | -0.0269 | -0.0816 | 0.0111 |
| purified | 72 | 68 | 4 | 0.0556 | 12 | -0.8 | -1.352 | -0.2097 | nan | nan | nan | -0.0323 | -0.0888 | 0.0005 |
| rotate | 72 | 64 | 8 | 0.1111 | 12 | -0.8569 | -1.429 | -0.2342 | nan | nan | nan | -0.0298 | -0.089 | 0.0005 |
| shared | 72 | 71 | 1 | 0.0139 | 12 | -1.611 | -1.994 | -1.18 | nan | nan | nan | 0.0446 | -0.0415 | 0.1478 |

