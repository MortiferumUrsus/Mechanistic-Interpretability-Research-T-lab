# Exp E: 48 fresh features on disjoint prompts (round four)

- features in `configs/features.yaml:test_r4`: 48
- fresh prompts in `data/prompts_r4.json`: not generated yet (prompts stage did not run)

## primary endpoint (concept_at_budget, budget = naive at c=1.0), test_r4
| arm | endpoint_mean | endpoint_median | n_features | endpoint_on_common | n_common | boot_mean | boot_lo95 | boot_hi95 | n_valid | delta_mean | delta_lo95 | delta_hi95 | p_gt_0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dirfix | 0.7903 | 0.925 | 48 | 0.7903 | 48 | 0.7944 | 0.7159 | 0.8696 | 2000 | 0.2787 | 0.2129 | 0.347 | 1 |
| naive | 0.511 | 0.4833 | 48 | 0.511 | 48 | 0.5157 | 0.4453 | 0.592 | 2000 | 0 | 0 | 0 | 0 |
| shared | 0.8785 | 1 | 48 | 0.8785 | 48 | 0.8821 | 0.8095 | 0.9442 | 2000 | 0.3664 | 0.2939 | 0.432 | 1 |

## paired differences at matched strength vs naive, test_r4
| arm | n_cells | n_features | d_logppl | logppl_lo95 | logppl_hi95 | d_concept | concept_lo95 | concept_hi95 | p_dominates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared | 192 | 48 | -2.474 | -2.839 | -2.09 | 0.4112 | 0.3194 | 0.4993 | 1 |
| dirfix | 192 | 48 | -1.235 | -1.599 | -0.896 | 0.3149 | 0.2335 | 0.3999 | 1 |
