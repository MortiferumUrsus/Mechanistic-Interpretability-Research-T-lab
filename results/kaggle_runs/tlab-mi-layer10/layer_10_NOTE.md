# Layer 10: does the same shared direction correction exist, and does it match layer 7?

## manifest
```
layer=10
HOOK=blocks.9.hook_resid_post
SAE_ID=blocks.10.hook_resid_pre
root=/kaggle/working/repo
```

## anatomy of the learned correction on this layer
- pairwise cosine between per-feature corrections: mean=0.15842433273792267 median=0.19708950817584991
- mean correction norm / mean direction norm: 0.3985538184642792
- correction-cloud variance share: top1=0.2926018238067627 top5=0.6903199627995491 top20=0.8594615701586008
- M singular values (top 10): [8.43231201171875, 5.479042053222656, 4.518512725830078, 3.4085474014282227, 3.128053903579712, 2.069329023361206, 1.6393840312957764, 1.471813678741455, 1.2255079746246338, 1.1986459493637085]
- M Frobenius norm: 13.042678833007812, participation ratio: 19.52527369778254

## d_bar cosine vs the layer-7 shared direction
cos(d_bar[L10], d_bar[L7]) = 0.756236

## arm x strength: fluency and concept metrics (test split, this layer)
| arm | c | logppl | keyword_hit | rep4 |
| --- | --- | --- | --- | --- |
| dirfix | 0 | 3.56 | 0.1194 | 0.005 |
| dirfix | 0.5 | 3.494 | 0.2222 | 0.0025 |
| dirfix | 1 | 4.008 | 0.4833 | 0.0101 |
| dirfix | 1.5 | 4.3 | 0.5778 | 0.0883 |
| dirfix | 2 | 4.141 | 0.5222 | 0.1982 |
| dirfix | 3 | 2.76 | 0.2278 | 0.4188 |
| naive | 0 | 3.56 | 0.1194 | 0.005 |
| naive | 0.5 | 3.742 | 0.2167 | 0.001 |
| naive | 1 | 4.387 | 0.3417 | 0.0013 |
| naive | 1.5 | 5.112 | 0.3472 | 0.0089 |
| naive | 2 | 5.104 | 0.2972 | 0.0767 |
| naive | 3 | 3.756 | 0.2083 | 0.2338 |
| shared | 0 | 3.56 | 0.1194 | 0.005 |
| shared | 0.5 | 3.267 | 0.1778 | 0.005 |
| shared | 1 | 3.3 | 0.5333 | 0.0383 |
| shared | 1.5 | 3.219 | 0.6472 | 0.1971 |
| shared | 2 | 2.602 | 0.5306 | 0.4412 |
| shared | 3 | 1.584 | 0.2806 | 0.6672 |

## paired differences vs naive, at matched strength
| arm | n_cells | n_features | d_logppl | logppl_lo95 | logppl_hi95 | d_concept | concept_lo95 | concept_hi95 | p_dominates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared | 48 | 12 | -1.914 | -2.311 | -1.544 | 0.1993 | 0.0444 | 0.3451 | 0.995 |
| dirfix | 48 | 12 | -0.7876 | -1.095 | -0.4897 | 0.1542 | 0.0646 | 0.2542 | 1 |

