# Exp A: anatomy of the direction correction, and the shared/residual/antimanifold arms

## Selected kappa (configs/expA.yaml)

- kappa_shared: 0.75
- kappa_anti: 0.75

## Anatomy of the correction (results/anatomy_summary.json)

- ckpt: dir_hot
- rank: 64
- n_cos_sample: 2000
- mean_pairwise_cos: 0.2939390242099762
- median_pairwise_cos: 0.3666073679924011
- mean_delta_norm_over_mean_norm: 0.5424869656562805
- delta_cloud_variance_share: {'top1': 0.24406012892723083, 'top5': 0.5134643651545048, 'top20': 0.7805689563974738}
- M_singular_values_top10: [8.964223861694336, 4.483459949493408, 2.8617870807647705, 2.539008378982544, 2.336015462875366, 2.0670411586761475, 2.041717529296875, 1.8614617586135864, 1.7115169763565063, 1.6682159900665283]
- M_frobenius_norm: 12.918960571289062
- M_participation_ratio: 25.160909778963305
- quad_inv_v_mean: 0.25216248631477356
- quad_v_mean: 16.835205078125
- quad_inv_w_mean: 0.7608999013900757
- quad_shrinkage_curve: [{'shrink': 0.001, 'quad_inv_v': 0.6416013836860657, 'quad_inv_w': 5.3588433265686035, 'quad_inv_ratio': 8.352293968200684, 'quad_v': 16.935260772705078, 'quad_w': 19.70061492919922, 'quad_fwd_ratio': 1.1632897853851318}, {'shrink': 0.01, 'quad_inv_v': 0.25216248631477356, 'quad_inv_w': 0.7608999013900757, 'quad_inv_ratio': 3.01749849319458, 'quad_v': 16.835205078125, 'quad_w': 19.575647354125977, 'quad_fwd_ratio': 1.1627804040908813}, {'shrink': 0.05, 'quad_inv_v': 0.20762841403484344, 'quad_inv_w': 0.32673147320747375, 'quad_inv_ratio': 1.5736356973648071, 'quad_v': 16.390512466430664, 'quad_w': 19.02022933959961, 'quad_fwd_ratio': 1.1604413986206055}, {'shrink': 0.2, 'quad_inv_v': 0.17809128761291504, 'quad_inv_w': 0.2099859118461609, 'quad_inv_ratio': 1.179091453552246, 'quad_v': 14.72291374206543, 'quad_w': 16.937410354614258, 'quad_fwd_ratio': 1.150411605834961}]
- quad_w_mean: 19.575647354125977
- pc5_mass_v_mean: 0.06072307005524635
- pc5_mass_w_mean: 0.08637218177318573
- shrink: 0.01
- n_pc: 5

## Arm x strength: log-PPL and keyword_hit, test_r3 (results/scored_expA_r3.csv)

| arm | c | logppl | keyword_hit |
| --- | --- | --- | --- |
| antimanifold | 0 | 3.56 | 0.0556 |
| antimanifold | 0.5 | 3.879 | 0.1556 |
| antimanifold | 1 | 4.659 | 0.5028 |
| antimanifold | 1.5 | 5.328 | 0.4444 |
| antimanifold | 2 | 5.694 | 0.3139 |
| antimanifold | 3 | 5.497 | 0.1 |
| antimanifold | 4 | 4.905 | 0.0583 |
| antimanifold | 5 | 4.476 | 0.05 |
| dirfix | 0 | 3.559 | 0.0556 |
| dirfix | 0.5 | 3.722 | 0.3306 |
| dirfix | 1 | 4.31 | 0.6778 |
| dirfix | 1.5 | 4.605 | 0.6833 |
| dirfix | 2 | 4.431 | 0.6528 |
| dirfix | 3 | 3.551 | 0.5083 |
| dirfix | 4 | 3.118 | 0.4306 |
| dirfix | 5 | 2.958 | 0.4167 |
| naive | 0 | 3.56 | 0.0556 |
| naive | 0.5 | 4.046 | 0.3222 |
| naive | 1 | 5.014 | 0.4861 |
| naive | 1.5 | 5.634 | 0.3528 |
| naive | 2 | 5.692 | 0.1778 |
| naive | 3 | 5.138 | 0.0583 |
| naive | 4 | 4.471 | 0.0417 |
| naive | 5 | 4.016 | 0.0417 |
| residual | 0 | 3.56 | 0.0556 |
| residual | 0.5 | 4.009 | 0.2806 |
| residual | 1 | 4.981 | 0.5333 |
| residual | 1.5 | 5.626 | 0.4167 |
| residual | 2 | 5.757 | 0.2833 |
| residual | 3 | 5.346 | 0.0972 |
| residual | 4 | 5.006 | 0.0917 |
| residual | 5 | 4.697 | 0.1 |
| shared | 0 | 3.56 | 0.0556 |
| shared | 0.5 | 3.503 | 0.3083 |
| shared | 1 | 3.677 | 0.7333 |
| shared | 1.5 | 3.602 | 0.7806 |
| shared | 2 | 3.327 | 0.7639 |
| shared | 3 | 2.667 | 0.6528 |
| shared | 4 | 2.39 | 0.5528 |
| shared | 5 | 2.421 | 0.5472 |
| shared_only | 0 | 3.56 | 0.0556 |
| shared_only | 0.5 | 3.025 | 0.0583 |
| shared_only | 1 | 2.369 | 0.0333 |
| shared_only | 1.5 | 1.629 | 0.0056 |
| shared_only | 2 | 1.235 | 0.0028 |
| shared_only | 3 | 1.924 | 0.0056 |
| shared_only | 4 | 3.795 | 0.0056 |
| shared_only | 5 | 7.102 | 0.0083 |

