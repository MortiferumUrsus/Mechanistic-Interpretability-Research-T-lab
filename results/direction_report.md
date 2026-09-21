# Direction report

Mean over 30 features x 3 splits (210 rows), per direction family. kappa_shared=0.75, ckpt=dir_hot.

```
                    cos_d_bar  cos_mu_hat  cos_v_hat  weak_half_energy  logit_reach
family                                                                             
decoder                0.0221      0.0459     1.0000            0.4766      37.9902
shared                 0.6110     -0.0492     0.8047            0.6008      33.9263
diffmeans             -0.0047      0.1049     0.8544            0.4963      40.2788
centred                0.0289     -0.0000     0.9962            0.4726      38.0560
purified               0.0000      0.0491     0.9991            0.4753      38.0169
diffmeans_purified    -0.0000      0.1043     0.8539            0.4948      40.2717
learned                0.3641     -0.0090     0.8469            0.5736      35.4447
```
