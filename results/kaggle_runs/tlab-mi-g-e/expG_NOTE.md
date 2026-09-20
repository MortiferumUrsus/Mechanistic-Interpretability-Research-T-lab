# Experiment G -- arm x strength

Facts pulled directly from `scored_expG_wiener.csv` and `scored_expG_learned.csv`; `naive` is present in both files and is averaged over both here.

| arm | c (strength) | log-PPL | keyword_hit | sae_act | prompt_dependence | n |
|---|---|---|---|---|---|---|
| cond_denoise | 0 | 3.5594 | 0.0556 | 0.0038 | 0.5803 | 360 |
| cond_denoise | 0.5 | 6.1134 | 0.0944 | 0.0058 | 0.0646 | 360 |
| cond_denoise | 1 | 7.3712 | 0.1222 | 0.0137 | 0.0704 | 360 |
| cond_denoise | 1.5 | 7.8646 | 0.1361 | 0.0335 | -0.0020 | 360 |
| cond_denoise | 2 | 7.9212 | 0.1194 | 0.0563 | -0.0019 | 360 |
| cond_denoise | 3 | 7.6795 | 0.0806 | 0.0270 | -0.0735 | 360 |
| cond_wiener | 0 | 3.5596 | 0.0556 | 0.0038 | 0.5799 | 360 |
| cond_wiener | 0.5 | 6.0864 | 0.4583 | 2.6610 | 0.0021 | 360 |
| cond_wiener | 1 | 6.0521 | 0.4528 | 2.5664 | -0.0040 | 360 |
| cond_wiener | 1.5 | 6.0496 | 0.4528 | 2.5439 | -0.0514 | 360 |
| cond_wiener | 2 | 6.0482 | 0.4528 | 2.5383 | -0.0322 | 360 |
| cond_wiener | 3 | 6.0486 | 0.4556 | 2.5400 | -0.0501 | 360 |
| naive | 0 | 3.5597 | 0.0556 | 0.0038 | 0.5471 | 720 |
| naive | 0.5 | 4.0456 | 0.3222 | 0.3398 | 0.4196 | 720 |
| naive | 1 | 5.0135 | 0.4861 | 0.7129 | 0.1703 | 720 |
| naive | 1.5 | 5.6338 | 0.3528 | 0.4852 | 0.0909 | 720 |
| naive | 2 | 5.6915 | 0.1778 | 0.1757 | 0.0903 | 720 |
| naive | 3 | 5.1384 | 0.0583 | 0.0682 | 0.0192 | 720 |
