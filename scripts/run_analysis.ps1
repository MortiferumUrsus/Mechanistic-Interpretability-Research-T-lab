# E5: mechanism measurements. Runs after the TEST pass; reads frozen.yaml for the arm settings.
$ErrorActionPreference = "Continue"
$py = "$PSScriptRoot\..\.venv\Scripts\python.exe"
Push-Location "$PSScriptRoot\..\src"

# how much of the steering each denoiser transmits, learned and closed-form side by side
& $py analysis.py transmission --split test --denoisers mlp_mix_cond1_s0,mlp_gauss_cond1_s0,mlp_fixed200,wiener:0.5,wiener:1.0,wiener:2.0

# where the perturbation energy sits in the covariance eigenbasis, before and after each repair
& $py analysis.py spectral --split test

# which non-target latents get clamped, and how many
& $py analysis.py surgery --split test

# tangent-aligned versus off-tangent downstream response
& $py analysis.py causal --split test

# how far generation-time activations drift from the training corpus
& $py analysis.py rollout_shift --split test --max-features 4

# per-feature gain against the two registered geometric predictors
& $py analysis.py predictors

& $py plots.py --scored scored_test.csv --concept keyword_hit
& $py make_html.py

Pop-Location
