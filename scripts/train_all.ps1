# Train every denoiser variant used in the ablations. ~6 minutes each on a GTX 1660 Ti.
$ErrorActionPreference = "Continue"
$py = "$PSScriptRoot\..\.venv\Scripts\python.exe"
Push-Location "$PSScriptRoot\..\src"
$steps = 60000

& $py train_denoiser.py --arch mlp    --noise mix   --cond 1 --seed 0 --steps $steps --log-every 20000
& $py train_denoiser.py --arch mlp    --noise gauss --cond 1 --seed 0 --steps $steps --log-every 20000
& $py train_denoiser.py --arch linear --noise mix   --cond 0 --seed 0 --steps $steps --log-every 20000
& $py train_denoiser.py --arch mlp    --noise mix   --cond 0 --seed 0 --steps $steps --log-every 20000
& $py train_denoiser.py --arch mlp    --noise mix   --cond 1 --seed 1 --steps $steps --log-every 20000
& $py train_denoiser.py --arch mlp    --noise mix   --cond 1 --seed 0 --steps $steps --log-every 20000 --fixed-rel 2.0 --name mlp_fixed200

Pop-Location
