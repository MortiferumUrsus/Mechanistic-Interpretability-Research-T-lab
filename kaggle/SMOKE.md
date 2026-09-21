# Running the pipeline on Kaggle

All commands below are run from the repository root, with an interpreter that has the Kaggle CLI
installed (`python -m pip install kaggle`; the repository `.venv` does not have it). Everything is
invoked as `python -m kaggle ...` rather than a bare `kaggle` entry point, so it works the same on
any platform.

Two environment variables control the account:

- `KAGGLE_USERNAME` -- the Kaggle account the dataset and the kernels belong to. Defaults to
  `mortiferumursus`, the account the reported runs used.
- `KAGGLE_API_TOKEN` -- the API token. `build_payload.py` and `push.py` read it from the
  environment; alternatively pass `--token-file path/to/token.txt` (a one-line file). The token is
  only ever set on the Kaggle CLI subprocess environment: it is never printed and never written
  anywhere. With neither set, both scripts exit with a one-line message.

## 0. Check authentication (once)

```
python -m kaggle kernels list --mine
```

## 1. Build and push the payload

The payload is everything the Kaggle kernel needs to reproduce the pipeline: `src/`, `scripts/`,
`configs/`, the small `data/` artifacts (`act_stats.pt`, `splits*.npz`, `prompts.json`,
`sae_feature_stats.pt`, `shared_direction.pt`), the few `results/` files later stages read as input
(`feature_stats.csv`, `gen_expE_r4.jsonl`), the starting checkpoints (`dir_hot.pt`,
`mlp_gauss_cond1_s0.pt`), and `kaggle/smoke_check.py`, which rides along as `scripts/smoke_check.py`.
The heavy activation dump (`data/acts.f16`, ~2.3 GB, and `data/toks.i32`) is deliberately left out:
the kernel rebuilds it with `activations.py dump` + `sae_stats.py`.

```
python kaggle/build_payload.py --dry-run
```

Check the output: the total must be far below 100 MB (currently ~13 MB). If it looks right, upload
it as the private dataset `$KAGGLE_USERNAME/tlab-mi-payload` (`datasets create` the first time,
`datasets version` every time after):

```
python kaggle/build_payload.py --push
```

## 2. Smoke kernel (10-15 minutes, environment check)

```
python kaggle/make_kernel.py --name tlab-mi-smoke --queue kaggle/smoke_queue.txt --out kaggle/_build/smoke
python kaggle/push.py --dir kaggle/_build/smoke --dry-run
python kaggle/push.py --dir kaggle/_build/smoke --push
```

Poll the status with `python kaggle/push.py --status tlab-mi-smoke`. Once it is `COMPLETE`:

```
python kaggle/push.py --fetch tlab-mi-smoke --out kaggle/_build/smoke/out
```

Look at `kaggle/_build/smoke/out/out/logs/smoke.log` -- it must say `7/7 checks passed`. Do not
start the long kernels before the smoke kernel passes in full: it is the check that the Kaggle
environment comes together at all (transformer_lens/sae_lens, gpt2-small, the SAE,
`activations.py verify`, internet access, generation).

## 3. Two long kernels in parallel (both GPU sessions at once)

The Kaggle session limit is 12 hours per kernel. Kernel 1 does not need the heavy activation dump
and is generated so that the data-restore step never runs at all (`--skip-data-restore`); kernel 2
does need it, so it takes longer.

Kernel 1 (`expA` + `expD`, ~5h, no data restore):

```
python kaggle/make_kernel.py --name tlab-mi-run1 --queue kaggle/kernel1_queue.txt --skip-data-restore --out kaggle/_build/run1
python kaggle/push.py --dir kaggle/_build/run1 --push
```

Kernel 2 (`expG` + `expE`, ~7h, restoring `data/acts.f16` via `activations.py dump` +
`sae_stats.py`):

```
python kaggle/make_kernel.py --name tlab-mi-run2 --queue kaggle/kernel2_queue.txt --out kaggle/_build/run2
python kaggle/push.py --dir kaggle/_build/run2 --push
```

The drivers `scripts/run_expA.py`, `run_expD.py`, `run_expE.py`, `run_expG.py` may not be in the
repository yet at run time -- if one is missing, its queue item lands in `QUEUE_SUMMARY.json` with
the status `missing driver` and the rest of the queue keeps running.

## 4. Fetch the results

```
python kaggle/push.py --status tlab-mi-run1
python kaggle/push.py --status tlab-mi-run2
python kaggle/push.py --fetch tlab-mi-run1 --out kaggle/_build/run1/out
python kaggle/push.py --fetch tlab-mi-run2 --out kaggle/_build/run2/out
```

Each `<run>/out/out/QUEUE_SUMMARY.json` holds the status and the time of every queue item plus the
total against the 12-hour limit (if it is close to or above the limit, part of the queue did not run
that night, visible in the `skip` / `missing driver` entries and in what is missing from
`results/`). The artifacts themselves are in `<run>/out/out/results`, `checkpoints`, `configs`, plus
the small `data/` files (act_stats.pt, splits*.npz, prompts.json, sae_feature_stats.pt and so on).
The heavy `acts.f16` will not be in the output: inside the kernel `data/` was a symlink to
`/kaggle/temp/data`, which Kaggle does not persist, and the notebook's final cell also deletes and
warns about any file larger than 200 MB that still ended up in `/kaggle/working`.

## Two platform pitfalls

1. **Nested dataset mount path.** The same dataset shows up either as `/kaggle/input/<slug>` or
   nested as `/kaggle/input/datasets/<owner>/<slug>`, depending on the Kaggle image. The notebook's
   setup cell therefore locates the payload by structure (the directory holding both `src/` and
   `configs/`) instead of by name, and prints what is actually mounted; do not hard-code the mount
   path. If nothing is found, the dataset was most likely still processing when the kernel was
   pushed -- attach `<username>/tlab-mi-payload` and re-run.
2. **Kaggle CLI encoding error on Windows.** On a Windows console with a legacy code page the CLI
   can die with a `UnicodeEncodeError: 'charmap' codec can't encode character ...` while printing
   its output (dataset or kernel listings, progress lines). Set `PYTHONUTF8=1` (or
   `PYTHONIOENCODING=utf-8`, or switch the console to UTF-8 with `chcp 65001`) before running the
   command. It is a
   printing failure only -- the upload or fetch itself is unaffected.
