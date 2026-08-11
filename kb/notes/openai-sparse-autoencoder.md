# openai/sparse_autoencoder

## (a) Full citation + URL

Code repo: https://github.com/openai/sparse_autoencoder — companion release to:

Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, Jeffrey Wu. **"Scaling and Evaluating Sparse Autoencoders."** arXiv:2406.04093 [cs.LG], June 2024.

- Paper abstract: https://arxiv.org/abs/2406.04093
- PDF: https://arxiv.org/pdf/2406.04093 (downloaded → `kb/pdf/openai-sae-scaling-2406.04093.pdf`)
- Repo README: https://github.com/openai/sparse_autoencoder/blob/main/README.md

## (b) Problem addressed
Same general SAE-scaling problem as SAELens's SAEs, but from OpenAI: reconstruction-vs-sparsity tradeoff is hard to tune with L1 penalties and suffers from dead latents at scale. Paper's fix: **k-sparse (TopK) autoencoders**, which directly control sparsity ($k$ active latents per token, exactly) instead of tuning an L1 coefficient, plus architectural tricks that keep dead-latent counts low even at very large scale. This produced clean scaling laws for SAE size vs. sparsity, and introduced new feature-quality metrics (hypothesized-feature recovery, activation-pattern explainability).

## (c) What SAEs are released, for which model/hook points, how to load

Repo hosts **GPT-2-small** SAEs (only GPT-2-small; no other model). From `sparse_autoencoder/paths.py` (fetched and read directly):

- **`v1(location, layer_index)`** and **`v4(location, layer_index)`** (v4 = same recipe, presumably retrained/refreshed): `location ∈ {"mlp_post_act", "resid_delta_mlp"}`, `layer_index ∈ range(12)`. 32,768 latents, ~64M training tokens, **ReLU** activation, L1 coefficient 0.01, inputs not layer-normed.
- **`v5_32k(location, layer_index)`**: `location ∈ {"resid_delta_attn", "resid_delta_mlp", "resid_post_attn", "resid_post_mlp"}`, `layer_index ∈ range(12)`. $2^{15}=32768$ latents, **TopK(k=32)** activation (not ReLU/L1 — this is the paper's headline architecture), inputs layer-normed.
- **`v5_128k(location, layer_index)`**: same locations, $2^{17}=131072$ latents, TopK(32), layer-normed inputs.
- Comment in source notes larger autoencoders (up to 8M latents, varying $n$/$k$) trained on **layer 8, `resid_post_mlp`** exist internally but were not (at time of writing) released.

**Hook-point vocabulary is OpenAI's own naming, not TransformerLens's** — `resid_delta_mlp` / `resid_delta_attn` mean the *delta* (output contribution) written to the residual stream by the MLP/attention sublayer at that layer (i.e. analogous to TransformerLens's `hook_mlp_out`/`hook_attn_out`, **not** the residual stream itself), whereas `resid_post_mlp` / `resid_post_attn` are the actual residual-stream state after that sublayer (closer to TransformerLens's `hook_resid_post`, though attention and MLP are split into two sub-points here rather than TransformerLens's per-block `hook_resid_mid`/`hook_resid_post`). Map carefully if you need exact hook-for-hook equivalence with a TransformerLens-based pipeline: `resid_post_mlp` at layer $L$ should be the closest analog to TransformerLens's `blocks.{L}.hook_resid_post`.

**Loading**: `pip install git+https://github.com/openai/sparse_autoencoder.git`, then paths like `sparse_autoencoder.paths.v5_32k("resid_post_mlp", 6)` return an Azure-blob URI (`az://openaipublic/sparse-autoencoder/gpt2-small/...`) to download via the package's own blob-fetch utility; load via `torch.load(...)` into the repo's `Autoencoder.from_state_dict(...)` (per README, exact call signature `[UNVERIFIED]` — not independently reproduced in this pass, re-check README/`model.py` directly before depending on it verbatim).

## (d) Evaluation practice (paper-level)
The **Scaling and Evaluating Sparse Autoencoders** paper's headline contributions are new SAE-quality metrics beyond simple reconstruction MSE — "recovery of hypothesized features" and "explainability of activation patterns" — plus clean power-law scaling curves for reconstruction-vs-sparsity as a function of autoencoder size. These are SAE-training-quality metrics, not steering-fluency metrics, so they're of secondary relevance to the task's Pareto-front eval, but the **TopK-vs-ReLU/L1 distinction** is directly relevant if you're deciding what kind of SAE (or its decoder columns) to source steering vectors from — TopK SAEs give an exact, controllable L0 (useful if you want a specific, known number of "active" concepts per token) whereas SAELens's `gpt2-small-res-jb` release (see `kb/notes/saelens.md`) is the older ReLU+L1 style with an *empirical, not exact* L0.

## Directly reusable techniques / pitfalls
1. **If you want an exact-L0 SAE for cleaner steering-vector provenance, prefer `v5_32k`/`v5_128k` (TopK) over `v1`/`v4` (ReLU+L1)** — TopK gives a precisely known sparsity per token rather than an empirical average.
2. **`resid_post_mlp` at layer $L$ ≈ TransformerLens `blocks.{L}.hook_resid_post`** — use this mapping if you want to cross-check an OpenAI-SAE-derived direction against a TransformerLens-hooked GPT-2-small forward pass.
3. **This repo only covers GPT-2-small** — convenient, since that's exactly the target task's model, but note the released SAEs are trained on the *base* (non-instruct) GPT-2-small — consistent with what the task needs (GPT-2 small, 124M, base LM).
4. **Pitfall**: the download mechanism is an Azure blob path (`az://...`), not a plain HTTPS URL or HuggingFace repo like SAELens's — you need the package's own fetch helper (or reimplement the equivalent public HTTPS mirror if one exists) rather than a naive `curl`/`requests.get`. `[UNVERIFIED]`: whether a plain HTTPS mirror of these blobs exists; not checked in this pass.
5. **Pitfall**: layer-normed vs non-layer-normed inputs differ between `v1`/`v4` (not normed) and `v5_*` (normed) — if you're comparing decoder-direction norms across these releases (e.g. to set a steering alpha), a layer-normed-input SAE's decoder columns live in a differently-scaled space than a raw-input one; don't mix conventions without renormalizing.
