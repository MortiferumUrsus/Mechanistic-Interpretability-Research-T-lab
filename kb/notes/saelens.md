# SAELens (SAE Lens)

## (a) Full citation + URL

Joseph Bloom, Curt Tigges, Anthony Duong, David Chanin. **"SAELens"** (software package/training-and-analysis codebase for sparse autoencoders on language models), 2024–present.

- Repo (current canonical location, `jbloomAus/SAELens` redirects here): https://github.com/decoderesearch/SAELens (older references / the task's stated URL `jbloomAus/SAELens` alias to the same repo)
- Docs: https://decoderesearch.github.io/SAELens/ (older mirror: https://jbloomaus.github.io/SAELens/)
- Pretrained SAE table: https://decoderesearch.github.io/SAELens/latest/pretrained_saes/
- Citation (from README):
  ```bibtex
  @misc{bloom2024saetrainingcodebase,
     title = {SAELens},
     author = {Bloom, Joseph and Tigges, Curt and Duong, Anthony and Chanin, David},
     year = {2024},
     howpublished = {\url{https://github.com/decoderesearch/SAELens}},
  }
  ```
- No accompanying arXiv paper — this is a software library, not a paper. The GPT-2-small residual-stream SAEs it hosts are documented in a LessWrong post: Joseph Bloom, ["Open Source Sparse Autoencoders for all Residual Stream Layers of GPT2-Small"](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream) (2024).

## (b) Problem addressed

Provides a standard, maintained interface for **training**, **loading pretrained**, and **analyzing** sparse autoencoders on language-model activations, with first-class integration into TransformerLens (`HookedSAETransformer`) so SAEs can be attached at any named hook point and used inside `run_with_hooks`/`run_with_cache`. Directly relevant to the task as the source of the GPT-2-small residual-stream SAE decoder columns you'd use as steering vectors `v`.

## (c) Key facts extracted for the task

### Release id: **`gpt2-small-res-jb`**
Confirmed by inspecting `sae_lens/pretrained_saes.yaml` in the repo (fetched directly, ~43.6k lines total, this release starts at line 36055):

```yaml
gpt2-small-res-jb:
  config_overrides:
    model_from_pretrained_kwargs:
      center_writing_weights: true
  links:
    dashboards: https://www.neuronpedia.org/gpt2sm-res-jb
    model: https://huggingface.co/gpt2
    publication: https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream
  model: gpt2-small
  repo_id: jbloom/GPT2-Small-SAEs-Reformatted
  saes:
    - {id: blocks.0.hook_resid_pre,  l0: 10.0, neuronpedia: gpt2-small/0-res-jb,  variance_explained: 0.999}
    - {id: blocks.1.hook_resid_pre,  l0: 10.0, neuronpedia: gpt2-small/1-res-jb,  variance_explained: 0.999}
    - {id: blocks.2.hook_resid_pre,  l0: 18.0, neuronpedia: gpt2-small/2-res-jb,  variance_explained: 0.999}
    - {id: blocks.3.hook_resid_pre,  l0: 23.0, neuronpedia: gpt2-small/3-res-jb,  variance_explained: 0.999}
    - {id: blocks.4.hook_resid_pre,  l0: 31.0, neuronpedia: gpt2-small/4-res-jb,  variance_explained: 0.9}
    - {id: blocks.5.hook_resid_pre,  l0: 41.0, neuronpedia: gpt2-small/5-res-jb,  variance_explained: 0.9}
    - {id: blocks.6.hook_resid_pre,  l0: 51.0, neuronpedia: gpt2-small/6-res-jb,  variance_explained: 0.9}
    - {id: blocks.7.hook_resid_pre,  l0: 54.0, neuronpedia: gpt2-small/7-res-jb,  variance_explained: 0.9}
    - {id: blocks.8.hook_resid_pre,  l0: 60.0, neuronpedia: gpt2-small/8-res-jb,  variance_explained: 0.9}
    - {id: blocks.9.hook_resid_pre,  l0: 70.0, neuronpedia: gpt2-small/9-res-jb,  variance_explained: 0.77}
    - {id: blocks.10.hook_resid_pre, l0: 52.0, neuronpedia: gpt2-small/10-res-jb, variance_explained: 0.77}
    - {id: blocks.11.hook_resid_pre, l0: 56.0, neuronpedia: gpt2-small/11-res-jb, variance_explained: 0.77}
    - {id: blocks.11.hook_resid_post, l0: 70.0, neuronpedia: gpt2-small/12-res-jb, variance_explained: 0.77}
```

**Critical hook-point detail**: this release trains one SAE per `blocks.{i}.hook_resid_pre` for `i = 0..11`, **plus one extra SAE at `blocks.11.hook_resid_post`** (i.e. the very final residual stream, after the last block). It does **not** provide `hook_resid_post` for the middle layers directly. Since (for a standard, attention-then-MLP, residual-stream architecture like GPT-2) `blocks.i.hook_resid_post` is numerically identical to `blocks.(i+1).hook_resid_pre` (nothing happens to the residual stream between the end of block $i$ and the start of block $i+1$ — it's the same tensor, no LayerNorm/reprojection in between in TransformerLens's convention), **the SAE trained on `blocks.7.hook_resid_pre` is what you want if you are steering "after middle layer 6" (i.e. at `blocks.6.hook_resid_post`)** for GPT-2 small (12 layers, indices 0–11, so layer 6 is the middle-ish layer). Confirm this equivalence empirically before relying on it (cache both hook names for one forward pass and diff them) — `[UNVERIFIED]` beyond the general TransformerLens convention.

### SAE config for `blocks.6.hook_resid_pre` (fetched from the actual HF `cfg.json`, `jbloom/GPT2-Small-SAEs-Reformatted` repo)
```json
{
  "model_name": "gpt2-small",
  "hook_point": "blocks.6.hook_resid_pre",
  "hook_point_layer": 6,
  "dataset_path": "Skylion007/openwebtext",
  "context_size": 128,
  "d_in": 768,
  "expansion_factor": 32,
  "d_sae": 24576,
  "l1_coefficient": 8e-05,
  "lr": 0.0004,
  "lr_warm_up_steps": 5000,
  "train_batch_size": 4096,
  "total_training_tokens": 300000000,
  "b_dec_init_method": "geometric_median"
}
```
So: **`d_in = 768`** (GPT-2 small residual width), **`expansion_factor = 32`** → **`d_sae = 24576`** latents, trained for 300M tokens on OpenWebText (`Skylion007/openwebtext`), context size 128, L1-penalty SAE (not TopK — this is the original 2024 "standard"/ReLU+L1 SAE architecture, `l1_coefficient=8e-5`, `lr=4e-4`). Average L0 (number of active latents per token) for layer 6 is **51.0** per the pretrained_saes.yaml table above (varies 10–70 across layers, generally increasing with depth).

### Minimal `SAE.from_pretrained` code (verbatim, from `tutorials/basic_loading_and_analysing.ipynb` in the repo)
```python
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained("gpt2-small", device=device)

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",       # see sae_lens/pretrained_saes.yaml for all release ids
    sae_id="blocks.8.hook_resid_pre",  # swap for "blocks.7.hook_resid_pre" to match resid_post of layer 6
    device=device,
)
```
(Note: current SAELens versions return just the `sae` object, not a `(sae, cfg_dict, sparsities)` tuple as in some older tutorials you may find online — check your installed version's return signature.)

### Encode/decode and getting decoder directions (W_dec)
```python
feature_acts = sae.encode(cache[sae.cfg.metadata.hook_name])   # -> [batch, pos, d_sae]
sae_out = sae.decode(feature_acts)                              # -> [batch, pos, d_in]
```
Verified from `sae_lens/saes/standard_sae.py` source:
```python
def encode(self, x):
    hidden_pre = self.hook_sae_acts_pre(x @ self.W_enc + self.b_enc)
    ...
def decode(self, feature_acts):
    sae_out_pre = feature_acts @ self.W_dec + self.b_dec
    ...
```
So **`sae.W_dec` has shape `[d_sae, d_in]` = `[24576, 768]`**, and **row `sae.W_dec[feature_idx]` is exactly the decoder direction for that feature in the model's residual-stream space** — this is literally your steering vector `v` for a given SAE feature. `sae.b_dec` (shape `[d_in]`) is the decoder bias (do not add it when using a feature direction as a steering vector — only used for reconstruction). Typical usage: normalize (`v = sae.W_dec[idx] / sae.W_dec[idx].norm()`) before scaling by your own alpha, since raw decoder-column norms vary across features/layers.

### Neuronpedia feature links
Every SAE + feature index has a corresponding Neuronpedia dashboard URL of the form:
```
https://neuronpedia.org/gpt2-small/{layer}-res-jb/{feature_idx}
```
e.g. layer 6, feature 14057 → `https://neuronpedia.org/gpt2-small/6-res-jb/14057`. Programmatically, the repo provides:
```python
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list
neuronpedia_quick_list = get_neuronpedia_quick_list(sae, feature_idx_list)
```
which builds a batch-viewer link for a list of feature indices on a given `sae` object (used in the tutorial to open a multi-feature dashboard in one click). Dashboard-level release name is `gpt2sm-res-jb` (https://www.neuronpedia.org/gpt2sm-res-jb).

### Reconstruction / L0 sanity check pattern (from the tutorial, reusable as-is)
```python
sae.eval()
with torch.no_grad():
    _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
    feature_acts = sae.encode(cache[sae.cfg.metadata.hook_name])
    sae_out = sae.decode(feature_acts)
    l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()   # exclude BOS token
    print("average l0", l0.mean().item())

from functools import partial
def reconstr_hook(activation, hook, sae_out):
    return sae_out
print("Orig", model(batch_tokens, return_type="loss").item())
print("reconstr", model.run_with_hooks(
    batch_tokens, fwd_hooks=[(sae.cfg.metadata.hook_name, partial(reconstr_hook, sae_out=sae_out))],
    return_type="loss").item())
```
This is exactly the "Delta LM Loss" pattern used by GLP (see `kb/notes/glp-generative-latent-prior.md`) — swap `sae_out` for your own denoiser's output `D(h + alpha*v)` to compute the equivalent metric for your task.

## (d) Evaluation protocol notes
SAELens itself defines no fixed "benchmark" — the tutorial demonstrates L0, reconstruction-loss-delta ("Delta LM Loss" style), and a specific-capability test (IOI-style prompt, e.g. "When John and Mary went to the shops, John gave the bag to ___") checked via `utils.test_prompt` before/after SAE reconstruction. SAEBench (https://github.com/adamkarvonen/SAEBench, linked from the README) is the more formal companion benchmark suite if you need a broader standard eval later.

## (e) Directly reusable techniques / pitfalls

1. **Use `blocks.7.hook_resid_pre`'s SAE (not a nonexistent `blocks.6.hook_resid_post`) to get decoder directions for "after layer 6" in GPT-2 small** — verify the resid_post(i)==resid_pre(i+1) identity empirically first.
2. **`sae.W_dec[feature_idx]` (raw, not normalized) is your ready-made steering vector `v`** for any of the 24,576 layer-6-ish features; normalize before applying your own alpha since raw column norms differ per feature.
3. **This is an L1/ReLU SAE, not TopK** — average L0 (~51 at layer 6) tells you roughly how many features are "on" per token; if you pick a feature with unusually high per-token activation variance it may fire inconsistently, worth checking `sparsity.safetensors` (also hosted in the HF repo) before picking a feature to steer with.
4. **`center_writing_weights: true`** is a required `model_from_pretrained_kwargs` override baked into the release config — if you load `gpt2-small` yourself via `HookedTransformer.from_pretrained` without SAELens driving the load, make sure you pass the same option or activations won't match what the SAE was trained on.
5. **Neuronpedia gives you free, pre-computed human-readable descriptions per feature** — useful as ground truth for an LLM-judge "does this generation match feature X's description" concept-presence metric (this is literally how the GLP paper scores its SAE-feature-steering Pareto front — see `kb/notes/glp-generative-latent-prior.md` §d, item 3).
6. **Pitfall**: don't confuse `sae_id` (the *training* hook point per the yaml, e.g. `"blocks.8.hook_resid_pre"`) with the hook point you might *steer* at at inference — SAELens's own docstring in the tutorial explicitly warns "won't always be a hook point" for `sae_id` in other releases, i.e. treat the string as an opaque identifier and always cross-check against `sae.cfg.metadata.hook_name` for the actual hook to attach to at runtime.
