# TransformerLens

## (a) Full citation + URL

Neel Nanda et al. **TransformerLens** — a library for mechanistic interpretability of GPT-2-style language models (hooked activations, standardized naming, `HookedTransformer`).

- Repo: https://github.com/TransformerLensOrg/TransformerLens
- Docs: https://transformerlensorg.github.io/TransformerLens/
- Main demo: https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html
- No single citable paper; cite the repo. (Originally by Neel Nanda, now maintained under the TransformerLensOrg GitHub org; the library has recently undergone a "TransformerBridge" refactor — `TransformerBridge.boot_transformers()` — but the classic `HookedTransformer` API described below is still the primary interface and is what all the SAELens/persona_vectors style code above uses.)

## (b) Problem addressed
Standardizes activation access/patching across model internals via **named hook points** on every intermediate tensor (residual stream pre/mid/post per block, attention Q/K/V/pattern/z, MLP output, etc.), so steering/probing/caching code doesn't need model-specific plumbing.

## (c) Hook naming convention (verified against source, `transformer_lens/HookedRootModule.py` + `hook_points.py` + community references)

Every hook name follows the pattern `blocks.{layer}.{sub_hook_name}`, plus two ungated top-level hooks:
- `hook_embed` — token embeddings
- `hook_pos_embed` — positional embeddings

Per-block residual stream (the ones relevant to steering):
- `blocks.{L}.hook_resid_pre` — residual stream **entering** block L (= output of block L-1)
- `blocks.{L}.hook_resid_mid` — residual stream after attention, before MLP, within block L (GPT-2-style pre-LN blocks only)
- `blocks.{L}.hook_resid_post` — residual stream **leaving** block L (= same tensor as `blocks.{L+1}.hook_resid_pre`)

Per-block attention/MLP internals:
- `blocks.{L}.hook_attn_out`, `blocks.{L}.hook_mlp_out` — sub-block outputs before being added back to the residual stream
- `blocks.{L}.attn.hook_q` / `hook_k` / `hook_v` — per-head Q/K/V
- `blocks.{L}.attn.hook_pattern` — post-softmax attention weights
- `blocks.{L}.attn.hook_attn_scores` — pre-softmax attention scores
- `blocks.{L}.attn.hook_z` — per-head weighted value output (pre-`W_O`)

Hook names can also be constructed programmatically via `transformer_lens.utils.get_act_name(name, layer)` instead of hand-building strings.

**For "after the middle layer" of GPT-2 small (12 blocks, indices 0–11), the relevant hook is `blocks.6.hook_resid_post`** (equivalently `blocks.7.hook_resid_pre` — same tensor, see the SAELens note for why this matters when matching a pretrained SAE's hook point).

## (d) Hook mechanics + minimal steering snippet during `model.generate` (verified against source)

Three ways to attach hooks, in increasing order of persistence:

1. **`model.run_with_hooks(tokens, fwd_hooks=[(name, fn)], ...)`** — runs exactly **one forward pass** with temporary hooks, auto-removed after. Internally this is just `with model.hooks(fwd_hooks): return model.forward(...)`. **It only calls `.forward()` once — it does NOT drive multi-step generation.** Confirmed from `HookedRootModule.py`:
   ```python
   def run_with_hooks(self, *model_args, fwd_hooks=[], bwd_hooks=[], reset_hooks_end=True, clear_contexts=False, **model_kwargs):
       with self.hooks(fwd_hooks, bwd_hooks, reset_hooks_end, clear_contexts) as hooked_model:
           return hooked_model.forward(*model_args, **model_kwargs)
   ```

2. **`model.hooks(fwd_hooks=[...])`** — the actual context manager (source, `HookedRootModule.py`):
   ```python
   def hooks(self, fwd_hooks=[], bwd_hooks=[], reset_hooks_end=True, clear_contexts=False):
       """
       Example:
           with model.hooks(fwd_hooks=my_hooks):
               hooked_loss = model(text, return_type="loss")
       """
   ```
   **This is what you use to wrap a multi-step `model.generate(...)` call** — `HookedTransformer.generate(...)` (verified via source, `transformer_lens/HookedTransformer.py`, `def generate(...)`) takes no `fwd_hooks` parameter itself; hooks must already be registered (either via this context manager wrapping the call, or via `add_hook`/`reset_hooks` below) before `generate` runs its internal per-token forward-pass loop.

3. **`model.add_hook(name, fn)` / `model.reset_hooks()` / `model.remove_hook(name)`** — persistent hooks that stay active across multiple calls until explicitly removed; useful if you want the same steering hook active for many `generate()` calls in a batch/eval loop without re-entering a `with` block each time.

### Minimal reusable snippet: steer `blocks.6.hook_resid_post` during `model.generate`
```python
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2", device="cuda")  # GPT-2 small, 12 layers
HOOK_NAME = "blocks.6.hook_resid_post"

def make_steer_hook(v, alpha):
    def hook_fn(resid, hook):          # signature: (activation_tensor, hook: HookPoint) -> tensor
        return resid + alpha * v.to(resid.dtype)
    return hook_fn

v = ...      # e.g. sae.W_dec[feature_idx], normalized
alpha = 8.0

with model.hooks(fwd_hooks=[(HOOK_NAME, make_steer_hook(v, alpha))]):
    tokens = model.generate("The weather today is", max_new_tokens=40, do_sample=True, top_p=0.9)
print(model.to_string(tokens))
```

To instead run your denoiser `D` on the steered activation before it continues through the rest of the network (i.e. implement `h~ = D(h + alpha*v)` in-place at the hook), replace the return value with `D(resid + alpha * v)` inside `hook_fn` — the hook's return value **replaces** the tensor at that point in the forward pass, so this is sufficient to splice a trained denoiser directly into `model.generate`'s decoding loop with no other plumbing.

### Reconstruction/ablation hook pattern (from SAELens tutorial, reusable for Delta-LM-Loss-style eval)
```python
from functools import partial
def reconstr_hook(activation, hook, replacement):
    return replacement

loss_with_denoiser = model.run_with_hooks(
    tokens, return_type="loss",
    fwd_hooks=[(HOOK_NAME, partial(reconstr_hook, replacement=D(h_noisy)))],
)
```

## (e) Directly reusable techniques / pitfalls

1. **Use `with model.hooks(fwd_hooks=[...]):` around `model.generate(...)`, not `run_with_hooks`** — `run_with_hooks` only executes a single forward pass and will silently *not* apply your hook across the autoregressive decoding steps `generate` performs internally. This is the single most likely footgun when porting steering code from a "one forward pass" example to actual multi-token generation.
2. **`blocks.{L}.hook_resid_post` and `blocks.{L+1}.hook_resid_pre` are the identical tensor** in GPT-2's architecture — pick whichever name matches your pretrained SAE/persona-vector's convention rather than assuming resid_post always exists at your layer of interest (see SAELens note: the `gpt2-small-res-jb` release only ships `hook_resid_pre` at each layer, so layer-6-output steering needs the layer-7 `hook_resid_pre` SAE).
3. **Hook function signature is `fn(tensor, hook) -> tensor_or_None`** — returning `None` leaves the activation unmodified (useful for read-only logging hooks); returning a tensor **replaces** the activation at that point.
4. **`reset_hooks_end=True` is the default** on both `run_with_hooks` and the `hooks()` context manager — hooks are automatically cleaned up when the `with` block exits or `run_with_hooks` returns, so you don't need to manually call `reset_hooks()` unless you used `add_hook` directly outside a context manager.
5. **For batched generation with left-padding** (as in the persona_vectors repo's `sample_steering`), remember hooks apply to the *entire* batch tensor at that hook point — if you only want to steer non-padding positions, you need position-masking logic inside your hook function (see `ActivationSteerer._hook_fn`'s `positions="response"` handling for a worked example, in `kb/notes/persona-vectors.md`).
