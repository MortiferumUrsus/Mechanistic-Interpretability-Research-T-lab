import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

print("--- model ---")
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
print(model.cfg.n_layers, model.cfg.d_model)
print([n for n in model.hook_dict if "blocks.6" in n or "blocks.7.hook_resid" in n])

print("--- sae ---")
import inspect

print(inspect.signature(SAE.from_pretrained))
sae = SAE.from_pretrained("gpt2-small-res-jb", "blocks.7.hook_resid_pre", device="cuda")
if isinstance(sae, tuple):
    print("tuple len", len(sae))
    sae = sae[0]
print(type(sae))
print("cfg:", sae.cfg)
print("W_dec", sae.W_dec.shape, "W_enc", sae.W_enc.shape)
print("hook", sae.cfg.metadata.hook_name if hasattr(sae.cfg, "metadata") else "n/a")

toks = model.to_tokens(["The Eiffel Tower is located in the city of"])
with torch.no_grad():
    _, cache = model.run_with_cache(toks, names_filter="blocks.7.hook_resid_pre")
h = cache["blocks.7.hook_resid_pre"]
print("h", h.shape, "norm", h.norm(dim=-1).mean().item())
feats = sae.encode(h)
print("feats", feats.shape, "nnz/token", (feats > 0).float().sum(-1).mean().item())
top = feats[0, -1].topk(10)
print("top feats", top.indices.tolist(), [round(v, 2) for v in top.values.tolist()])
