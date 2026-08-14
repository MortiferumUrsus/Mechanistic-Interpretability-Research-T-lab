import torch
from transformer_lens import HookedTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)
print("model ok", model.cfg.n_layers, model.cfg.d_model)

_, cache = model.run_with_cache("The Eiffel Tower is located in", names_filter="blocks.6.hook_resid_post")
h = cache["blocks.6.hook_resid_post"]
print("resid", tuple(h.shape), "norm/token", h.norm(dim=-1).mean().item())

from sae_lens import SAE
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

d = get_pretrained_saes_directory()
cands = [k for k in d if "gpt2" in k.lower()]
print("gpt2 releases:", cands)
for r in cands:
    ids = list(d[r].saes_map.keys())
    print(" ", r, "->", ids[:14], "..." if len(ids) > 14 else "")
