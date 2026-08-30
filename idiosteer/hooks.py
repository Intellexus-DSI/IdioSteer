"""Activation-steering hooks + generation. Version-agnostic (tensor or tuple layer output)."""
import math, torch
class AdditiveSteeringHook:
    """Add alpha * (unit) vector at target_pos of layer `layer` output (prefill only; KV-cache safe)."""
    def __init__(self, model, layer, vector, alpha, target_pos):
        self.vector, self.alpha, self.target_pos = vector, alpha, target_pos
        self._h = model.model.layers[layer].register_forward_hook(self._hook)
    def _hook(self, module, inp, out):
        is_tuple = isinstance(out, tuple); h = out[0] if is_tuple else out
        if self.target_pos >= h.shape[1]: return out
        h2 = h.clone(); h2[:, self.target_pos, :] += self.alpha * self.vector
        return ((h2,) + tuple(out[1:])) if is_tuple else h2
    def remove(self): self._h.remove()

class AngularSteeringHook:
    """Rotate h at target_pos by theta in the (v, rejection) plane (norm-preserving)."""
    def __init__(self, model, layer, vector, theta, target_pos, device):
        self.target_pos, self.cos_t, self.sin_t = target_pos, math.cos(theta), math.sin(theta)
        v = vector.float(); self.v = (v/(v.norm()+1e-8)).to(vector.dtype).to(device)
        self._h = model.model.layers[layer].register_forward_hook(self._hook)
    def _hook(self, module, inp, out):
        is_tuple = isinstance(out, tuple); h = out[0] if is_tuple else out
        if self.target_pos >= h.shape[1]: return out
        p = self.target_pos; hp = h[:, p:p+1, :]
        c1 = (hp*self.v).sum(-1, keepdim=True); rej = hp - c1*self.v; c2 = rej.norm(dim=-1, keepdim=True)
        b2 = rej/(c2+1e-8); nc1 = c1*self.cos_t - c2*self.sin_t; nc2 = c1*self.sin_t + c2*self.cos_t
        h2 = h.clone(); h2[:, p:p+1, :] = nc1*self.v + nc2*b2
        return ((h2,) + tuple(out[1:])) if is_tuple else h2
    def remove(self): self._h.remove()

@torch.no_grad()
def generate_completions(model, tokenizer, input_ids, n=5, max_new=50, temperature=0.7, top_p=0.9):
    attn = torch.ones_like(input_ids)
    out = model.generate(input_ids, attention_mask=attn, max_new_tokens=max_new, do_sample=True,
                         temperature=temperature, top_p=top_p, num_return_sequences=n,
                         pad_token_id=tokenizer.eos_token_id)
    plen = input_ids.shape[1]
    return [tokenizer.decode(out[i][plen:], skip_special_tokens=True) for i in range(n)]
