"""Whitened (and raw sanity) word-level gate for bare-word v_GJ. Usage: <key>"""
import os, sys, json
import numpy as np, torch
CONC = os.path.expanduser("./concreteness"); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from idiosteer.pipeline import MODELS, collect_activations, build_md_vector
from transformers import AutoModelForCausalLM, AutoTokenizer
KEY = sys.argv[1]; M = MODELS[KEY]; Ls = M["Ls"]; SEED = 42
DTYPE = torch.bfloat16 if KEY in ("gemma3_4b", "gemma3_12b") else torch.float16
rw = lambda p: [w.strip() for w in open(f"{CONC}/wordlists/{p}") if w.strip()]
def cos(a, b): a = a/(np.linalg.norm(a)+1e-8); b = b/(np.linalg.norm(b)+1e-8); return float(np.dot(a, b))
def md(p, n): return p.mean(0)-n.mean(0)

tok = AutoTokenizer.from_pretrained(M["hf"])
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(M["hf"], torch_dtype=DTYPE, device_map="auto", attn_implementation="sdpa").eval()
def bw(words): return [{"ids": tok(w, add_special_tokens=True).input_ids, "pos": len(tok(w, add_special_tokens=True).input_ids)-1} for w in words]
cwords, awords = rw("gj_concrete_500.txt"), rw("gj_abstract_500.txt")
C = collect_activations(model, tok, bw(cwords), Ls, tag="gc", log_every=0)
A = collect_activations(model, tok, bw(awords), Ls, tag="ga", log_every=0)

def gate(Cx, Ax):
    rng = np.random.default_rng(SEED)
    ci = rng.permutation(len(Cx)); ai = rng.permutation(len(Ax))
    cA, cB = ci[:len(Cx)//2], ci[len(Cx)//2:]; aA, aB = ai[:len(Ax)//2], ai[len(Ax)//2:]
    v1 = md(Cx[cA], Ax[aA]); v2 = md(Cx[cB], Ax[aB]); hc = cos(v1, v2)
    v1u = v1/np.linalg.norm(v1)
    tr = np.r_[Cx[cA] @ v1u, Ax[aA] @ v1u]; trl = np.r_[np.ones(len(cA)), np.zeros(len(aA))]
    te = np.r_[Cx[cB] @ v1u, Ax[aB] @ v1u]; tel = np.r_[np.ones(len(cB)), np.zeros(len(aB))]
    cand = np.unique(tr); mids = (cand[:-1]+cand[1:])/2 if len(cand) > 1 else cand
    bt, ba = (mids[0] if len(mids) else 0.0), -1
    for t in mids:
        a = ((tr > t).astype(int) == trl).mean()
        if a > ba: ba, bt = a, t
    return {"half_cos": round(hc, 4), "heldout_acc": round(float(((te > bt).astype(int) == tel).mean()), 4)}
pool = np.vstack([C, A]); mu = pool.mean(0); sd = pool.std(0)+1e-6
raw = gate(C, A); whit = gate((C-mu)/sd, (A-mu)/sd)
res = {"model": KEY, "raw_gjbw": raw, "whitened_gjbw": whit}
json.dump(res, open(f"{CONC}/results/{KEY}_gjbw_whiten.json", "w"), indent=2)
print(KEY, json.dumps(res))
