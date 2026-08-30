"""Items 3-5: unrelated-contrast control (raw cos), whitened geometry, whitened v_BRY re-gate.
Usage: outlier_345.py <key>. Re-extracts acts (forward-pass only)."""
import os, sys, json
import numpy as np, torch, pandas as pd
CONC = os.path.expanduser("./concreteness"); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from idiosteer.pipeline import MODELS, collect_activations, build_md_vector, find_last_span_pos, build_frame_samples, IDIOLINK
from transformers import AutoModelForCausalLM, AutoTokenizer
KEY = sys.argv[1]; M = MODELS[KEY]; Ls = M["Ls"]; SEED = 42
def cos(a, b): a = a/(np.linalg.norm(a)+1e-8); b = b/(np.linalg.norm(b)+1e-8); return float(np.dot(a, b))
def md(pos, neg): return pos.mean(0) - neg.mean(0)
rw = lambda p: [w.strip() for w in open(f"{CONC}/wordlists/{p}") if w.strip()]
GRP_A = {"Politics", "Business"}; GRP_B = {"Sport", "Entertainment"}
DTYPE = torch.bfloat16 if KEY in ("gemma3_4b", "gemma3_12b") else torch.float16

tok = AutoTokenizer.from_pretrained(M["hf"])
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(M["hf"], torch_dtype=DTYPE, device_map="auto", attn_implementation="sdpa").eval()

# IdioLink acts (+ fig/lit + subject)
df = pd.read_csv(IDIOLINK); tr = df[df["split"] == "train"]
Isamp, lab, subj = [], [], []
for _, r in tr.iterrows():
    ids, pos = find_last_span_pos(tok, r["sentence"], r["idiom"])
    if ids is None or pos is None: continue
    Isamp.append({"ids": ids, "pos": pos}); lab.append(1 if r["type"] == "figurative" else 0); subj.append(str(r["subject"]))
I = collect_activations(model, tok, Isamp, Ls, tag="I", log_every=0); lab = np.array(lab); subj = np.array(subj)
# frame acts (+ conc/abst + word)
cw, aw = rw("concrete_250.txt"), rw("abstract_250.txt")
cs, cmeta, _ = build_frame_samples(tok, cw); as_, ameta, _ = build_frame_samples(tok, aw)
C = collect_activations(model, tok, cs, Ls, tag="C", log_every=0); A = collect_activations(model, tok, as_, Ls, tag="A", log_every=0)
cwords = [m["word"] for m in cmeta]; awords = [m["word"] for m in ameta]

v_idiom = md(I[lab == 1], I[lab == 0]); v_bry = md(C, A)
saved_i = np.load(f"{CONC}/vectors/{KEY}/v_IDIOM.npy").astype(np.float64); saved_b = np.load(f"{CONC}/vectors/{KEY}/v_BRY.npy").astype(np.float64)

# item 3: unrelated subject-category MD (raw cos)
mA = np.isin(subj, list(GRP_A)); mB = np.isin(subj, list(GRP_B))
v_unrel = md(I[mA], I[mB])
r3 = {"cos_idiom_unrel_raw": cos(v_idiom, v_unrel), "cos_bry_unrel_raw": cos(v_bry, v_unrel),
      "nA": int(mA.sum()), "nB": int(mB.sum())}

# item 4: whitened geometry (pooled per-dim z-score over IdioLink + frame acts)
pool = np.vstack([I, C, A]); mu = pool.mean(0); sd = pool.std(0) + 1e-6
def z(x): return (x - mu)/sd
Iz, Cz, Az = z(I), z(C), z(A)
wv_idiom = md(Iz[lab == 1], Iz[lab == 0]); wv_bry = md(Cz, Az)
r4 = {"whitened_cos_idiom_bry": cos(wv_idiom, wv_bry)}

# item 5: whitened v_BRY held-out gate (word split on z-scored frame acts)
rng = np.random.default_rng(SEED)
def wsplit(words):
    u = sorted(set(words)); p = rng.permutation(len(u)); h = len(u)//2
    SA = {u[i] for i in p[:h]}; SB = {u[i] for i in p[h:]}
    return (np.array([i for i, w in enumerate(words) if w in SA]), np.array([i for i, w in enumerate(words) if w in SB]))
cA, cB = wsplit(cwords); aA, aB = wsplit(awords)
v1 = md(Cz[cA], Az[aA]); v2 = md(Cz[cB], Az[aB]); hc = cos(v1, v2)
v1u = v1/np.linalg.norm(v1)
tr_s = np.r_[Cz[cA] @ v1u, Az[aA] @ v1u]; trl = np.r_[np.ones(len(cA)), np.zeros(len(aA))]
te_s = np.r_[Cz[cB] @ v1u, Az[aB] @ v1u]; tel = np.r_[np.ones(len(cB)), np.zeros(len(aB))]
cand = np.unique(tr_s); mids = (cand[:-1]+cand[1:])/2 if len(cand) > 1 else cand
bt, ba = (mids[0] if len(mids) else 0.0), -1
for t in mids:
    a = ((tr_s > t).astype(int) == trl).mean()
    if a > ba: ba, bt = a, t
acc = float(((te_s > bt).astype(int) == tel).mean())
r5 = {"whitened_vBRY_half_cos": hc, "whitened_vBRY_heldout_acc": acc, "gate_4a_pass": hc >= 0.7, "gate_4b_pass": acc >= 0.85}

res = {"model": KEY, "Ls": Ls, "dtype": str(DTYPE),
       "sanity_cos_idiom_vs_saved": cos(v_idiom, saved_i), "sanity_cos_bry_vs_saved": cos(v_bry, saved_b),
       "cos_idiom_bry_raw": cos(v_idiom, v_bry), "item3_unrelated": r3, "item4_whitened": r4, "item5_whitened_gate": r5}
json.dump(res, open(f"{CONC}/results/{KEY}_outlier345.json", "w"), indent=2)
print(json.dumps(res, indent=2))
