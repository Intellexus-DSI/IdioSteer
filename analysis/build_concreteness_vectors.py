"""Build v_IDIOM, v_BRY, v_GJ_bareword + Fix-B WORD-split gates for one model.
Saves vectors (guards existing: asserts reproduction, no clobber) + results/{key}_conc.json.
Usage: build_conc_all.py <key>"""
import os, sys, json, hashlib
import numpy as np, torch
CONC = os.path.expanduser("./concreteness"); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from idiosteer.pipeline import (MODELS, collect_activations, build_md_vector,
                      build_idiom_samples, build_frame_samples, DEVICE)
from transformers import AutoModelForCausalLM, AutoTokenizer
def load_md(hf, dtype):
    tok = AutoTokenizer.from_pretrained(hf)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=dtype, device_map="auto", attn_implementation="sdpa")
    m.eval(); return m, tok
KEY = sys.argv[1]; M = MODELS[KEY]; Ls = M["Ls"]; SEED = 42
WL = os.path.join(CONC, "wordlists"); VDIR = os.path.join(CONC, "vectors", KEY); os.makedirs(VDIR, exist_ok=True)
RESD = os.path.join(CONC, "results"); os.makedirs(RESD, exist_ok=True)
rw = lambda p: [w.strip() for w in open(os.path.join(WL, p)) if w.strip()]
def cos(a, b): a = a/(np.linalg.norm(a)+1e-8); b = b/(np.linalg.norm(b)+1e-8); return float(np.dot(a, b))

def save_or_verify(name, vec):
    p = os.path.join(VDIR, name)
    if os.path.exists(p):
        old = np.load(p).astype(np.float64)
        c = cos(vec, old)
        assert c > 0.999, f"{name} mismatch vs saved (cos={c:.4f})"
        return "verified(existing)", float(c)
    np.save(p, vec.astype(np.float32)); return "saved(new)", 1.0

def word_split_gate(c_acts, c_words, a_acts, a_words):
    rng = np.random.default_rng(SEED)
    def split(words):
        u = sorted(set(words)); perm = rng.permutation(len(u)); half = len(u)//2
        A = {u[i] for i in perm[:half]}; B = {u[i] for i in perm[half:]}
        iA = np.array([i for i, w in enumerate(words) if w in A]); iB = np.array([i for i, w in enumerate(words) if w in B])
        return iA, iB
    cA, cB = split(c_words); aA, aB = split(a_words)
    v1, _ = build_md_vector(c_acts[cA], a_acts[aA]); v2, _ = build_md_vector(c_acts[cB], a_acts[aB])
    hc = cos(v1, v2)
    tr = np.concatenate([c_acts[cA] @ v1, a_acts[aA] @ v1]); trl = np.r_[np.ones(len(cA)), np.zeros(len(aA))]
    te = np.concatenate([c_acts[cB] @ v1, a_acts[aB] @ v1]); tel = np.r_[np.ones(len(cB)), np.zeros(len(aB))]
    cand = np.unique(tr); mids = (cand[:-1]+cand[1:])/2 if len(cand) > 1 else cand
    bt, ba = (mids[0] if len(mids) else 0.0), -1
    for t in mids:
        a = ((tr > t).astype(int) == trl).mean()
        if a > ba: ba, bt = a, t
    acc = float(((te > bt).astype(int) == tel).mean())
    return {"half_split_cos": hc, "heldout_acc": acc, "gate_4a_pass": hc >= 0.7, "gate_4b_pass": acc >= 0.85,
            "n_words": {"conc": len(set(c_words)), "abst": len(set(a_words))}}

DTYPE = torch.bfloat16 if KEY in ("gemma3_4b", "gemma3_12b") else torch.float16
print(f"Loading {M['hf']} (Ls={Ls}, dtype={DTYPE}) ...", flush=True)
model, tok = load_md(M["hf"], DTYPE)

# v_IDIOM
isamp, ilab, idrop = build_idiom_samples(tok)
iacts = collect_activations(model, tok, isamp, Ls, tag="idi")
v_idiom, idi_rn = build_md_vector(iacts[ilab == 1], iacts[ilab == 0])
# v_BRY (framed)
cw, aw = rw("concrete_250.txt"), rw("abstract_250.txt")
cs, cmeta, _ = build_frame_samples(tok, cw); as_, ameta, _ = build_frame_samples(tok, aw)
c_acts = collect_activations(model, tok, cs, Ls, tag="bry-c"); a_acts = collect_activations(model, tok, as_, Ls, tag="bry-a")
v_bry, bry_rn = build_md_vector(c_acts, a_acts)
c_words = [m["word"] for m in cmeta]; a_words = [m["word"] for m in ameta]
# v_GJ bareword
gcw, gaw = rw("gj_concrete_500.txt"), rw("gj_abstract_500.txt")
def bw(words): return [{"ids": tok(w, add_special_tokens=True).input_ids, "pos": len(tok(w, add_special_tokens=True).input_ids)-1} for w in words]
gc = collect_activations(model, tok, bw(gcw), Ls, tag="gj-c"); ga = collect_activations(model, tok, bw(gaw), Ls, tag="gj-a")
v_gjbw, gj_rn = build_md_vector(gc, ga)

s1 = save_or_verify("v_IDIOM.npy", v_idiom); s2 = save_or_verify("v_BRY.npy", v_bry); s3 = save_or_verify("v_GJ_bareword.npy", v_gjbw)

res = {
  "model_key": KEY, "hf": M["hf"], "Ls": Ls,
  "raw_norms": {"v_IDIOM": idi_rn, "v_BRY": bry_rn, "v_GJ_bareword": gj_rn},
  "vector_status": {"v_IDIOM": s1, "v_BRY": s2, "v_GJ_bareword": s3},
  "geometry": {"cos_idiom_bry": cos(v_idiom, v_bry), "cos_idiom_gjbw": cos(v_idiom, v_gjbw),
               "cos_gjbw_bry": cos(v_gjbw, v_bry)},
  "gates_v_BRY_wordsplit": word_split_gate(c_acts, c_words, a_acts, a_words),
  "gates_v_GJ_bareword_wordsplit": word_split_gate(gc, gcw, ga, gaw),
  "counts": {"idiom_fig": int((ilab == 1).sum()), "idiom_lit": int((ilab == 0).sum()), "idiom_drop": idrop},
}
json.dump(res, open(os.path.join(RESD, f"{KEY}_conc.json"), "w"), indent=2)
print(json.dumps({"geometry": res["geometry"], "vBRY_gate": res["gates_v_BRY_wordsplit"],
                  "vGJbw_gate": res["gates_v_GJ_bareword_wordsplit"], "status": res["vector_status"]}, indent=2))
print(f"[{KEY}] done")
