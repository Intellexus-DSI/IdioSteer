"""v_GJ BARE-WORD rebuild (geometry only, no generation).
Forward each GJ word ALONE (bare, add_special_tokens=True), take the final token's
residual at Ls, MD(concrete - abstract), unit-norm. Free of frame-grammaticality
artifacts. Saves v_GJ_bareword.npy; archives the framed v_GJ; prints Stage-3 cosines.
Usage: build_vgj_bareword.py <model_key>   (default llama3b)
"""
import os, sys, json, shutil
import numpy as np
CONC = os.path.expanduser("./concreteness")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from idiosteer.pipeline import MODELS, load_model_and_tokenizer, collect_activations, build_md_vector

KEY = sys.argv[1] if len(sys.argv) > 1 else "llama3b"
M = MODELS[KEY]; Ls = M["Ls"]
WL = os.path.join(CONC, "wordlists"); VDIR = os.path.join(CONC, "vectors", KEY)
ARCH = os.path.join(CONC, "vectors", KEY, "archive"); os.makedirs(ARCH, exist_ok=True)

def readwords(p): return [w.strip() for w in open(os.path.join(WL, p)) if w.strip()]

def bareword_samples(tok, words):
    s = []
    for w in words:
        ids = tok(w, add_special_tokens=True).input_ids   # [BOS, w_tok...]
        s.append({"ids": ids, "pos": len(ids) - 1})        # final token of the word
    return s

def cos(a, b):
    a = a/(np.linalg.norm(a)+1e-8); b = b/(np.linalg.norm(b)+1e-8); return float(np.dot(a, b))

print(f"Loading {M['hf']} (Ls={Ls}) ...", flush=True)
model, tok = load_model_and_tokenizer(M["hf"])
cw, aw = readwords("gj_concrete_500.txt"), readwords("gj_abstract_500.txt")
c_acts = collect_activations(model, tok, bareword_samples(tok, cw), Ls, tag="gjbw-c")
a_acts = collect_activations(model, tok, bareword_samples(tok, aw), Ls, tag="gjbw-a")
v_bw, rawnorm = build_md_vector(c_acts, a_acts)   # +concrete

# archive framed v_GJ, save bare-word
framed = os.path.join(VDIR, "v_GJ.npy")
if os.path.exists(framed):
    shutil.copy(framed, os.path.join(ARCH, "v_GJ_framed_deprecated.npy"))
np.save(os.path.join(VDIR, "v_GJ_bareword.npy"), v_bw)

v_idiom = np.load(os.path.join(VDIR, "v_IDIOM.npy")).astype(np.float64)
v_bry   = np.load(os.path.join(VDIR, "v_BRY.npy")).astype(np.float64)
v_gjf   = np.load(framed).astype(np.float64)
geo = {
    "model": KEY, "Ls": Ls, "v_GJ_bareword_raw_norm": rawnorm,
    "n_concrete": len(cw), "n_abstract": len(aw),
    "cos(v_IDIOM,+fig ; v_GJ_bareword,+conc)": round(cos(v_idiom, v_bw), 4),
    "cos(v_GJ_bareword,+conc ; v_BRY,+conc)": round(cos(v_bw, v_bry), 4),
    "cos(v_GJ_bareword ; v_GJ_framed) [old vs new]": round(cos(v_bw, v_gjf), 4),
}
json.dump(geo, open(os.path.join(VDIR, "v_GJ_bareword_geometry.json"), "w"), indent=2)
print(json.dumps(geo, indent=2))
print(f"saved v_GJ_bareword.npy; archived framed -> archive/v_GJ_framed_deprecated.npy")
