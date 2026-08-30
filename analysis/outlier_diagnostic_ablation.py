"""Items 1-2: outlier profile + ablated cosine, from saved MD vectors (scale-invariant)."""
import os, json, numpy as np
CONC = os.path.expanduser("./concreteness")
MODELS = ["llama3b", "llama8b", "gemma2_2b", "gemma3_4b", "gemma3_12b"]
def u(x): return x/(np.linalg.norm(x)+1e-8)
def cos(a, b): return float(np.dot(u(a), u(b)))
out = {}
for k in MODELS:
    vi = np.load(f"{CONC}/vectors/{k}/v_IDIOM.npy").astype(np.float64)
    vb = np.load(f"{CONC}/vectors/{k}/v_BRY.npy").astype(np.float64)
    print(f"\n=== {k} (d={len(vi)}) — cos(v_IDIOM,v_BRY)={cos(vi,vb):+.4f} ===")
    prof = {}
    for name, v in [("v_IDIOM", vi), ("v_BRY", vb)]:
        sq = v**2; tot = sq.sum(); order = np.argsort(sq)[::-1]
        top10 = order[:10]
        frac10 = float(sq[top10].sum()/tot)
        print(f"  {name}: top-10 dims carry {frac10:.1%} of squared norm | top1={sq[order[0]]/tot:.1%} | dims={top10.tolist()}")
        prof[name] = {"top10_frac_sqnorm": frac10, "top1_frac": float(sq[order[0]]/tot), "top10_dims": top10.tolist()}
    # ablated cosine: union top-k dims across both vectors
    abl = {}
    si = np.argsort(vi**2)[::-1]; sb = np.argsort(vb**2)[::-1]
    print(f"  ablated cos(v_IDIOM,v_BRY) after zeroing union top-k:")
    for kk in [1, 2, 5, 10, 50]:
        union = set(si[:kk].tolist()) | set(sb[:kk].tolist())
        m = np.ones(len(vi)); m[list(union)] = 0
        c = cos(vi*m, vb*m)
        abl[kk] = {"cos": round(c, 4), "n_dims_zeroed": len(union)}
        print(f"    k={kk:2} (|union|={len(union):3}): cos={c:+.4f}")
    out[k] = {"cos_raw": cos(vi, vb), "profile": prof, "ablated": abl}
json.dump(out, open(f"{CONC}/../camera_ready/p9_outlier_items12.json", "w"), indent=2)
print("\nsaved camera_ready/p9_outlier_items12.json")
