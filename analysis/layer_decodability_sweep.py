"""Per-layer probe sweep (Appendix-D): IdioLink-train activations at final idiom token,
all hidden_states layers, logistic probe (LR C=1.0 lbfgs), 5-fold StratifiedGroupKFold-by-idiom.
Usage: layer_sweep.py <model_key>   (llama3b default). Writes layer_decodability/<key>/layer_vs_accuracy.csv
"""
import os, sys, time
import numpy as np, pandas as pd, torch
CONC = os.path.expanduser("./concreteness")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from idiosteer.pipeline import MODELS, load_model_and_tokenizer, find_last_span_pos, IDIOLINK, DEVICE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

KEY = sys.argv[1] if len(sys.argv) > 1 else "llama3b"
M = MODELS[KEY]
OUT = os.path.expanduser(f"./layer_decodability/{KEY}"); os.makedirs(OUT, exist_ok=True)

print(f"Loading {M['hf']} ...", flush=True)
model, tok = load_model_and_tokenizer(M["hf"])
n_hs = model.config.num_hidden_layers + 1     # hidden_states 0..num_hidden_layers
print(f"hidden_states layers: 0..{n_hs-1}", flush=True)

df = pd.read_csv(IDIOLINK)
train = df[df["split"] == "train"]
# reuse find_last_span_pos; one forward pass per sample -> all layers at final idiom token
acts = [[] for _ in range(n_hs)]; ys = []; groups = []; drop = 0
t0 = time.time()
with torch.no_grad():
    for i, (_, row) in enumerate(train.iterrows()):
        ids, pos = find_last_span_pos(tok, row["sentence"], row["idiom"])
        if ids is None or pos is None:
            drop += 1; continue
        out = model(torch.tensor([ids], device=DEVICE), output_hidden_states=True)
        for L in range(n_hs):
            acts[L].append(out.hidden_states[L][0, pos, :].cpu().float().numpy())
        ys.append(1 if row["type"] == "figurative" else 0)
        groups.append(row["idiom"])
        if (i + 1) % 300 == 0:
            print(f"  extract {i+1}/{len(train)} ({time.time()-t0:.0f}s)", flush=True)
y = np.array(ys); groups = np.array(groups)
print(f"extraction done: N={len(y)} (dropped {drop}), fig={int(y.sum())} lit={int((y==0).sum())}, "
      f"idioms={len(set(groups))}, {time.time()-t0:.0f}s", flush=True)

rows = []
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for L in range(n_hs):
    X = np.stack(acts[L])
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    scores = cross_val_score(clf, X, y, cv=sgkf, groups=groups, scoring="accuracy")
    rows.append({"hidden_state_layer": L, "physical_layer": L - 1,
                 "cv_accuracy": float(np.mean(scores)), "cv_std": float(np.std(scores)),
                 "n_samples": len(y)})
    print(f"  hs[{L:2d}] (L{L-1:>2}): acc={np.mean(scores):.4f} ± {np.std(scores):.4f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "layer_vs_accuracy.csv"), index=False)
peak = res.loc[res["cv_accuracy"].idxmax()]
print(f"\nPEAK: hs[{int(peak.hidden_state_layer)}] acc={peak.cv_accuracy:.4f}")
if KEY == "llama3b":
    l14 = res[res.hidden_state_layer == 14]["cv_accuracy"].iloc[0]
    print(f"SANITY hs[14] = {l14:.4f} (expect 0.975 ± ~0.01): {'PASS' if abs(l14-0.975)<=0.015 else 'CHECK'}")
print(f"saved {os.path.join(OUT,'layer_vs_accuracy.csv')}")
