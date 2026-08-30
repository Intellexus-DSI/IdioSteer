import os, json, collections, math
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
REPO = os.path.expanduser(".")
JV14 = os.path.join(REPO, "data_outputs", "idiom_evaluations", "judge_v14")
T3 = os.path.join(REPO, "concreteness", "tier3"); CR = os.path.join(REPO, "camera_ready")
LABELS = ["figurative", "literal", "incoherent"]
val = json.load(open(os.path.join(JV14, "judged_three_methods_eval_validated_FULL.json")))
bry = json.load(open(os.path.join(T3, "judged_v_BRY.json")))
perp = json.load(open(os.path.join(T3, "judged_v_PERP.json")))

def rates(doc, conds):
    cs = set(conds); lit = coh = n = 0
    for r in doc["results"]:
        for c, jl in (r.get("judge_v14") or {}).items():
            if c not in cs: continue
            for j in jl:
                if not j: continue
                lab = j.get("label"); n += 1
                lit += (lab == "literal"); coh += (lab in ("figurative", "literal"))
    return lit/n, coh/n, n

print("## Item 2 — literal & coherence at headline points (Llama-3.2-3B, all-gens denom)")
print(f"{'point':28} {'literal':>8} {'coherence':>10} {'n':>5}")
for name, doc, cond in [("baseline", val, "baseline"), ("v_IDIOM (mean_diff) a=-10", val, "mean_diff_alpha_-10"),
                        ("v_BRY a=+10", bry, "v_BRY_raw_+10"), ("v_PERP a=-10", perp, "v_PERP_raw_-10")]:
    l, c, n = rates(doc, [cond]); print(f"{name:28} {l:8.3f} {c:10.3f} {n:5}")

print("\n## Item 3 — corrected literal at best-alpha (batch-C v3 confusion)")
key3 = pd.read_csv(os.path.join(REPO, "judge_audit", "v3", "v3_key.csv"))
g3 = pd.read_csv(os.path.join(REPO, "judge_audit", "v3", "v3_resolved_GT.csv"))
vdf = key3.merge(g3, on="blind_id"); vdf["gold"] = vdf["human_label"].str.lower(); vdf["judge"] = vdf["judge_label"].str.lower()
def Pgj(st):
    s = vdf[vdf["stratum"] == st]; cm = confusion_matrix(s["gold"], s["judge"], labels=LABELS).astype(float)
    col = cm.sum(0); return np.divide(cm, col, out=np.zeros_like(cm), where=col > 0), len(s)
def jdist(doc, conds):
    cs = set(conds); c = collections.Counter()
    for r in doc["results"]:
        for cd, jl in (r.get("judge_v14") or {}).items():
            if cd in cs:
                for j in jl:
                    if j: c[j.get("label", "figurative")] += 1
    t = sum(c.values()); return {l: c.get(l, 0)/t for l in LABELS}
li = LABELS.index("literal")
def corr(doc, conds, st):
    P, nA = Pgj(st); Pj = jdist(doc, conds)
    return sum(Pj[LABELS[j]]*P[li, j] for j in range(len(LABELS))), nA
print(f"{'condition':22} {'raw literal':>12} {'corrected':>10} {'confusion N':>12}")
for name, doc, conds, st in [("v_BRY a=+10 (single)", bry, ["v_BRY_raw_+10"], "v_BRY"),
                             ("v_BRY pooled +10/15/20", bry, ["v_BRY_raw_+10", "v_BRY_raw_+15", "v_BRY_raw_+20"], "v_BRY"),
                             ("v_PERP a=-10 (single)", perp, ["v_PERP_raw_-10"], "v_PERP"),
                             ("v_PERP pooled -10/-15", perp, ["v_PERP_raw_-10", "v_PERP_raw_-15"], "v_PERP")]:
    raw = jdist(doc, conds)["literal"]; c, nA = corr(doc, conds, st)
    print(f"{name:22} {raw:12.3f} {c:10.3f} {nA:12}")

print("\n## Item 4 — p7 per-idiom (full, sorted by v_BRY - v_PERP)")
p7 = pd.read_csv(os.path.join(CR, "p7_per_idiom_components.csv"))
cols = ["idiom", "baseline", "vIDIOM", "vIDIOM_lo", "vIDIOM_hi", "vIDIOM_n",
        "vPERP", "vPERP_lo", "vPERP_hi", "vPERP_n", "vBRY", "vBRY_lo", "vBRY_hi", "vBRY_n"]
print(p7[cols].round(3).to_string(index=False))
for idi in ["break the ice", "lie low"]:
    r = p7[p7.idiom == idi]
    if len(r): print(f"  CHECK {idi}: v_BRY={float(r['vBRY'].iloc[0]):.3f} [{float(r['vBRY_lo'].iloc[0]):.3f},{float(r['vBRY_hi'].iloc[0]):.3f}] n={int(r['vBRY_n'].iloc[0])}")
print("\np7 figure files present:", [f for f in os.listdir(CR) if f.startswith("p7")])
