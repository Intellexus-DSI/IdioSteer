import os, re, json, collections
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
REPO = os.path.expanduser(".")
JV14 = os.path.join(REPO, "data_outputs", "idiom_evaluations", "judge_v14")
LABELS = ["figurative", "literal", "incoherent"]
FULLS = {"Llama-3.2-3B": "judged_three_methods_eval_validated_FULL.json",
         "Llama-3.1-8B": "judged_three_methods_eval_llama8b_FULL.json",
         "Gemma-2-2B": "judged_three_methods_eval_gemma2_FULL.json",
         "Gemma-3-4B": "judged_three_methods_eval_gemma3_4b_FULL.json",
         "Gemma-3-12B": "judged_three_methods_eval_gemma3_12b_FULL.json"}
IDMAP = {"Llama-3.1-8B": "meta-llama/Llama-3.1-8B", "Gemma-2-2B": "google/gemma-2-2b",
         "Gemma-3-4B": "google/gemma-3-4b-pt", "Gemma-3-12B": "google/gemma-3-12b-pt",
         "Llama-3.2-3B": "meta-llama/Llama-3.2-3B"}
S_MD = {"Gemma-2-2B": 96.18357849121094, "Llama-3.2-3B": 4.182276}
def factor(model, cond):
    m = re.search(r"(mult|alpha)_([+-]?\d+(?:\.\d+)?)$", cond)
    v = float(m.group(2)); return v/S_MD[model] if (m.group(1) == "alpha") else v

def md_literal_rates(doc):
    agg = collections.defaultdict(lambda: [0, 0])
    for r in doc["results"]:
        for c, jl in (r.get("judge_v14") or {}).items():
            if re.match(r"^mean_diff_(mult|alpha)_[+-]?\d", c):
                for j in jl:
                    if j: agg[c][1] += 1; agg[c][0] += (j.get("label") == "literal")
    return {c: v[0]/v[1] for c, v in agg.items() if v[1]}

def jdist(doc, cond):
    c = collections.Counter()
    for r in doc["results"]:
        for j in (r.get("judge_v14") or {}).get(cond, []):
            if j: c[j.get("label", "figurative")] += 1
    t = sum(c.values()); return {l: c.get(l, 0)/t for l in LABELS}, t

gold = pd.read_csv(os.path.join(REPO, "judge_audit", "v2", "audit_gold_joined.csv"))
gold["gold"] = gold["gold"].str.lower(); gold["judge"] = gold["judge_label"].str.lower()
bkey = pd.read_csv(os.path.join(REPO, "judge_audit", "batch1", "batch1_key.csv"))
def Pgj(sub):
    cm = confusion_matrix(sub["gold"], sub["judge"], labels=LABELS).astype(float)
    col = cm.sum(0); return np.divide(cm, col, out=np.zeros_like(cm), where=col > 0), len(sub)
li = LABELS.index("literal")
def corr(Pj, P): return sum(Pj[LABELS[j]]*P[li, j] for j in range(len(LABELS)))

print("## 1-2. Headline placement + audit stratum rule\n")
print("Audit stratum rule (from original audit build): md_add_EXTREME = most-negative alpha in the")
print("model's calibrated MD grid; md_add_MODERATE = the SECOND-most-negative (the one just before extreme).")
print("Assigned by GRID RANK (the two most-negative conditions), not a |factor| threshold.\n")
p4_current = {"Llama-3.1-8B": "md_add_extreme", "Gemma-2-2B": "md_add_extreme",
              "Gemma-3-4B": "md_add_extreme", "Gemma-3-12B": "md_add_extreme", "Llama-3.2-3B": "pooled"}
rows = []; pool = gold[gold["stratum"].isin(["md_add_moderate", "md_add_extreme"])]; P_pool, _ = Pgj(pool)
worked = None
for M in FULLS:
    doc = json.load(open(os.path.join(JV14, FULLS[M])))
    rates = md_literal_rates(doc)
    hc = max(rates, key=rates.get); raw = re.search(r"_([+-]?\d+(?:\.\d+)?)$", hc).group(1); hf = factor(M, hc)
    negs = sorted([(factor(M, c), c) for c in rates if factor(M, c) < 0])  # most-neg first
    grid = [round(factor(M, c), 2) for c in sorted(rates, key=lambda c: factor(M, c))]
    rank = [c for _, c in negs].index(hc) + 1  # 1 = most negative
    # audit stratum alphas from batch1_key
    ak = {}
    for st in ["md_add_moderate", "md_add_extreme"]:
        s = bkey[(bkey.model == IDMAP[M]) & (bkey.stratum == st)]
        ak[st] = round(float(s["alpha_factor"].astype(float).iloc[0]), 3) if len(s) else None
    # nearest audited stratum by |factor|
    nearest = min(["md_add_moderate", "md_add_extreme"], key=lambda st: abs(abs(hf) - abs(ak[st])) if ak[st] is not None else 9)
    print(f"{M}: headline={hc} (raw α={raw}, α_factor={hf:+.2f}) | neg-grid rank {rank} of {len(negs)} (1=most-neg)")
    print(f"    full α_factor grid: {grid}")
    print(f"    audit moderate α_factor={ak['md_add_moderate']}  extreme={ak['md_add_extreme']}  -> headline nearest = {nearest.upper()}")
    print(f"    p4 used: {p4_current[M]}\n")
    # confusion matrices
    gm = gold[gold["model"] == IDMAP[M]]
    Pm = {st: Pgj(gm[gm.stratum == st])[0] for st in ["md_add_moderate", "md_add_extreme"]}
    Pj, tot = jdist(doc, hc)
    if M == "Llama-3.2-3B":
        correct_st = "pooled"; P_correct = P_pool
    else:
        correct_st = nearest; P_correct = Pm[nearest]
    c_correct = corr(Pj, P_correct); c_mod = corr(Pj, Pm["md_add_moderate"]); c_ext = corr(Pj, Pm["md_add_extreme"]); c_pool = corr(Pj, P_pool)
    rows.append({"model": M, "headline": hc, "alpha_factor": round(hf, 3), "assigned_stratum": correct_st,
                 "raw_literal": round(Pj["literal"], 4), "corrected_literal": round(c_correct, 4),
                 "corr_moderate": round(c_mod, 4), "corr_extreme": round(c_ext, 4), "corr_pooled": round(c_pool, 4),
                 "p4_stratum": p4_current[M], "changed": (correct_st != p4_current[M])})
    if M == "Llama-3.1-8B": worked = (M, hc, Pj, tot)

print("## 3. Revised corrected-headline table (correct stratum) — CHANGED rows flagged\n")
print(f"{'model':13} {'headline':22} {'af':>6} {'assigned':16} {'raw':>6} {'corrected':>9} {'p4 was':16} {'Δ vs p4-extreme':>16} CHG")
df = pd.DataFrame(rows)
p4vals = {"Llama-3.1-8B": 0.512, "Gemma-2-2B": 0.626, "Gemma-3-4B": 0.394, "Gemma-3-12B": 0.381, "Llama-3.2-3B": 0.535}
for r in rows:
    prev = p4vals[r["model"]]; d = r["corrected_literal"] - prev
    print(f"{r['model']:13} {r['headline']:22} {r['alpha_factor']:+6.2f} {r['assigned_stratum']:16} {r['raw_literal']:6.3f} {r['corrected_literal']:9.3f} {r['p4_stratum']:16} {d:+16.3f} {'*** ' if r['changed'] else ''}")

print("\n## 4. Sensitivity: corrected under BOTH stratum matrices\n")
print(f"{'model':13} {'corr_moderate':>13} {'corr_extreme':>13} {'corr_pooled':>12}")
for r in rows:
    print(f"{r['model']:13} {r['corr_moderate']:13.3f} {r['corr_extreme']:13.3f} {r['corr_pooled']:12.3f}")

print("\n## 5. Sanity — correction applied to the SAME condition's judged distribution (not pooled across conditions)")
M, hc, Pj, tot = worked
print(f"  worked example {M}, condition {hc} (N={tot} generations): P(judge)=" +
      ", ".join(f"{l}={Pj[l]:.3f}" for l in LABELS))
print("  corrected P(gold=literal) = Σ_j P(judge=j)·P(gold=lit|judge=j) using that condition's own distribution.")
df.to_csv(os.path.join(REPO, "camera_ready", "p4_corrected_headline_REVISED.csv"), index=False)
print("\nwrote camera_ready/p4_corrected_headline_REVISED.csv")
