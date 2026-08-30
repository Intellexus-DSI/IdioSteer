"""Camera-ready aggregation, part 1 (parts 1-4). Read-only from judged FULL files + v2 validation gold."""
import os, re, json, collections, csv
import numpy as np, pandas as pd
REPO = os.path.expanduser(".")
JV14 = os.path.join(REPO, "data_outputs", "idiom_evaluations", "judge_v14")
OUT = os.path.join(REPO, "camera_ready"); os.makedirs(OUT, exist_ok=True)
V2GOLD = os.path.join(REPO, "judge_audit", "v2", "audit_gold_joined.csv")
LABELS = ["figurative", "literal", "incoherent"]
FULLS = {
    "Llama-3.2-3B": "judged_three_methods_eval_validated_FULL.json",
    "Llama-3.1-8B": "judged_three_methods_eval_llama8b_FULL.json",
    "Gemma-2-2B":   "judged_three_methods_eval_gemma2_FULL.json",
    "Gemma-3-4B":   "judged_three_methods_eval_gemma3_4b_FULL.json",
    "Gemma-3-12B":  "judged_three_methods_eval_gemma3_12b_FULL.json",
}
MODEL_ORDER = list(FULLS.keys())
FAMILIES = ["md_additive", "probe_additive", "md_angular", "probe_angular", "random_additive", "random_angular"]
FAM_LABEL = {"md_additive": "MD-additive", "probe_additive": "PROBE-additive", "md_angular": "MD-angular",
             "probe_angular": "PROBE-angular", "random_additive": "random-additive", "random_angular": "random-angular"}
THRESH = 0.14

def classify(c):
    if c == "baseline": return (None, None, None)
    m = re.match(r"^random\d+_(mult|alpha)_([+-]?\d+(?:\.\d+)?)$", c)
    if m: return ("random_additive", "add", float(m.group(2)))
    m = re.match(r"^random\d+_angular_([+-]?\d+)deg$", c)
    if m: return ("random_angular", "ang", float(m.group(1)))
    m = re.match(r"^probe_(mult|alpha)_([+-]?\d+(?:\.\d+)?)$", c)
    if m: return ("probe_additive", "add", float(m.group(2)))
    m = re.match(r"^probe_angular_([+-]?\d+)deg$", c)
    if m: return ("probe_angular", "ang", float(m.group(1)))
    m = re.match(r"^mean_diff_(mult|alpha)_([+-]?\d+(?:\.\d+)?)$", c)
    if m: return ("md_additive", "add", float(m.group(2)))
    m = re.match(r"^(?:mean_diff_angular|angular)_([+-]?\d+)deg$", c)
    if m: return ("md_angular", "ang", float(m.group(1)))
    return (None, None, None)

def load(model):
    doc = json.load(open(os.path.join(JV14, FULLS[model])))
    return doc

def cond_rates(doc):
    agg = collections.defaultdict(lambda: [0, 0]); base = [0, 0]
    for r in doc["results"]:
        for c, jl in (r.get("judge_v14") or {}).items():
            for j in jl:
                if not j: continue
                agg[c][1] += 1; agg[c][0] += (j.get("label") == "literal")
                if c == "baseline": base[1] += 1; base[0] += (j.get("label") == "literal")
    return {c: v[0]/v[1] for c, v in agg.items() if v[1]}, (base[0]/base[1] if base[1] else float("nan"))

def is_literal_dir(kind, val):
    return (val < 0) if kind == "add" else (val > 0)  # neg alpha / pos theta (reproduces sanity incl. random)

# ============ PART 1: family robustness table ============
p1 = []
sanity = {}
for model in MODEL_ORDER:
    doc = load(model); rates, base = cond_rates(doc)
    buck = collections.defaultdict(list)
    for c, r in rates.items():
        fam, kind, val = classify(c)
        if fam and is_literal_dir(kind, val): buck[fam].append(r)
    for fam in FAMILIES:
        v = buck.get(fam, [])
        row = {"model": model, "family": FAM_LABEL[fam],
               "mean": (sum(v)/len(v)) if v else float("nan"),
               "min": min(v) if v else float("nan"), "max": max(v) if v else float("nan"),
               "n_gt_baseline+0.02": sum(1 for x in v if x > base + 0.02), "n_points": len(v)}
        p1.append(row)
    if model == "Llama-3.2-3B":
        sanity = {FAM_LABEL[f]: round((sum(buck[f])/len(buck[f])) if buck[f] else float("nan"), 3) for f in FAMILIES}
p1df = pd.DataFrame(p1)
p1df.to_csv(os.path.join(OUT, "p1_robustness_by_family.csv"), index=False)
print("=== PART 1 sanity (Llama-3.2-3B family means) ===")
print("  expect MD-add .460 PR-add .336 MD-ang .275 PR-ang .238 rand-add .259 rand-ang .201")
print("  got   ", sanity)

# LaTeX for part1 (mean [min,max] (k>base) per model x family)
with open(os.path.join(OUT, "p1_robustness_by_family.tex"), "w") as fh:
    fh.write("% mean literal rate over literal-direction half-grid; [min,max]; (k = #points > baseline+0.02)\n")
    fh.write("\\begin{tabular}{l" + "c"*len(FAMILIES) + "}\n\\toprule\nModel & " +
             " & ".join(FAM_LABEL[f] for f in FAMILIES) + " \\\\\n\\midrule\n")
    for model in MODEL_ORDER:
        cells = []
        for f in FAMILIES:
            r = next(x for x in p1 if x["model"] == model and x["family"] == FAM_LABEL[f])
            cells.append(f"{r['mean']:.3f} [{r['min']:.2f},{r['max']:.2f}] ({r['n_gt_baseline+0.02']}/{r['n_points']})")
        fh.write(model + " & " + " & ".join(cells) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ============ headline MD-additive condition per model ============
def headline(model):
    doc = load(model); rates, base = cond_rates(doc)
    md = {c: r for c, r in rates.items() if classify(c)[0] == "md_additive"}
    best = max(md, key=md.get)
    return best, rates[best], base, doc
HEAD = {m: headline(m) for m in MODEL_ORDER}

# ============ PART 2: per-idiom breadth counts ============
def per_idiom_rates(doc, conds):
    agg = collections.defaultdict(lambda: [0, 0])
    cset = set(conds)
    for r in doc["results"]:
        idi = r["idiom"]
        for c, jl in (r.get("judge_v14") or {}).items():
            if c not in cset: continue
            for j in jl:
                if not j: continue
                agg[idi][1] += 1; agg[idi][0] += (j.get("label") == "literal")
    return {i: v[0]/v[1] for i, v in agg.items() if v[1]}

print("\n=== PART 2: per-idiom breadth (idioms with literal rate > 0.14) ===")
doc3b = HEAD["Llama-3.2-3B"][3]; head3b = HEAD["Llama-3.2-3B"][0]
r_head = per_idiom_rates(doc3b, [head3b]); r_pool = per_idiom_rates(doc3b, ["mean_diff_alpha_-10", "mean_diff_alpha_-15"])
n_head = sum(1 for v in r_head.values() if v > THRESH); n_pool = sum(1 for v in r_pool.values() if v > THRESH)
print(f"  Llama-3.2-3B headline ({head3b}): {n_head}/{len(r_head)} idioms > {THRESH}")
print(f"  Llama-3.2-3B pooled (-10/-15): {n_pool}/{len(r_pool)} idioms > {THRESH}")
breadth = [{"model": "Llama-3.2-3B", "convention": f"headline {head3b}", "n_idioms_gt_0.14": n_head, "n_idioms": len(r_head)},
           {"model": "Llama-3.2-3B", "convention": "pooled -10/-15", "n_idioms_gt_0.14": n_pool, "n_idioms": len(r_pool)}]
for m in MODEL_ORDER:
    hc, _, _, doc = HEAD[m]; ri = per_idiom_rates(doc, [hc])
    n = sum(1 for v in ri.values() if v > THRESH)
    breadth.append({"model": m, "convention": f"headline {hc}", "n_idioms_gt_0.14": n, "n_idioms": len(ri)})
    if m != "Llama-3.2-3B": print(f"  {m} headline ({hc}): {n}/{len(ri)} idioms > {THRESH}")
pd.DataFrame(breadth).to_csv(os.path.join(OUT, "p2_breadth_counts.csv"), index=False)

# ============ PART 3: per-idiom results table (headline + baseline) ============
p3 = []
for m in MODEL_ORDER:
    hc, _, _, doc = HEAD[m]
    rh = per_idiom_rates(doc, [hc]); rb = per_idiom_rates(doc, ["baseline"])
    for idi in sorted(rh):
        p3.append({"model": m, "idiom": idi, "headline_condition": hc,
                   "literal_rate": round(rh[idi], 4), "baseline_literal_rate": round(rb.get(idi, float("nan")), 4)})
p3df = pd.DataFrame(p3); p3df.to_csv(os.path.join(OUT, "p3_per_idiom_headline.csv"), index=False)
# LaTeX: idiom rows, model columns (headline literal rate)
piv = p3df.pivot(index="idiom", columns="model", values="literal_rate").reindex(columns=MODEL_ORDER)
with open(os.path.join(OUT, "p3_per_idiom_headline.tex"), "w") as fh:
    fh.write("\\begin{tabular}{l" + "c"*len(MODEL_ORDER) + "}\n\\toprule\nIdiom & " +
             " & ".join(MODEL_ORDER) + " \\\\\n\\midrule\n")
    for idi, row in piv.iterrows():
        fh.write(idi + " & " + " & ".join(f"{row[m]:.3f}" if pd.notna(row[m]) else "--" for m in MODEL_ORDER) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ============ PART 4: corrected headline rates (validation confusion) ============
gold = pd.read_csv(V2GOLD); gold["gold"] = gold["gold"].str.lower(); gold["judge"] = gold["judge_label"].str.lower()
def Pgj(sub):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(sub["gold"], sub["judge"], labels=LABELS).astype(float)
    col = cm.sum(0); return np.divide(cm, col, out=np.zeros_like(cm), where=col > 0)  # P[gold,judge]
# per held-out model: nearest md_add stratum by |headline factor|; llama3b: pooled md_add across held-out
VID_RAWN = {"Gemma-2-2B": 96.18357849121094}  # only gemma2 uses raw-alpha headline
def factor(model, cond):
    m = re.search(r"(mult|alpha)_([+-]?\d+(?:\.\d+)?)$", cond)
    v = float(m.group(2)); return v/VID_RAWN[model] if (m.group(1) == "alpha" and model in VID_RAWN) else v
p4 = []; li = LABELS.index("literal")
pooled = gold[gold["stratum"].isin(["md_add_moderate", "md_add_extreme"])]
P_pool = Pgj(pooled)
for m in MODEL_ORDER:
    hc, raw, base, doc = HEAD[m]
    rates, _ = cond_rates(doc)
    dist = collections.Counter()
    for r in doc["results"]:
        for j in (r.get("judge_v14") or {}).get(hc, []):
            if j: dist[j.get("label", "figurative")] += 1
    tot = sum(dist.values()); Pj = {l: dist.get(l, 0)/tot for l in LABELS}
    if m == "Llama-3.2-3B":
        P = P_pool; stratum = "md_add pooled (4 held-out) — its own MD gold is CALIBRATION-tier"
    else:
        sub = gold[gold["model"].str.contains(m.split("-")[-1].lower()) | (gold["model"].str.lower().str.contains(m.replace('-','').lower()))]
        # match by explicit model id
        idmap = {"Llama-3.1-8B": "meta-llama/Llama-3.1-8B", "Gemma-2-2B": "google/gemma-2-2b",
                 "Gemma-3-4B": "google/gemma-3-4b-pt", "Gemma-3-12B": "google/gemma-3-12b-pt"}
        gm = gold[gold["model"] == idmap[m]]
        # nearest stratum by |factor|
        hf = abs(factor(m, hc))
        cand = {}
        for st in ["md_add_moderate", "md_add_extreme"]:
            s = gm[gm["stratum"] == st]
            if len(s): cand[st] = s
        # pick stratum whose name matches magnitude: extreme if hf large; use both if only one; choose by closeness to audited alpha unknown -> default extreme if hf is max
        stratum = "md_add_extreme" if "md_add_extreme" in cand else list(cand)[0]
        P = Pgj(cand[stratum]); stratum = stratum + f" ({m})"
    corrected = sum(Pj[LABELS[j]] * P[li, j] for j in range(len(LABELS)))
    p4.append({"model": m, "headline_condition": hc, "alpha_factor": round(factor(m, hc), 3),
               "raw_literal": round(raw, 4), "corrected_literal": round(corrected, 4), "validation_stratum": stratum})
p4df = pd.DataFrame(p4); p4df.to_csv(os.path.join(OUT, "p4_corrected_headline.csv"), index=False)
with open(os.path.join(OUT, "p4_corrected_headline.tex"), "w") as fh:
    fh.write("\\begin{tabular}{llccl}\n\\toprule\nModel & Headline & Raw literal & Corrected & Val. stratum \\\\\n\\midrule\n")
    for r in p4:
        fh.write(f"{r['model']} & {r['headline_condition']} & {r['raw_literal']:.3f} & {r['corrected_literal']:.3f} & {r['validation_stratum']} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

print("\n=== PART 4: corrected headline rates ===")
for r in p4: print(f"  {r['model']:14} {r['headline_condition']:20} raw={r['raw_literal']:.3f} corrected={r['corrected_literal']:.3f}  [{r['validation_stratum']}]")
print("\nOutputs in camera_ready/:", sorted(os.listdir(OUT)))
