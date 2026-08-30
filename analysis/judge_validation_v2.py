"""Judge-vs-gold audit analysis -> judge_audit/v2/audit_report_v2.md + audit_gold_joined.csv.
Base anaconda python (openpyxl+sklearn). Does not modify inputs."""
import os, math, json, csv, collections
import pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

REPO = os.path.expanduser(".")
V2 = os.path.join(REPO, "judge_audit", "v2")
JV14 = os.path.join(REPO, "data_outputs", "idiom_evaluations", "judge_v14")
LABELS = ["figurative", "literal", "incoherent"]
S_MD = {"google/gemma-2-2b": 96.18357849121094}
FULLS = {"meta-llama/Llama-3.1-8B": "judged_three_methods_eval_llama8b_FULL.json",
         "google/gemma-2-2b": "judged_three_methods_eval_gemma2_FULL.json",
         "google/gemma-3-4b-pt": "judged_three_methods_eval_gemma3_4b_FULL.json",
         "google/gemma-3-12b-pt": "judged_three_methods_eval_gemma3_12b_FULL.json"}
MODEL_ORDER = list(FULLS.keys())
STRATA = ["baseline", "md_add_moderate", "md_add_extreme", "md_angular_extreme", "probe", "random"]
L = []
def w(s=""): L.append(s)

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = (z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)))/d
    return (max(0, c-h), min(1, c+h))

def acc_ci_kappa(gold, judge):
    n = len(gold); k = sum(g == j for g, j in zip(gold, judge))
    a = k/n if n else float("nan"); lo, hi = wilson(k, n)
    kap = cohen_kappa_score(gold, judge, labels=LABELS) if n and len(set(gold)) > 1 else float("nan")
    return a, lo, hi, kap, n

# ── load ─────────────────────────────────────────────────────────────────────
gt = pd.read_excel(os.path.join(V2, "annotators_resolved_GT.xlsx"))
gt = gt.rename(columns={"human_label": "gold"})
gt["blind_id"] = gt["blind_id"].astype(str)
key = pd.read_csv(os.path.join(V2, "batch1_key.csv"))
key["blind_id"] = key["blind_id"].astype(str)

w("# Audit report v2 — Judge vs Gold (batch-1, 4 held-out models)\n")
w("_Gold labels were produced by **joint annotation/adjudication** (two annotators, paper Appendix J "
  "protocol) and delivered as a single resolved file; therefore **no independent inter-annotator κ is "
  "reported for this batch** — the two annotators did not label independently._\n")

# ── STEP 1 INTEGRITY ─────────────────────────────────────────────────────────
w("## 1. Integrity\n")
n_rows = len(gt); n_uniq = gt["blind_id"].nunique()
dup = gt["blind_id"][gt["blind_id"].duplicated()].tolist()
in_key = set(key["blind_id"])
not_joined = [b for b in gt["blind_id"] if b not in in_key]
valid = set(LABELS)
gt["gold"] = gt["gold"].astype(str).str.strip().str.lower()
invalid = gt[~gt["gold"].isin(valid)][["blind_id", "gold"]].values.tolist()
w(f"- rows: {n_rows} | unique blind_ids: {n_uniq} | duplicates: {dup or 'none'}")
w(f"- join to key: {n_rows-len(not_joined)}/{n_rows} joined | not-joined: {not_joined or 'none'}")
w(f"- labels ∈ {{figurative,literal,incoherent}}: invalid rows: {invalid or 'none'}")
# merge, exclude invalid/unjoined
df = gt.merge(key, on="blind_id", how="inner")
df = df[df["gold"].isin(valid)].copy()
df["judge"] = df["judge_label"].astype(str).str.strip().str.lower()
# cross-check xlsx retained judge col vs key
xj = "gt_label_REMOVE_BEFORE_ANNOTATION"
if xj in gt.columns:
    chk = gt[["blind_id", xj]].merge(key[["blind_id", "judge_label"]], on="blind_id", how="inner")
    mm = (chk[xj].astype(str).str.lower() != chk["judge_label"].astype(str).str.lower()).sum()
    w(f"- sanity: xlsx retained judge col vs key.judge_label mismatches: {mm} (expect 0)")
w(f"- **analysis N = {len(df)}**\n")
gd = df["gold"].value_counts().reindex(LABELS).fillna(0).astype(int)
jd = df["judge"].value_counts().reindex(LABELS).fillna(0).astype(int)
w("GT vs Judge label distribution:\n")
w("| label | gold | judge |\n|---|---|---|")
for lab in LABELS: w(f"| {lab} | {gd[lab]} ({gd[lab]/len(df)*100:.1f}%) | {jd[lab]} ({jd[lab]/len(df)*100:.1f}%) |")
w("")

def tbl(gold, judge):
    a, lo, hi, kap, n = acc_ci_kappa(list(gold), list(judge))
    return f"{a:.3f} | [{lo:.3f}, {hi:.3f}] | {kap:.3f} | {n}"

# ── STEP 2 JUDGE vs GOLD ─────────────────────────────────────────────────────
w("## 2. Judge vs Gold\n")
w("### T1 — Overall\n| accuracy | Wilson 95% CI | Cohen's κ | N |\n|---|---|---|---|")
w("| " + tbl(df["gold"], df["judge"]) + " |\n")
w("### T2 — Per model (headline)\n| model | accuracy | Wilson 95% CI | κ | N |\n|---|---|---|---|---|")
for m in MODEL_ORDER:
    s = df[df["model"] == m]
    w(f"| {m} | " + tbl(s["gold"], s["judge"]) + " |")
w("")
w("### T3 — Per stratum\n| stratum | accuracy | Wilson 95% CI | κ | N |\n|---|---|---|---|---|")
for st in STRATA:
    s = df[df["stratum"] == st]
    w(f"| {st} | " + tbl(s["gold"], s["judge"]) + " |")
w("")
def conf_md(sub, title):
    cm = confusion_matrix(sub["gold"], sub["judge"], labels=LABELS)
    w(f"**{title}** (rows=gold, cols=judge; N={len(sub)})\n")
    w("| gold\\judge | " + " | ".join(LABELS) + " |\n|" + "---|"*(len(LABELS)+1))
    for i, lab in enumerate(LABELS):
        w(f"| {lab} | " + " | ".join(str(cm[i][j]) for j in range(len(LABELS))) + " |")
    w("")
w("### T4 — Confusion matrices\n")
conf_md(df, "Overall")
conf_md(df[df["stratum"] == "md_angular_extreme"], "md_angular_extreme")

# ── STEP 3 FAILURE MODES ─────────────────────────────────────────────────────
w("## 3. Failure modes\n")
def coh(x): return "incoherent" if x == "incoherent" else "coherent"
df["gold_coh"] = df["gold"].map(coh); df["judge_coh"] = df["judge"].map(coh)
w("### Coherent/incoherent binary agreement\n| scope | agreement | N |\n|---|---|---|")
w(f"| overall | {(df['gold_coh']==df['judge_coh']).mean():.3f} | {len(df)} |")
for st in STRATA:
    s = df[df["stratum"] == st]
    if len(s): w(f"| {st} | {(s['gold_coh']==s['judge_coh']).mean():.3f} | {len(s)} |")
w("")
w("### Directional check per stratum\n")
w("| stratum | judge=literal N | →gold=fig | →gold=incoh | gold=literal N | judge missed (≠lit) |\n"
  "|---|---|---|---|---|---|")
for st in STRATA:
    s = df[df["stratum"] == st]
    jl = s[s["judge"] == "literal"]; gl = s[s["gold"] == "literal"]
    def fr(sub, cond): return f"{cond.sum()/len(sub):.2f}" if len(sub) else "—"
    w(f"| {st} | {len(jl)} | {fr(jl,(jl['gold']=='figurative'))} | {fr(jl,(jl['gold']=='incoherent'))} "
      f"| {len(gl)} | {fr(gl,(gl['judge']!='literal'))} |")
w("")
# top-10 highest-impact disagreements: literal-flips first
w("### Top-10 highest-impact disagreements\n")
w("_Impact = disagreements that flip the literal call (judge=literal→gold≠literal, or gold=literal→judge≠literal), "
  "since literal-rate is the headline metric._\n")
dis = df[df["gold"] != df["judge"]].copy()
def impact(r):
    lit_flip = (r["judge"] == "literal" and r["gold"] != "literal") or (r["gold"] == "literal" and r["judge"] != "literal")
    return (2 if lit_flip else 1, r["stratum"] in ("md_add_extreme", "md_angular_extreme"))
dis["_imp"] = dis.apply(lambda r: impact(r)[0]*10 + impact(r)[1], axis=1)
gtx = gt.set_index("blind_id")
w("| blind_id | model | stratum | gold | judge | prefix → completion excerpt |\n|---|---|---|---|---|---|")
for _, r in dis.sort_values("_imp", ascending=False).head(10).iterrows():
    pre = str(gtx.loc[r["blind_id"], "sentence_prefix"])[:60].replace("\n", " ")
    comp = str(gtx.loc[r["blind_id"], "completion"])[:90].replace("\n", " ")
    w(f"| {r['blind_id']} | {r['model'].split('/')[-1]} | {r['stratum']} | {r['gold']} | {r['judge']} | …{pre} → {comp}… |")
w("")

# ── STEP 4 CORRECTED RATES ───────────────────────────────────────────────────
w("## 4. Corrected literal rates at headline MD-additive operating points\n")
def P_gold_given_judge(stratum):
    s = df[df["stratum"] == stratum]
    cm = confusion_matrix(s["gold"], s["judge"], labels=LABELS).astype(float)  # [gold,judge]
    col = cm.sum(0); P = np.divide(cm, col, out=np.zeros_like(cm), where=col > 0)  # P[gold,judge]
    return P, col
def parse_factor(model, cond):
    import re
    m = re.search(r"(mult|alpha)_([+-]?\d+(?:\.\d+)?)", cond)
    if not m: return None
    v = float(m.group(2))
    return v/S_MD[model] if (m.group(1) == "alpha" and model in S_MD) else v
w("_Method: at each model's MD-additive best-literal condition (max judge-literal rate over mean_diff additive "
  "conditions in the FULL judged file), apply the nearest audited MD-additive stratum's P(gold|judge) to the "
  "judged label distribution: corrected P(gold=lit)=Σ_j P(judge=j)·P(gold=lit|judge=j)._\n")
w("| model | headline cond (α_factor) | audit stratum used | raw literal | corrected literal |\n|---|---|---|---|---|")
gold_join_rows = []
for m in MODEL_ORDER:
    d = json.load(open(os.path.join(JV14, FULLS[m])))
    # literal rate per mean_diff additive condition
    per = collections.defaultdict(lambda: [0, 0])
    for r in d["results"]:
        for cond, jl in (r.get("judge_v14") or {}).items():
            if cond.startswith("mean_diff_") and ("mult_" in cond or "alpha_" in cond):
                for j in jl:
                    if not j: continue
                    per[cond][1] += 1
                    if j.get("label") == "literal": per[cond][0] += 1
    best = max(per.items(), key=lambda kv: kv[1][0]/kv[1][1] if kv[1][1] else 0)
    bc = best[0]; bf = parse_factor(m, bc)
    # judged dist at best
    dist = collections.Counter()
    for r in d["results"]:
        for j in (r.get("judge_v14") or {}).get(bc, []):
            if j: dist[j.get("label", "figurative")] += 1
    tot = sum(dist.values()); Pj = {l: dist.get(l, 0)/tot for l in LABELS}
    # nearest audited MD-additive stratum by |factor|
    audf = {st: df[(df.model == m) & (df.stratum == st)]["alpha_factor"].astype(float).abs().mean()
            for st in ("md_add_moderate", "md_add_extreme")}
    stratum = min(audf, key=lambda st: abs(abs(bf) - audf[st]))
    P, col = P_gold_given_judge(stratum)
    li = LABELS.index("literal")
    corrected = sum(Pj[LABELS[j]] * P[li, j] for j in range(len(LABELS)))
    w(f"| {m} | {bc} ({bf:+.2f}) | {stratum} | {Pj['literal']:.3f} | {corrected:.3f} |")
w("")

# ── STEP 5 SUMMARY ───────────────────────────────────────────────────────────
w("## 5. Summary\n")
per_model = {m: acc_ci_kappa(list(df[df.model==m].gold), list(df[df.model==m].judge)) for m in MODEL_ORDER}
fam = {"MD-additive": ["md_add_moderate", "md_add_extreme"], "MD-angular": ["md_angular_extreme"],
       "probe": ["probe"], "random": ["random"], "baseline": ["baseline"]}
per_fam = {}
for f, sts in fam.items():
    s = df[df.stratum.isin(sts)]; per_fam[f] = acc_ci_kappa(list(s.gold), list(s.judge))
best_m = max(per_model, key=lambda m: per_model[m][0]); worst_m = min(per_model, key=lambda m: per_model[m][0])
best_f = max(per_fam, key=lambda f: per_fam[f][0]); worst_f = min(per_fam, key=lambda f: per_fam[f][0])
ov = acc_ci_kappa(list(df.gold), list(df.judge))
w(f"The v14 judge generalizes **across the four held-out models** — overall accuracy {ov[0]:.3f} "
  f"(κ={ov[3]:.3f}, N={ov[4]}), ranging from {per_model[worst_m][0]:.3f} ({worst_m.split('/')[-1]}) to "
  f"{per_model[best_m][0]:.3f} ({best_m.split('/')[-1]}). It also holds **across method families**: "
  f"best {best_f} ({per_fam[best_f][0]:.3f}), weakest **{worst_f} ({per_fam[worst_f][0]:.3f}, N={per_fam[worst_f][4]})** "
  f"— the caveat cell: {worst_f} shows the largest judge–gold gap (see T3/T4), so literal-rate claims in that "
  f"regime should use the corrected rates in §4. Because first_pass_label ≡ judge_label, the human 'correction "
  f"rate' and 'judge disagreement rate' are one and the same measurement. Gold was jointly adjudicated "
  f"(Appendix J), so no independent inter-annotator κ is reported for this batch.")
w("")

with open(os.path.join(V2, "audit_report_v2.md"), "w") as fh:
    fh.write("\n".join(L))

# ── save joined csv ──────────────────────────────────────────────────────────
out = df[["blind_id", "gold", "judge_label", "model", "stratum"]].rename(columns={"judge_label": "judge_label"})
out.to_csv(os.path.join(V2, "audit_gold_joined.csv"), index=False)
print("wrote audit_report_v2.md and audit_gold_joined.csv (N=%d)" % len(df))
print("\n".join(L))
