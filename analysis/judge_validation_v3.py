"""Camera-ready Appendix-J analysis (v3). WRITE NOW, RUN AFTER TAGGING.
Provenance tiers: CALIBRATION (judge-selection dev set; reported separately, never pooled)
vs VALIDATION (v2 600 + batches C/D). Degrades gracefully until v3 gold exists.

Inputs:
  judge_audit/v2/audit_gold_joined.csv         (blind_id, gold, judge_label, model, stratum)  [always]
  judge_audit/v3/v3_key.csv                     (blind_id, ..., judge_label, condition, stratum, batch)
  judge_audit/v3/v3_resolved_GT.csv             (blind_id, human_label)  [OPTIONAL — from annotator adjudication]
  data_outputs/.../judge_eval/eval_*v14*rescored*.json  (calibration ground_truth vs predicted)
Output: judge_audit/v3/audit_report_v3.md
"""
import os, math, glob, json, collections
import pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

REPO = os.path.expanduser(".")
AUD = os.path.join(REPO, "judge_audit"); V2 = os.path.join(AUD, "v2"); V3 = os.path.join(AUD, "v3")
T3 = os.path.join(REPO, "concreteness", "tier3")
JV14 = os.path.join(REPO, "data_outputs", "idiom_evaluations", "judge_v14")
LABELS = ["figurative", "literal", "incoherent"]
FLAGSHIP = ["out of the blue", "behind bars", "spill the beans", "in the pink"]
L = []; w = L.append

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = (z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return (max(0, c-h), min(1, c+h))

def metrics(gold, judge):
    gold, judge = list(gold), list(judge)
    n = len(gold); k = sum(g == j for g, j in zip(gold, judge))
    a = k/n if n else float("nan"); lo, hi = wilson(k, n)
    kap = cohen_kappa_score(gold, judge, labels=LABELS) if n and len(set(gold)) > 1 else float("nan")
    return a, lo, hi, kap, n

def row(name, gold, judge):
    a, lo, hi, kap, n = metrics(gold, judge)
    return f"| {name} | {a:.3f} | [{lo:.3f}, {hi:.3f}] | {kap:.3f} | {n} |"

def conf_block(sub, title):
    if len(sub) == 0:
        w(f"**{title}** — no data yet\n"); return
    cm = confusion_matrix(sub["gold"], sub["judge"], labels=LABELS)
    w(f"**{title}** (rows=gold, cols=judge; N={len(sub)})\n")
    w("| gold\\judge | " + " | ".join(LABELS) + " |\n|" + "---|"*(len(LABELS)+1))
    for i, lab in enumerate(LABELS):
        w(f"| {lab} | " + " | ".join(str(cm[i][j]) for j in range(len(LABELS))) + " |")
    w("")

# ── VALIDATION set assembly ──────────────────────────────────────────────────
v2 = pd.read_csv(os.path.join(V2, "audit_gold_joined.csv"))
v2 = v2.rename(columns={"judge_label": "judge"}); v2["gold"] = v2["gold"].str.lower(); v2["judge"] = v2["judge"].str.lower()
v2["batch"] = "v2"; v2["condition"] = ""
parts = [v2[["gold", "judge", "model", "stratum", "batch", "condition", "blind_id"]]]

v3gold_path = os.path.join(V3, "v3_resolved_GT.csv")
HAVE_V3 = os.path.exists(v3gold_path)
if HAVE_V3:
    key = pd.read_csv(os.path.join(V3, "v3_key.csv"))
    g = pd.read_csv(v3gold_path)  # blind_id, human_label
    m = key.merge(g, on="blind_id", how="inner")
    m["gold"] = m["human_label"].str.lower(); m["judge"] = m["judge_label"].str.lower()
    m["idiom"] = m["idiom"]
    parts.append(m[["gold", "judge", "model", "stratum", "batch", "condition", "blind_id"]])
    V3ITEMS = m
else:
    V3ITEMS = None
val = pd.concat(parts, ignore_index=True)

w("# Camera-ready judge validation — Appendix J tables (v3)\n")
w("_**Provenance tiers.** CALIBRATION items were used for judge selection and are reported separately — "
  "**never pooled into validation**. VALIDATION = v2 audit (600, four held-out models) + batches C "
  "(concreteness, Llama-3.2-3B) + D (probe/angular, Llama-3.2-3B). Gold produced by joint adjudication "
  "(Appendix J); no independent inter-annotator κ reported._\n")
if not HAVE_V3:
    w("> ⚠️ **Batches C/D not yet annotated** (`judge_audit/v3/v3_resolved_GT.csv` absent). Rows/tables that "
      "depend on C/D gold are marked PENDING; rerun after adjudication.\n")

# ── T1 overall validation ────────────────────────────────────────────────────
w("## T1 — Overall validation\n| set | accuracy | Wilson 95% CI | κ | N |\n|---|---|---|---|---|")
w(row("VALIDATION (all)", val["gold"], val["judge"]))
w(row("  v2 only (4 held-out)", v2["gold"], v2["judge"]))
if HAVE_V3: w(row("  batches C+D (Llama-3.2-3B)", V3ITEMS["gold"], V3ITEMS["judge"]))
w("")

# ── T2 per model ─────────────────────────────────────────────────────────────
w("## T2 — Per model (validation)\n| model | accuracy | Wilson 95% CI | κ | N | note |\n|---|---|---|---|---|---|")
MODELS5 = ["meta-llama/Llama-3.1-8B", "google/gemma-2-2b", "google/gemma-3-4b-pt",
           "google/gemma-3-12b-pt", "meta-llama/Llama-3.2-3B"]
for mdl in MODELS5:
    s = val[val["model"] == mdl]
    note = ""
    if mdl == "meta-llama/Llama-3.2-3B":
        note = ("batches C+D: " + ", ".join(sorted(s["stratum"].unique()))) if len(s) else "PENDING (C/D not annotated)"
    if len(s) == 0:
        w(f"| {mdl} | — | — | — | 0 | {note} |")
    else:
        a, lo, hi, kap, n = metrics(s["gold"], s["judge"])
        w(f"| {mdl} | {a:.3f} | [{lo:.3f}, {hi:.3f}] | {kap:.3f} | {n} | {note} |")
w("")

# ── T3 per stratum ───────────────────────────────────────────────────────────
w("## T3 — Per stratum (validation; new C/D strata included)\n| stratum | accuracy | Wilson 95% CI | κ | N |\n|---|---|---|---|---|")
STRATA_ORDER = ["baseline", "md_add_moderate", "md_add_extreme", "md_angular_extreme",
                "probe", "random", "v_BRY", "v_PERP"]
for st in STRATA_ORDER:
    s = val[val["stratum"] == st]
    if len(s): w(row(st, s["gold"], s["judge"]))
    else: w(f"| {st} | PENDING | — | — | 0 |")
w("")

# ── T4 confusion matrices ────────────────────────────────────────────────────
w("## T4 — Confusion matrices\n")
conf_block(val, "Overall validation")
conf_block(val[val["stratum"] == "md_angular_extreme"], "md_angular_extreme (all models)")
conf_block(val[val["batch"] == "C"], "Batch C (concreteness: v_BRY + v_PERP)")

# ── Corrected literal rates ──────────────────────────────────────────────────
w("## Corrected literal rates\n")
def P_gold_given_judge(sub):
    cm = confusion_matrix(sub["gold"], sub["judge"], labels=LABELS).astype(float)
    col = cm.sum(0); P = np.divide(cm, col, out=np.zeros_like(cm), where=col > 0)
    return P  # P[gold, judge]
def pooled_judge_dist(path, conds):
    d = json.load(open(path)); c = collections.Counter()
    for r in d["results"]:
        for cond in conds:
            for j in (r.get("judge_v14") or {}).get(cond, []):
                if j: c[j.get("label", "figurative")] += 1
    tot = sum(c.values()); return {l: c.get(l, 0)/tot for l in LABELS}, tot
if HAVE_V3:
    li = LABELS.index("literal")
    w("Applying batch-C P(gold|judge) to the pooled judged distributions (feeds §6.4 / concreteness appendix):\n")
    w("| condition group | audit confusion | raw literal | corrected literal | N_audit |\n|---|---|---|---|---|")
    for fam, path, conds in [("v_BRY (+10/+15/+20)", os.path.join(T3, "judged_v_BRY.json"),
                              ["v_BRY_raw_+10", "v_BRY_raw_+15", "v_BRY_raw_+20"]),
                             ("v_PERP (-10/-15)", os.path.join(T3, "judged_v_PERP.json"),
                              ["v_PERP_raw_-10", "v_PERP_raw_-15"])]:
        stratum = "v_BRY" if fam.startswith("v_BRY") else "v_PERP"
        sub = val[val["stratum"] == stratum]
        P = P_gold_given_judge(sub)
        Pj, _ = pooled_judge_dist(path, conds)
        corr = sum(Pj[LABELS[j]] * P[li, j] for j in range(len(LABELS)))
        w(f"| {fam} | batch-C {stratum} (N={len(sub)}) | {Pj['literal']:.3f} | {corr:.3f} | {len(sub)} |")
    w("\n(Headline MD-additive per-model corrected rates are unchanged from v2 §4 — recompute here once "
      "all five models' MD-additive strata carry validation gold.)")
else:
    w("_PENDING batch-C gold — corrected v_BRY/v_PERP literal rates recompute after adjudication._")
w("")

# ── Flagship-idiom agreement split (batch C) ─────────────────────────────────
w("## Flagship-idiom agreement (within Batch C)\n")
if HAVE_V3:
    bc = V3ITEMS[V3ITEMS["batch"] == "C"]
    w("| idiom | accuracy | N |\n|---|---|---|")
    for idi in FLAGSHIP:
        s = bc[bc["idiom"] == idi]
        if len(s):
            a = (s["gold"] == s["judge"]).mean(); w(f"| {idi} | {a:.3f} | {len(s)} |")
        else: w(f"| {idi} | — | 0 |")
    rest = bc[~bc["idiom"].isin(FLAGSHIP)]
    if len(rest): w(f"| (non-flagship) | {(rest['gold']==rest['judge']).mean():.3f} | {len(rest)} |")
else:
    w("_PENDING batch-C gold._")
w("")

# ── CALIBRATION tier (separate; never pooled) ────────────────────────────────
w("## CALIBRATION tier (judge-selection dev set — reported separately, NOT pooled into validation)\n")
evals = sorted(glob.glob(os.path.join(REPO, "data_outputs/idiom_evaluations/judge_eval/eval_*v14*rescored*.json")))
if not evals:
    evals = sorted(glob.glob(os.path.join(REPO, "data_outputs/idiom_evaluations/judge_eval/eval_*v14*.json")))
if evals:
    d = json.load(open(evals[-1])); recs = d.get("predictions", d)
    cg = [str(r.get("ground_truth")).lower() for r in recs]
    cj = [str(r.get("predicted")).lower() for r in recs]
    a, lo, hi, kap, n = metrics(cg, cj)
    w(f"- source: `{os.path.basename(evals[-1])}`")
    w(f"- **accuracy {a:.3f} [{lo:.3f}, {hi:.3f}], κ {kap:.3f}, N={n}**")
    w(f"- This is the full calibration set (N={n}), used to SELECT the judge; reported here separately and "
      "**never pooled into the validation tables above**. Strata: baseline + MD-additive ±10/±20 + random ±20 "
      "(no probe/angular — those are covered for Llama-3.2-3B by validation batch D).")
else:
    w("- calibration eval file not found.")
w("")

open(os.path.join(V3, "audit_report_v3.md"), "w").write("\n".join(L))
print("wrote", os.path.join(V3, "audit_report_v3.md"), "| HAVE_V3_GOLD =", HAVE_V3)
print("\n".join(L))
