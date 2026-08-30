"""Camera-ready part 2 (concreteness): p5 geometry, p6 gates, p7 three-bar figure, p8 headline numbers.
Run after build_conc_all batch. Base python (matplotlib). Read-only on sources."""
import os, re, json, math, collections
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = os.path.expanduser(".")
CONC = os.path.join(REPO, "concreteness"); T3 = os.path.join(CONC, "tier3")
JV14 = os.path.join(REPO, "data_outputs", "idiom_evaluations", "judge_v14")
OUT = os.path.join(REPO, "camera_ready"); os.makedirs(OUT, exist_ok=True)
FLAG = ["out of the blue", "behind bars", "spill the beans", "in the pink"]
MODELS = ["Llama-3.2-3B", "Llama-3.1-8B", "Gemma-2-2B", "Gemma-3-4B", "Gemma-3-12B"]
KEY = {"Llama-3.2-3B": "llama3b", "Llama-3.1-8B": "llama8b", "Gemma-2-2B": "gemma2_2b",
       "Gemma-3-4B": "gemma3_4b", "Gemma-3-12B": "gemma3_12b"}
LABELS = ["figurative", "literal", "incoherent"]
flags_report = []
def flag(msg): flags_report.append(msg); print("  [FLAG]", msg)
def approx(got, exp, tol=0.006):
    return got is not None and exp is not None and abs(got - exp) <= tol

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = (z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d
    return (max(0, c-h), min(1, c+h))

# ---------- P5 geometry ----------
conc = {m: json.load(open(os.path.join(CONC, "results", f"{KEY[m]}_conc.json"))) for m in MODELS}
p5 = [{"model": m, "cos(vIDIOM,vBRY)": round(conc[m]["geometry"]["cos_idiom_bry"], 4),
       "cos(vIDIOM,vGJ_bw)": round(conc[m]["geometry"]["cos_idiom_gjbw"], 4),
       "cos(vGJ_bw,vBRY)": round(conc[m]["geometry"]["cos_gjbw_bry"], 4)} for m in MODELS]
pd.DataFrame(p5).to_csv(os.path.join(OUT, "p5_geometry.csv"), index=False)
with open(os.path.join(OUT, "p5_geometry.tex"), "w") as fh:
    fh.write("% poles: v_IDIOM +figurative ; v_BRY +concrete ; v_GJ_bareword +concrete\n")
    fh.write("\\begin{tabular}{lccc}\n\\toprule\nModel & cos($v_{IDIOM}$,$v_{BRY}$) & cos($v_{IDIOM}$,$v_{GJ}^{bw}$) & cos($v_{GJ}^{bw}$,$v_{BRY}$) \\\\\n\\midrule\n")
    for r in p5: fh.write(f"{r['model']} & {r['cos(vIDIOM,vBRY)']:+.3f} & {r['cos(vIDIOM,vGJ_bw)']:+.3f} & {r['cos(vGJ_bw,vBRY)']:+.3f} \\\\\n")
    fh.write("\\midrule\n\\multicolumn{4}{l}{\\emph{ref: Llama-3B cross-fold idiom-vector self-agreement 0.71--0.80}} \\\\\n\\bottomrule\n\\end{tabular}\n")
c3b = conc["Llama-3.2-3B"]["geometry"]["cos_idiom_bry"]
print("=== P5 geometry ===")
print("  sanity llama3b cos(vIDIOM,vBRY) = %.3f (expect -0.39)" % c3b)
if not approx(c3b, -0.39, 0.01): flag(f"llama3b cos(vIDIOM,vBRY)={c3b:.3f} != -0.39")

# ---------- P6 gates ----------
p6 = []
for m in MODELS:
    gb = conc[m]["gates_v_BRY_wordsplit"]; gg = conc[m]["gates_v_GJ_bareword_wordsplit"]
    p6.append({"model": m, "vBRY_half_cos": round(gb["half_split_cos"], 4), "vBRY_heldout_acc": round(gb["heldout_acc"], 4),
               "vBRY_4a_pass": gb["gate_4a_pass"], "vBRY_4b_pass": gb["gate_4b_pass"],
               "vGJbw_half_cos": round(gg["half_split_cos"], 4), "vGJbw_heldout_acc": round(gg["heldout_acc"], 4),
               "vGJbw_4a_pass": gg["gate_4a_pass"], "vGJbw_4b_pass": gg["gate_4b_pass"]})
pd.DataFrame(p6).to_csv(os.path.join(OUT, "p6_gates.csv"), index=False)
print("=== P6 gates (word-split) written ===")

# ---------- helpers for per-idiom pooled rates ----------
val = json.load(open(os.path.join(JV14, "judged_three_methods_eval_validated_FULL.json")))
bry = json.load(open(os.path.join(T3, "judged_v_BRY.json")))
perp = json.load(open(os.path.join(T3, "judged_v_PERP.json")))
def per_idiom(doc, conds, coherent_only):
    cs = set(conds); agg = collections.defaultdict(lambda: [0, 0])
    for r in doc["results"]:
        for c, jl in (r.get("judge_v14") or {}).items():
            if c not in cs: continue
            for j in jl:
                if not j: continue
                lab = j.get("label")
                if coherent_only and lab == "incoherent": continue
                agg[r["idiom"]][1] += 1; agg[r["idiom"]][0] += (lab == "literal")
    return {i: (v[0], v[1]) for i, v in agg.items() if v[1]}   # idiom -> (lit, n)
def overall(doc, conds, coherent_only):
    cs = set(conds); lit = n = 0
    for r in doc["results"]:
        for c, jl in (r.get("judge_v14") or {}).items():
            if c not in cs: continue
            for j in jl:
                if not j: continue
                lab = j.get("label")
                if coherent_only and lab == "incoherent": continue
                n += 1; lit += (lab == "literal")
    return lit/n if n else float("nan"), lit, n

IDIOM_C = ["mean_diff_alpha_-10", "mean_diff_alpha_-15"]
PERP_C = ["v_PERP_raw_-10", "v_PERP_raw_-15"]
BRY_C = ["v_BRY_raw_+10", "v_BRY_raw_+15", "v_BRY_raw_+20"]

# ---------- P7 three-bar per-idiom (coherent-only, Wilson) ----------
bi = per_idiom(val, ["baseline"], True); vi = per_idiom(val, IDIOM_C, True)
vp = per_idiom(perp, PERP_C, True); vb = per_idiom(bry, BRY_C, True)
idioms = sorted(vi)
rows = []
for idi in idioms:
    def rc(d): lit, n = d.get(idi, (0, 0)); r = lit/n if n else float("nan"); lo, hi = wilson(lit, n); return r, lo, hi, n
    br, blo, bhi, bn = rc(bi); ir, ilo, ihi, iN = rc(vi); pr, plo, phi, pn = rc(vp); vr, vlo, vhi, vn = rc(vb)
    rows.append({"idiom": idi, "flagship": idi in FLAG, "baseline": br,
                 "vIDIOM": ir, "vIDIOM_lo": ilo, "vIDIOM_hi": ihi, "vIDIOM_n": iN,
                 "vPERP": pr, "vPERP_lo": plo, "vPERP_hi": phi, "vPERP_n": pn,
                 "vBRY": vr, "vBRY_lo": vlo, "vBRY_hi": vhi, "vBRY_n": vn,
                 "vBRY_minus_vPERP": vr - pr})
p7 = pd.DataFrame(rows).sort_values("vBRY_minus_vPERP")
p7.round(4).to_csv(os.path.join(OUT, "p7_per_idiom_components.csv"), index=False)
# draft plot
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(p7)); w = 0.26
for off, col, c in [(-w, "vIDIOM", "#4c72b0"), (0, "vPERP", "#dd8452"), (w, "vBRY", "#55a868")]:
    ax.bar(x+off, p7[col], w, label=col, color=c,
           yerr=[p7[col]-p7[col+"_lo"], p7[col+"_hi"]-p7[col]], capsize=2, ecolor="#333", error_kw={"lw": 0.7})
ax.scatter(x, p7["baseline"], marker="_", s=220, color="black", zorder=6, label="baseline")
ax.set_xticks(x)
labs = ax.set_xticklabels(p7["idiom"], rotation=55, ha="right", fontsize=8)
for t, fl in zip(labs, p7["flagship"]):
    if fl: t.set_fontweight("bold"); t.set_color("#b22222")
ax.set_ylabel("literal rate (coherent-only)"); ax.set_ylim(0, 1)
ax.set_title("Llama-3.2-3B per-idiom: v_IDIOM / v_PERP / v_BRY (pooled), baseline tick; flagship in red")
ax.legend(ncol=4, fontsize=9); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(OUT, "p7_per_idiom_components.png"), dpi=140)
print("=== P7 three-bar figure written ===")

# ---------- corrected pooled via batch-C confusion (reproduce 0.362/0.309) ----------
from sklearn.metrics import confusion_matrix
key3 = pd.read_csv(os.path.join(REPO, "judge_audit", "v3", "v3_key.csv"))
g3 = pd.read_csv(os.path.join(REPO, "judge_audit", "v3", "v3_resolved_GT.csv"))
vdf = key3.merge(g3, on="blind_id"); vdf["gold"] = vdf["human_label"].str.lower(); vdf["judge"] = vdf["judge_label"].str.lower()
def Pgj(stratum):
    s = vdf[vdf["stratum"] == stratum]; cm = confusion_matrix(s["gold"], s["judge"], labels=LABELS).astype(float)
    col = cm.sum(0); return np.divide(cm, col, out=np.zeros_like(cm), where=col > 0)
def judged_dist(doc, conds):
    cs = set(conds); c = collections.Counter()
    for r in doc["results"]:
        for cd, jl in (r.get("judge_v14") or {}).items():
            if cd not in cs: continue
            for j in jl:
                if j: c[j.get("label", "figurative")] += 1
    tot = sum(c.values()); return {l: c.get(l, 0)/tot for l in LABELS}
li = LABELS.index("literal")
def corrected(doc, conds, stratum):
    Pj = judged_dist(doc, conds); P = Pgj(stratum); return sum(Pj[LABELS[j]]*P[li, j] for j in range(len(LABELS)))

# ---------- P8 headline numbers ----------
def raw_over(doc, conds): r, _, _ = overall(doc, conds, False); return r
base_all = raw_over(val, ["baseline"])
vidiom_best = raw_over(val, ["mean_diff_alpha_-10"])
vidiom_pool_coh, _, _ = overall(val, IDIOM_C, True)
p4 = pd.read_csv(os.path.join(OUT, "p4_corrected_headline.csv")).set_index("model")
vidiom_corr = float(p4.loc["Llama-3.2-3B", "corrected_literal"])
vbry_best = max(raw_over(bry, [c]) for c in BRY_C)
vbry_pool_raw = raw_over(bry, BRY_C); vbry_pool_corr = corrected(bry, BRY_C, "v_BRY")
vperp_best = max(raw_over(perp, [c]) for c in PERP_C)
vperp_pool_raw = raw_over(perp, PERP_C); vperp_pool_corr = corrected(perp, PERP_C, "v_PERP")
geo = json.load(open(os.path.join(T3, "vectors", "geometry.json")))
vpar = np.load(os.path.join(T3, "vectors", "v_PAR.npy")).astype(np.float64)
vbry_vec = np.load(os.path.join(CONC, "vectors", "llama3b", "v_BRY.npy")).astype(np.float64)
cos_par_bry = abs(float(np.dot(vpar/np.linalg.norm(vpar), vbry_vec/np.linalg.norm(vbry_vec))))

# flagship rates (coherent-only pooled)
def rate(d, idi): lit, n = d.get(idi, (0, 0)); return (lit/n if n else None)
flagship = {idi: {"baseline": rate(bi, idi), "vIDIOM": rate(vi, idi), "vPERP": rate(vp, idi), "vBRY": rate(vb, idi)} for idi in FLAG}
# correlation PERP vs BRY per-idiom profiles
pv = np.array([rate(vp, i) for i in idioms]); bv = np.array([rate(vb, i) for i in idioms])
r_perp_bry = float(np.corrcoef(pv, bv)[0, 1])

# breadth from p2
p2 = pd.read_csv(os.path.join(OUT, "p2_breadth_counts.csv"))

checks = [("v_IDIOM best-alpha", vidiom_best, 0.518), ("v_IDIOM pooled coherent-only", vidiom_pool_coh, 0.552),
          ("v_IDIOM corrected", vidiom_corr, 0.535), ("v_BRY best-alpha raw", vbry_best, 0.437),
          ("v_BRY pooled raw", vbry_pool_raw, 0.423), ("v_BRY pooled corrected", vbry_pool_corr, 0.362),
          ("v_PERP best-alpha raw", vperp_best, 0.345), ("v_PERP pooled raw", vperp_pool_raw, 0.335),
          ("v_PERP pooled corrected", vperp_pool_corr, 0.309), ("cos(vIDIOM,vBRY)", c3b, -0.39),
          ("resid norm", geo["||v_IDIOM - proj_C|| (before renorm)"], 0.92)]
print("=== P8 sanity checks (got vs expected) ===")
for name, got, exp in checks:
    ok = approx(got, exp, 0.01 if "cos" in name or "resid" in name else 0.006)
    print(f"  {name:32} got={got:.4f} exp={exp}  {'OK' if ok else 'MISMATCH'}")
    if not ok: flag(f"{name}: got {got:.4f} vs expected {exp}")
# proj norm note (spec said 0.39 but proj-norm is 0.41; 0.39 == |cos|)
proj_norm = geo["||proj_C(v_IDIOM)|| (before renorm)"]
if not approx(proj_norm, 0.39, 0.005):
    flag(f"ablation proj-norm is {proj_norm:.3f} (spec said 0.39); note 0.39 == |cos(vIDIOM,vBRY)|, proj-norm differs")

p8 = {
  "conventions_note": "each value labeled by convention; raw=all-gens judge label; coherent-only excludes incoherent; corrected=batch-C confusion",
  "baseline_paper_conv_all_gens": round(base_all, 4),
  "random_band": {"random_additive": 0.259, "random_angular": 0.201, "source": "p1 literal-direction half"},
  "v_IDIOM": {"best_alpha_all_gens": round(vidiom_best, 4), "pooled_coherent_only": round(vidiom_pool_coh, 4),
              "corrected_headline_from_p4": round(vidiom_corr, 4)},
  "v_BRY": {"best_alpha_raw": round(vbry_best, 4), "pooled_raw": round(vbry_pool_raw, 4), "pooled_corrected": round(vbry_pool_corr, 4)},
  "v_PERP": {"best_alpha_raw": round(vperp_best, 4), "pooled_raw": round(vperp_pool_raw, 4), "pooled_corrected": round(vperp_pool_corr, 4)},
  "geometry_llama3b": {"cos_idiom_bry": round(c3b, 4),
                       "cos_idiom_gjbw": round(conc["Llama-3.2-3B"]["geometry"]["cos_idiom_gjbw"], 4),
                       "cos_gjbw_bry": round(conc["Llama-3.2-3B"]["geometry"]["cos_gjbw_bry"], 4)},
  "ablation_norm_shares": {"proj_norm": round(proj_norm, 4), "resid_norm": round(geo["||v_IDIOM - proj_C|| (before renorm)"], 4),
                           "note": "spec '0.39/0.92' = |cos(vIDIOM,vBRY)| and resid-norm; proj-norm=%.3f" % proj_norm},
  "flagship_per_idiom_coherent_only_pooled": {k: {kk: (round(vv, 4) if vv is not None else None) for kk, vv in v.items()} for k, v in flagship.items()},
  "per_idiom_corr_r_PERP_vs_BRY_pooled": round(r_perp_bry, 4),
  "breadth": {"pooled_gt_0.14_above_baseline": "19/20", "pooled_gt_0.40": "16/20", "source": "p2"},
  "verifications": {"abs_cos_vPAR_vBRY": round(cos_par_bry, 4)},
  "flags": flags_report,
}

# ---------- item 5: CI separation list (PERP vs BRY per idiom) ----------
sep = []
for idi in idioms:
    pl, pn = vp.get(idi, (0, 0)); bl, bn = vb.get(idi, (0, 0))
    pr = pl/pn if pn else float("nan"); brr = bl/bn if bn else float("nan")
    plo, phi = wilson(pl, pn); blo, bhi = wilson(bl, bn)
    if phi < blo:  sep.append((idi, "BRY", round(brr-pr, 3)))
    elif bhi < plo: sep.append((idi, "PERP", round(pr-brr, 3)))
p8["verifications"]["CI_separated_idioms_PERP_vs_BRY"] = [{"idiom": i, "higher": d, "gap": g} for i, d, g in sep]
exp_perp = {"out of the blue", "behind bars"}; exp_bry = {"break the ice", "green light", "lie low", "pull someone's leg", "spill the beans"}
got_perp = {i for i, d, g in sep if d == "PERP"}; got_bry = {i for i, d, g in sep if d == "BRY"}
print("=== item5 CI-separation ===")
print("  PERP-higher:", sorted(got_perp), "| expected:", sorted(exp_perp))
print("  BRY-higher:", sorted(got_bry), "| expected:", sorted(exp_bry))
if got_perp != exp_perp: flag(f"PERP-separated idioms {sorted(got_perp)} != expected {sorted(exp_perp)}")
if not exp_bry <= got_bry: flag(f"BRY-separated missing {sorted(exp_bry-got_bry)}")

json.dump(p8, open(os.path.join(OUT, "p8_section64_numbers.json"), "w"), indent=2)
print("\nFLAGS:", flags_report or "none")
print("Outputs:", sorted(f for f in os.listdir(OUT) if f.startswith(("p5", "p6", "p7", "p8"))))
