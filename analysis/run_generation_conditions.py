"""
concreteness/tier3/run_tier3.py
==============================
Tier-3 behavioral dissociation on Llama-3.2-3B. REUSES the existing generation
mechanics (AdditiveSteeringHook + generate params from three_methods_evaluation.py)
and the existing judge (annotations/judge_v14.py, google/gemma-4-31b-it) UNCHANGED.

Per condition (order v_PERP, v_BRY, v_PAR, v_GJ): generate 100 sentences x 8 alphas
x 5 samples, judge via judge_v14.py subprocess, compute literal/coherence rates +
lexicon-concreteness, print summary, write tier3/results_<cond>.json.

Vectors are unit-norm; alpha_raw in +-{5,10,15,20} gives matched perturbation.
"""
import os, sys, json, time, math, subprocess, argparse, re, hashlib
import numpy as np
import torch

REPO = os.path.expanduser(".")
CONC = os.path.join(REPO, "concreteness")
T3 = os.path.join(CONC, "tier3")
os.makedirs(T3, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # for pipeline (model loader / find_last_span_pos)
from idiosteer.pipeline import load_model_and_tokenizer, MODELS

# ── faithful generation constants (from three_methods_evaluation.py) ─────────
APPLY_LAYER = 2
K = 5
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.7
TOP_P = 0.9
ALPHAS_FULL = [-20.0, -15.0, -10.0, -5.0, 5.0, 10.0, 15.0, 20.0]
ALPHAS_GJ = [-20.0, -10.0, 10.0, 20.0]          # reduced robustness grid (spec allows)
V_IDIOM_RAWNORM = 4.182276                        # pilot v_IDIOM raw norm (alpha_factor = alpha_raw / this)

VALIDATED_FULL = os.path.join(REPO, "data_outputs", "idiom_evaluations", "judge_v14",
                              "judged_three_methods_eval_validated_FULL.json")
JUDGE = os.path.join(REPO, "annotations", "judge_v14.py")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VEC_PATHS = {
    "v_BRY":  os.path.join(CONC, "vectors", "llama3b", "v_BRY.npy"),
    "v_GJ":   os.path.join(CONC, "vectors", "llama3b", "v_GJ.npy"),
    "v_PERP": os.path.join(T3, "vectors", "v_PERP.npy"),
    "v_PAR":  os.path.join(T3, "vectors", "v_PAR.npy"),
}


# ── REUSED verbatim: AdditiveSteeringHook (three_methods_evaluation.py:494) ──
class AdditiveSteeringHook:
    def __init__(self, model, layer, vector, alpha, target_pos):
        self.vector = vector; self.alpha = alpha; self.target_pos = target_pos
        self._handle = model.model.layers[layer].register_forward_hook(self._hook)
    def _hook(self, module, input, output):
        # transformers-version-agnostic unpack/repack (repo used _unpack/_repack helpers):
        # some versions return a bare hidden_states tensor, others a (hidden_states, ...) tuple.
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if self.target_pos >= h.shape[1]:
            return output
        h_new = h.clone()
        h_new[:, self.target_pos, :] += self.alpha * self.vector
        return ((h_new,) + tuple(output[1:])) if is_tuple else h_new
    def remove(self):
        self._handle.remove()


# ── REUSED verbatim: generate_completions (three_methods_evaluation.py:556) ──
def generate_completions(model, tokenizer, input_ids, n=K, max_new=MAX_NEW_TOKENS):
    # Same params as three_methods_evaluation.py; the K=5 samples are drawn via
    # num_return_sequences (i.i.d. batched) instead of a 5-iteration loop — same
    # sampling distribution, ~5x faster. Hook applies only at prefill (target_pos
    # >= 1 during decode -> KV-cache-safe early return), unchanged behavior.
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=attn, max_new_tokens=max_new,
                             do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                             num_return_sequences=n, pad_token_id=tokenizer.eos_token_id)
    plen = input_ids.shape[1]
    return [tokenizer.decode(out[i][plen:], skip_special_tokens=True) for i in range(n)]


def load_eval_sentences(limit=None):
    d = json.load(open(VALIDATED_FULL))
    rows = []
    for r in d["results"]:
        rows.append({"idiom": r["idiom"], "sentence": r["sentence"],
                     "variant": r.get("variant", 0),
                     "target_pos": int(r["last_idiom_token_idx"]),
                     "last_idiom_token": r.get("last_idiom_token", "")})
    return rows[:limit] if limit else rows


def load_vec(name):
    v = np.load(VEC_PATHS[name]).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)   # ensure unit
    return torch.tensor(v, dtype=torch.float16, device=DEVICE)


# ── FIX A: empirical literal-direction sign (never inferred from name) ───────
def compute_literal_signs(cond_names):
    """literal_direction_sign(v) = +1 if cos(v, v_BRY) > 0 else -1  (v_BRY +pole = concrete;
    literal continuations are the concrete ones). v_PERP is orthogonal to the concreteness
    subspace -> hardcode -1 (and assert near-orthogonality). Hard-fail on mismatch."""
    vbry = np.load(VEC_PATHS["v_BRY"]).astype(np.float64); vbry /= (np.linalg.norm(vbry) + 1e-8)
    EXPECTED = {"v_PERP": -1, "v_BRY": +1, "v_PAR": -1, "v_GJ": +1}
    signs, table = {}, []
    print("\n=== FIX A: literal-direction signs (cos with v_BRY, +pole=concrete) ===", flush=True)
    for c in cond_names:
        v = np.load(VEC_PATHS[c]).astype(np.float64); v /= (np.linalg.norm(v) + 1e-8)
        cb = float(np.dot(v, vbry))
        if c == "v_PERP":
            assert abs(cb) < 0.05, f"v_PERP not ~orthogonal to v_BRY (cos={cb:.4f})"
            sign = -1
        else:
            sign = +1 if cb > 0 else -1
        exp = EXPECTED.get(c)
        ok = exp is None or sign == exp
        print(f"  {c}: cos(v,v_BRY)={cb:+.4f} -> literal_dir_sign={sign:+d} (expected {exp}) {'OK' if ok else 'MISMATCH'}", flush=True)
        assert ok, f"{c}: literal_dir_sign {sign} != expected {exp}"
        signs[c] = sign
        table.append({"vector": c, "cos_with_v_BRY": round(cb, 4), "literal_dir_sign": sign})
    json.dump(table, open(os.path.join(T3, "literal_sign_table.json"), "w"), indent=2)
    return signs


# ── lexicon concreteness scorer ──────────────────────────────────────────────
def build_conc_dict():
    import pandas as pd
    df = pd.read_csv(os.path.join(REPO, "data_outputs", "Concreteness_ratings_Brysbaert_et_al_BRM.txt"), sep="\t")
    df = df[df["Bigram"] == 0]
    df["Word"] = df["Word"].astype(str).str.lower()
    df["Conc.M"] = pd.to_numeric(df["Conc.M"], errors="coerce")
    df = df.dropna(subset=["Conc.M"])
    return dict(zip(df["Word"], df["Conc.M"]))

_STOP = set("""a an the of to in on at for and or but if then else with without within into onto from by as is
are was were be been being am do does did doing have has had having i you he she it we they them his her its
their our your my me him us this that these those there here what which who whom whose when where why how not no
nor so than too very can will just should now up down out over under again further once about above below off
""".split())

def get_lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        wnl.lemmatize("tables")  # trigger wordnet load; raises if missing
        return lambda w: wnl.lemmatize(wnl.lemmatize(w, "n"), "v")
    except Exception as e:
        print(f"  [lexicon] nltk lemmatizer unavailable ({type(e).__name__}); light fallback", flush=True)
        def light(w):
            for suf in ("ing", "ed", "es", "s"):
                if len(w) > len(suf) + 2 and w.endswith(suf):
                    return w[:-len(suf)]
            return w
        return light

def score_concreteness(completions, conc_dict, lemm):
    """Return (mean_conc_over_found, coverage_pct). Content tokens = alpha, len>=2, non-stopword."""
    found, total = [], 0
    for c in completions:
        for tok in re.findall(r"[a-zA-Z]+", (c or "").lower()):
            if len(tok) < 2 or tok in _STOP:
                continue
            total += 1
            lem = lemm(tok)
            v = conc_dict.get(lem, conc_dict.get(tok))
            if v is not None:
                found.append(v)
    mean = float(np.mean(found)) if found else float("nan")
    cov = (len(found) / total * 100.0) if total else 0.0
    return mean, cov, len(found), total


# ── judge a generation file via the existing CLI (unchanged) ─────────────────
def run_judge(gen_path, judged_path, workers):
    cmd = [sys.executable, JUDGE, "--input", gen_path, "--output", judged_path, "--max-workers", str(workers)]
    print(f"  [judge] {' '.join(cmd[-6:])}", flush=True)
    subprocess.run(cmd, check=True)


def metrics_from_judged(judged_path, conc_dict, lemm):
    d = json.load(open(judged_path))
    # accumulate per-condition
    per = {}
    for r in d["results"]:
        for cond, judglist in (r.get("judge_v14") or {}).items():
            comps = r["completions"][cond]
            p = per.setdefault(cond, {"lit": 0, "coh": 0, "fig": 0, "inc": 0, "n": 0, "comps": []})
            for j in judglist:
                if j is None:
                    continue
                p["n"] += 1
                lab = j.get("label", "figurative")
                if lab == "literal": p["lit"] += 1
                elif lab == "incoherent": p["inc"] += 1
                else: p["fig"] += 1
                if j.get("is_coherent") is True: p["coh"] += 1
            p["comps"].extend(comps)
    out = {}
    for cond, p in per.items():
        n = max(p["n"], 1)
        mean_c, cov, nf, nt = score_concreteness(p["comps"], conc_dict, lemm)
        out[cond] = {"n": p["n"], "literal_rate": p["lit"]/n, "coherence_rate": p["coh"]/n,
                     "figurative_rate": p["fig"]/n, "incoherent_rate": p["inc"]/n,
                     "out_concreteness": mean_c, "conc_coverage_pct": cov}
    return out


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="v_PERP,v_BRY,v_PAR,v_GJ")
    ap.add_argument("--limit-sentences", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="2 sentences, 2 alphas, first condition only")
    args = ap.parse_args()

    conds = args.conditions.split(",")
    if args.smoke:
        conds = conds[:1]; args.limit_sentences = 2

    signs = compute_literal_signs(conds)   # FIX A: empirical sign labeling (hard-fails on mismatch)
    conc_dict = build_conc_dict(); lemm = get_lemmatizer()
    print(f"Brysbaert dict: {len(conc_dict)} words | conditions order: {conds}", flush=True)

    sents = load_eval_sentences(args.limit_sentences)
    print(f"Eval sentences: {len(sents)}", flush=True)

    print("Loading Llama-3.2-3B...", flush=True)
    model, tok = load_model_and_tokenizer(MODELS["llama3b"]["hf"])

    for cname in conds:
        vec = load_vec(cname)
        alphas = ALPHAS_GJ if (cname == "v_GJ" and not args.smoke) else (ALPHAS_FULL[:2] if args.smoke else ALPHAS_FULL)
        cond_names = [f"{cname}_raw_{a:+g}" for a in alphas]
        print(f"\n{'#'*66}\nCONDITION {cname}  alphas={alphas}\n{'#'*66}", flush=True)
        t0 = time.time()

        # pre-tokenize prefixes
        results = []
        for si, s in enumerate(sents):
            ids = tok(s["sentence"], return_tensors="pt").input_ids.to(DEVICE)
            entry = {"idiom": s["idiom"], "sentence": s["sentence"], "variant": s["variant"],
                     "last_idiom_token_idx": s["target_pos"], "completions": {}}
            for a, nm in zip(alphas, cond_names):
                hook = AdditiveSteeringHook(model, APPLY_LAYER, vec, a, s["target_pos"])
                try:
                    entry["completions"][nm] = generate_completions(model, tok, ids)
                finally:
                    hook.remove()
            results.append(entry)
            if (si + 1) % 20 == 0:
                print(f"    gen {si+1}/{len(sents)}  ({time.time()-t0:.0f}s)", flush=True)

        gen_path = os.path.join(T3, f"gen_{cname}.json")
        judged_path = os.path.join(T3, f"judged_{cname}.json")
        vhash = hashlib.sha256(open(VEC_PATHS[cname], "rb").read()).hexdigest()
        json.dump({"metadata": {"model": MODELS["llama3b"]["hf"], "condition_group": cname,
                                "apply_layer": APPLY_LAYER, "alphas_raw": alphas,
                                "vector_path": VEC_PATHS[cname], "vector_sha256": vhash,
                                "literal_dir_sign": signs[cname],
                                "v_idiom_raw_norm": V_IDIOM_RAWNORM, "temperature": TEMPERATURE,
                                "top_p": TOP_P, "max_new_tokens": MAX_NEW_TOKENS, "K": K},
                   "results": results}, open(gen_path, "w"), indent=2)
        print(f"  gen saved: {gen_path}  ({time.time()-t0:.0f}s generation)", flush=True)

        run_judge(gen_path, judged_path, args.workers)
        m = metrics_from_judged(judged_path, conc_dict, lemm)

        # attach alpha + semantic ordering
        res = {"group": cname, "alphas_raw": alphas, "literal_dir_sign": signs[cname], "grid": {}}
        for a, nm in zip(alphas, cond_names):
            direction = "toward_literal" if a * signs[cname] > 0 else "toward_figurative"
            res["grid"][nm] = {"alpha_raw": a, "alpha_factor": round(a / V_IDIOM_RAWNORM, 3),
                               "semantic_direction": direction, **m.get(nm, {})}
        json.dump(res, open(os.path.join(T3, f"results_{cname}.json"), "w"), indent=2)

        # print incremental summary
        print(f"\n----- {cname} SUMMARY (sign={signs[cname]:+d}) "
              f"(α_raw | α_fac | direction | literal | coher | out-conc | cov%) -----", flush=True)
        for a, nm in zip(alphas, cond_names):
            g = res["grid"][nm]
            print(f"  {a:+6.1f} | {g['alpha_factor']:+5.2f} | {g['semantic_direction']:>16} | "
                  f"lit={g.get('literal_rate',0):.3f} | coh={g.get('coherence_rate',0):.3f} | "
                  f"conc={g.get('out_concreteness',float('nan')):.3f} | cov={g.get('conc_coverage_pct',0):.0f}%", flush=True)
        print(f"  [{cname} done in {time.time()-t0:.0f}s]", flush=True)

    print("\nALL_TIER3_CONDITIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
