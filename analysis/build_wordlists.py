"""
concreteness/build_wordlists.py
===============================
Stage 2b word selection + v_GJ word selection, from the local Brysbaert norms.
Produces releasable word lists filtered by cross-model tokenizer coverage.

v_BRY pools : concrete Conc.M >= 4.5 ; abstract Conc.M <= 2.0 ; NOUNS only
              (Dom_Pos == 'Noun') ; sample 250/pool (seed 42).
v_GJ  pools : concrete Conc.M >= 4.0 ; abstract Conc.M <= 2.0 ; NO POS filter
              (Ghosh & Jiang recipe) ; cap 500/pool (seed 42).

Shared hygiene for both: lowercase single alphabetic word, and drop any word
that tokenizes to > 4 pieces in ANY of the five model tokenizers (trivial
vocab-coverage sense). Word tokenized in-context (leading space).
"""
import os, json, random
import pandas as pd

from idiosteer.pipeline import MODELS

REPO = os.path.expanduser(".")
NORMS = os.path.join(REPO, "data_outputs", "Concreteness_ratings_Brysbaert_et_al_BRM.txt")
OUT = os.path.join(REPO, "concreteness", "wordlists")
os.makedirs(OUT, exist_ok=True)
SEED = 42
MAX_PIECES = 4

# ── Load norms ───────────────────────────────────────────────────────────────
df = pd.read_csv(NORMS, sep="\t")
df["Word"] = df["Word"].astype(str)
df = df[df["Bigram"] == 0]                                   # single-word entries only
df = df[df["Word"].str.fullmatch(r"[A-Za-z]+")]              # alphabetic
df["Word"] = df["Word"].str.lower()
df = df.drop_duplicates("Word")
df["Conc.M"] = pd.to_numeric(df["Conc.M"], errors="coerce")
df = df.dropna(subset=["Conc.M"])
print(f"Brysbaert usable single alphabetic words: {len(df)}")

# ── Load all 5 tokenizers for cross-model coverage filter ────────────────────
from transformers import AutoTokenizer
tokenizers = {}
for key, m in MODELS.items():
    print(f"  loading tokenizer: {key} ({m['hf']})", flush=True)
    tokenizers[key] = AutoTokenizer.from_pretrained(m["hf"])


def max_pieces_across_models(word):
    """Max #subword pieces for ' word' (in-context, leading space) across all 5 tokenizers."""
    mx = 0
    for tok in tokenizers.values():
        n = len(tok.tokenize(" " + word))
        if n > mx:
            mx = n
    return mx


def coverage_ok(words):
    keep, dropped = [], 0
    for w in words:
        if max_pieces_across_models(w) <= MAX_PIECES:
            keep.append(w)
        else:
            dropped += 1
    return keep, dropped


def select(mask, pos_filter_noun, n_target, cap):
    sub = df[mask].copy()
    if pos_filter_noun:
        sub = sub[sub["Dom_Pos"].astype(str).str.lower() == "noun"]
    cands = sorted(sub["Word"].tolist())
    cands, n_covdrop = coverage_ok(cands)
    n_avail = len(cands)
    rng = random.Random(SEED)
    if cap is not None and n_avail > cap:
        chosen = sorted(rng.sample(cands, cap))
        relaxed = False
    elif n_target is not None and n_avail >= n_target:
        chosen = sorted(rng.sample(cands, n_target))
        relaxed = False
    else:
        chosen = cands  # take all available (short of target)
        relaxed = True
    return chosen, {"n_pool_pre_cov": len(cands) + n_covdrop, "n_cov_dropped": n_covdrop,
                    "n_available": n_avail, "n_selected": len(chosen), "short_of_target": relaxed}


prov = {"seed": SEED, "max_pieces": MAX_PIECES, "norms_file": os.path.relpath(NORMS, REPO),
        "n_brysbaert_usable": len(df), "pools": {}}

# v_BRY (nouns, >=4.5 / <=2.0, sample 250)
bry_conc, s1 = select(df["Conc.M"] >= 4.5, True, 250, None)
bry_abst, s2 = select(df["Conc.M"] <= 2.0, True, 250, None)
prov["pools"]["v_BRY_concrete_ge4.5_noun"] = s1
prov["pools"]["v_BRY_abstract_le2.0_noun"] = s2

# v_GJ (no POS, >=4.0 / <=2.0, cap 500)
gj_conc, s3 = select(df["Conc.M"] >= 4.0, False, None, 500)
gj_abst, s4 = select(df["Conc.M"] <= 2.0, False, None, 500)
prov["pools"]["v_GJ_concrete_ge4.0_noPOS"] = s3
prov["pools"]["v_GJ_abstract_le2.0_noPOS"] = s4


def dump(path, words):
    with open(path, "w") as fh:
        fh.write("\n".join(words) + "\n")


dump(os.path.join(OUT, "concrete_250.txt"), bry_conc)
dump(os.path.join(OUT, "abstract_250.txt"), bry_abst)
dump(os.path.join(OUT, "gj_concrete_500.txt"), gj_conc)
dump(os.path.join(OUT, "gj_abstract_500.txt"), gj_abst)
with open(os.path.join(OUT, "provenance.json"), "w") as fh:
    json.dump(prov, fh, indent=2)

print("\n=== word-list selection ===")
for k, v in prov["pools"].items():
    print(f"  {k}: {v}")
print(f"\nsaved to {OUT}/  (concrete_250, abstract_250, gj_concrete_500, gj_abstract_500, provenance.json)")
