# IdioSteer: Evaluating Activation Steering for Figurative vs. Literal Idiom Interpretation

**EMNLP 2026** · [[Paper]](#) · dual-licensed (code: Apache-2.0 · data: CC BY 4.0)

Can additive activation steering shift an LLM's *figurative* default reading of a sentence-final idiom
toward a *literal* one? We evaluate five models and several steering families (mean-difference / probe /
random × additive / angular), with an LLM-as-a-judge validated against human annotations.

**5 models | 20 idioms × 100 prefixes | mean-diff / probe / random vectors | additive + angular operators**

---

## 🚀 Quick Start

```bash
git clone <repo-url> && cd idiosteer-release
pip install -r requirements.txt
export IDIOSTEER_ROOT="$PWD"
# reproduce every figure/table from committed aggregates (CPU, seconds, no key):
python run_all.py
```

For GPU steering you also need gated weights: `export HF_TOKEN=...`. For the judge: `export OPENROUTER_API_KEY=...`.

> **Hardware:** vector construction + generation needs one GPU (Gemma-3 requires ≥24 GB and bfloat16).
> The figures/tables reproduce on CPU from `assets/`.

---

## 📋 Overview

**Research question:** does injecting a concreteness / figurative-vs-literal direction into the residual
stream at the final idiom token flip the model's interpretation from figurative to literal while staying
coherent? We measure the **literal rate** and **coherence rate** under an LLM judge, validate the judge
against human labels, and dissociate the mean-difference direction from a pure concreteness direction.

### Metrics
- **Literal rate** — fraction of completions the judge reads as the literal interpretation.
- **Coherence rate** — fraction that are coherent (figurative or literal, not incoherent).
- Reported both **all-generations** (headline) and **coherent-only**; see `docs/FIGURE_SOURCES.md`.

---

## 📁 Data

Ships derived, redistributable data (no raw model completions — see Ethics below):

| Directory | Contents |
|-----------|----------|
| `data/idiosteer/` | 100 validated prefixes × 20 idioms, with idiom-span offsets + construction guidelines |
| `data/concreteness/` | v_BRY (Brysbaert) and v_GJ word lists, 3 templates, provenance |
| `data/idiolink_subset/` | retained IdioLink ids + counts + filter script (IdioLink itself not redistributed) |

External sources (not redistributed): Brysbaert et al. (2014) norms and the full IdioLink dataset —
`python download_data.py --idiolink` pulls IdioLink from HuggingFace; Brysbaert URL is in
`data/concreteness/README.md`.

---

## ⚙️ Models & Configuration

Central config in `config.yaml`. Canonical read (source) layer per model:

| Model | HF id | source layer L_s | dtype |
|-------|-------|------------------|-------|
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B` | 14 | fp16 |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B` | 16 | fp16 |
| Gemma-2-2B | `google/gemma-2-2b` | 20 | fp16 |
| Gemma-3-4B | `google/gemma-3-4b-pt` | 17 | **bf16** |
| Gemma-3-12B | `google/gemma-3-12b-pt` | 24 | **bf16** |

Steering: additive at injection layer 2, α = `multiplier × ‖md_source‖`; judge: `google/gemma-4-31b-it`
(`v14_coherence_strict`, T=0.0). All in `config.yaml`.

---

## 🏃 Running

```bash
python run_figures.py                       # figures from assets/ (CPU)
python run_build_vectors.py --model llama3b # build steering vectors (GPU, HF_TOKEN)
python run_eval.py --vector v_BRY           # steered generation grid (GPU)
python run_judge.py --input gen.json --output judged.json   # LLM judge (OPENROUTER_API_KEY)
pytest -q                                    # sanity tests (no GPU/API)
```

Full end-to-end reproduction and compute/cost expectations: `docs/REPRODUCE.md`.

---

## 🗂️ Repository layout

```
idiosteer/        importable package: pipeline, hooks, vectors, judge, utils (load_config)
analysis/          scripts that build vectors and aggregate tables (assets/p1-p9)
figures/           plotters (read assets/, write TrueType PDFs)
data/              redistributable derived data (idiosteer, concreteness, idiolink_subset)
assets/            committed paper tables (p1-p9) + figure PDFs + labels-only sweep JSONs
results/           judge-audit aggregates (labels/counts only)
docs/              REPRODUCE, FIGURE_SOURCES, ANNOTATION
tests/             pytest sanity checks
run_*.py           entry-point CLIs;  config.yaml / keys.yaml.example;  download_data.py
```

---

## 🔒 Ethics — no raw model completions
Per the paper's Ethics Statement, **no raw generated text is released**. Every committed artifact holds
only judge/human **labels and aggregate counts** (the layer-sweep JSONs are stripped of completions and
judge reasoning). Do not add raw completions to this repo.

## 📄 Licensing
Code under **Apache-2.0** (`LICENSE`); data & result artifacts under **CC BY 4.0** (`LICENSE-DATA`).
`requirements.txt` is pinned to the exact experiment versions for reproducibility.

## 📝 Citation
See `CITATION.cff`.

## ⚠️ Known limitations
Generation is sampling-based (aggregate rates stable, exact text not reproducible); the judge depends on
a hosted model; Gemma-3 cross-model cosines are anisotropy-inflated (report whitened geometry); IdioLink
is not redistributed.
