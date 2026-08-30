# Reproduce

All commands run from `src/` with `IDIOSTEER_ROOT` = repo root. Secrets via env only (`HF_TOKEN`, `OPENROUTER_API_KEY`).

## 0. Environment
```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
export IDIOSTEER_ROOT="$PWD"
```

## 1. Reproduce tables/figures from shipped aggregates (no GPU, no key; seconds)
```bash
cd src
python figures/fig_layer_decodability.py llama3b        # per-layer decodability (Fig)
python figures/fig_concreteness_per_idiom.py            # concreteness per-idiom (Fig)
python figures/fig4_layer_sweep_3panel.py               # Fig 4 (labels-only JSON in assets/)
python figures/appM_layer_sweep_heatmap.py              # Appendix M
```
Table numbers are already in `assets/p1`–`p9` and `results/`.

## 2. Rebuild steering vectors (GPU; minutes/model; needs HF_TOKEN)
```bash
python -m analysis.build_concreteness_vectors llama3b   # v_IDIOM / v_BRY / bare-word v_GJ + gates
python -m analysis.build_ablation_vectors               # v_PERP / v_PAR from saved vectors
```
Gemma-3 is loaded in bfloat16 automatically.

## 3. Full regeneration (expensive: ~106 GPU-hours + ~$92 judge)
```bash
bash scripts/run_eval.sh v_BRY                          # generate steered completions (GPU)
OPENROUTER_API_KEY=... bash scripts/run_judge.sh gen.json judged.json 8   # judge (API cost)
python -m analysis.judge_validation_v3                  # then re-derive aggregates
```

## Runtime/cost summary
| step | hardware | time | cost |
|---|---|---|---|
| figures/tables from aggregates | CPU | seconds | $0 |
| one vector + 5 prefixes + judge | 1 GPU + API | ~2 min | ~$0.01 |
| full generation grid (5 models) | 1 GPU | ~106 h | $0 |
| full judge pass | API | ~11 h | ~$92 |
