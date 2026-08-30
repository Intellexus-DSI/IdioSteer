# analysis/

Reference scripts that produce the paper's tables and numbers from judged generations. They import the
`idiosteer` package (run from the repo root with `IDIOSTEER_ROOT=$PWD`). Highlights:

- `build_concreteness_vectors.py`, `build_ablation_vectors.py`, `build_vgj_bareword.py`, `build_wordlists.py` — vector construction.
- `run_generation_conditions.py` — steered generation grid.
- `aggregate_robustness_peridiom_corrected.py`, `aggregate_concreteness_geometry_gates.py`, `coherence_and_corrected_bestalpha.py`, `verify_corrected_stratum.py` — table aggregation (assets/p1-p8).
- `whitened_gates.py`, `outlier_diagnostic_geometry.py`, `outlier_diagnostic_ablation.py` — geometry / Gemma-3 outlier diagnostics (assets/p6, p9).
- `judge_validation_v2.py`, `judge_validation_v3.py`, `layer_decodability_sweep.py`.
