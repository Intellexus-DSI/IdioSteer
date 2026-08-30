# Figure / table -> source artifact -> script

Layer-sweep JSON filenames in the source repo were timestamped and not self-explanatory; here they are the
**labels-only** copies in `assets/` (completions + judge reasoning stripped).

| paper float | source artifact | script |
|---|---|---|
| Table (5-model robustness half-grid) | `assets/p1_robustness_by_family.csv` | `analysis/aggregate_robustness_peridiom_corrected.py` |
| Breadth counts (§ per-idiom) | `assets/p2_breadth_counts.csv` | same |
| Per-idiom headline table (appendix) | `assets/p3_per_idiom_headline.csv` | same |
| Corrected headline rates (Appendix J §4) | `assets/p4_corrected_headline.csv` | `analysis/verify_corrected_stratum.py` |
| Concreteness geometry (5-model) | `assets/p5_geometry.csv` | `analysis/aggregate_concreteness_geometry_gates.py` |
| Stage-4 gates | `assets/p6_gates.csv` | `analysis/whitened_gates.py`, `analysis/build_concreteness_vectors.py` |
| Concreteness per-idiom figure | `assets/p7_per_idiom_components.csv` | `figures/fig_concreteness_per_idiom.py` |
| §6.4 headline numbers | `assets/p8_section64_numbers.json` | `analysis/aggregate_concreteness_geometry_gates.py` |
| Outlier diagnostic | `assets/p9_outlier_items12.json` | `analysis/outlier_diagnostic_ablation.py`, `..._geometry.py` |
| **Figure 4** (3-panel layer sweep) | `assets/layer_sweep_{llama3b,llama8b,gemma2_2b}_labelsonly.json` (+ baseline from `results/per_condition_label_counts.csv`) | `figures/fig4_layer_sweep_3panel.py` |
| **Appendix M** (llama8b heatmap) | `assets/layer_sweep_llama8b_labelsonly.json` | `figures/appM_layer_sweep_heatmap.py` |
| Per-layer decodability | `assets/layer_vs_accuracy_{llama3b,gemma2_2b}.csv` | `figures/fig_layer_decodability.py` |
| Judge validation (per-model/stratum, κ, CIs) | `results/audit_report_v3.md`, `results/audit_gold_joined.csv` | `analysis/judge_validation_v3.py` |
