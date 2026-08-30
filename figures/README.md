# figures/

Plotters for the paper floats. Each reads committed aggregates from `assets/` and writes a PDF+PNG
(TrueType embedded, `pdf.fonttype=42`). Run via `python run_figures.py`.

- `fig_layer_decodability.py <model>` -> `assets/layer_vs_accuracy_<model>.pdf`
- `fig_concreteness_per_idiom.py`     -> `assets/p7_per_idiom_components.pdf`
- `fig4_layer_sweep_3panel.py`        -> Figure 4 (reads `assets/layer_sweep_*_labelsonly.json`)
- `appM_layer_sweep_heatmap.py`       -> Appendix M
See `docs/FIGURE_SOURCES.md` for the float -> artifact -> script map.
