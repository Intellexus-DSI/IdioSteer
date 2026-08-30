# Camera-ready judge validation — Appendix J tables (v3)

_**Provenance tiers.** CALIBRATION items were used for judge selection and are reported separately — **never pooled into validation**. VALIDATION = v2 audit (600, four held-out models) + batches C (concreteness, Llama-3.2-3B) + D (probe/angular, Llama-3.2-3B). Gold produced by joint adjudication (Appendix J); no independent inter-annotator κ reported._

## T1 — Overall validation
| set | accuracy | Wilson 95% CI | κ | N |
|---|---|---|---|---|
| VALIDATION (all) | 0.919 | [0.899, 0.936] | 0.868 | 795 |
|   v2 only (4 held-out) | 0.932 | [0.909, 0.949] | 0.888 | 600 |
|   batches C+D (Llama-3.2-3B) | 0.882 | [0.829, 0.920] | 0.806 | 195 |

## T2 — Per model (validation)
| model | accuracy | Wilson 95% CI | κ | N | note |
|---|---|---|---|---|---|
| meta-llama/Llama-3.1-8B | 0.947 | [0.898, 0.973] | 0.917 | 150 |  |
| google/gemma-2-2b | 0.933 | [0.882, 0.963] | 0.892 | 150 |  |
| google/gemma-3-4b-pt | 0.927 | [0.873, 0.959] | 0.867 | 150 |  |
| google/gemma-3-12b-pt | 0.920 | [0.865, 0.954] | 0.868 | 150 |  |
| meta-llama/Llama-3.2-3B | 0.882 | [0.829, 0.920] | 0.806 | 195 | batches C+D: md_angular_extreme, probe, v_BRY, v_PERP |

## T3 — Per stratum (validation; new C/D strata included)
| stratum | accuracy | Wilson 95% CI | κ | N |
|---|---|---|---|---|
| baseline | 0.933 | [0.841, 0.974] | 0.753 | 60 |
| md_add_moderate | 0.980 | [0.930, 0.994] | 0.964 | 100 |
| md_add_extreme | 0.958 | [0.906, 0.982] | 0.934 | 120 |
| md_angular_extreme | 0.869 | [0.811, 0.911] | 0.779 | 175 |
| probe | 0.970 | [0.915, 0.990] | 0.921 | 100 |
| random | 0.910 | [0.838, 0.952] | 0.849 | 100 |
| v_BRY | 0.900 | [0.808, 0.951] | 0.830 | 70 |
| v_PERP | 0.843 | [0.740, 0.910] | 0.748 | 70 |

## T4 — Confusion matrices

**Overall validation** (rows=gold, cols=judge; N=795)

| gold\judge | figurative | literal | incoherent |
|---|---|---|---|
| figurative | 392 | 15 | 1 |
| literal | 11 | 186 | 1 |
| incoherent | 27 | 9 | 153 |

**md_angular_extreme (all models)** (rows=gold, cols=judge; N=175)

| gold\judge | figurative | literal | incoherent |
|---|---|---|---|
| figurative | 55 | 2 | 1 |
| literal | 0 | 18 | 1 |
| incoherent | 14 | 5 | 79 |

**Batch C (concreteness: v_BRY + v_PERP)** (rows=gold, cols=judge; N=140)

| gold\judge | figurative | literal | incoherent |
|---|---|---|---|
| figurative | 64 | 7 | 0 |
| literal | 4 | 38 | 0 |
| incoherent | 5 | 2 | 20 |

## Corrected literal rates

Applying batch-C P(gold|judge) to the pooled judged distributions (feeds §6.4 / concreteness appendix):

| condition group | audit confusion | raw literal | corrected literal | N_audit |
|---|---|---|---|---|
| v_BRY (+10/+15/+20) | batch-C v_BRY (N=70) | 0.423 | 0.362 | 70 |
| v_PERP (-10/-15) | batch-C v_PERP (N=70) | 0.335 | 0.309 | 70 |

(Headline MD-additive per-model corrected rates are unchanged from v2 §4 — recompute here once all five models' MD-additive strata carry validation gold.)

## Flagship-idiom agreement (within Batch C)

| idiom | accuracy | N |
|---|---|---|
| out of the blue | 1.000 | 9 |
| behind bars | 1.000 | 8 |
| spill the beans | 1.000 | 9 |
| in the pink | 0.857 | 7 |
| (non-flagship) | 0.841 | 107 |

## CALIBRATION tier (judge-selection dev set — reported separately, NOT pooled into validation)

- source: `eval_gemma-4-31b-it_v14_coherence_strict_20260418_173826__rescored_20260504_143449.json`
- **accuracy 0.900 [0.859, 0.930], κ 0.821, N=280**
- This is the full calibration set (N=280), used to SELECT the judge; reported here separately and **never pooled into the validation tables above**. Strata: baseline + MD-additive ±10/±20 + random ±20 (no probe/angular — those are covered for Llama-3.2-3B by validation batch D).
