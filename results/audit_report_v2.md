# Audit report v2 — Judge vs Gold (batch-1, 4 held-out models)

_Gold labels were produced by **joint annotation/adjudication** (two annotators, paper Appendix J protocol) and delivered as a single resolved file; therefore **no independent inter-annotator κ is reported for this batch** — the two annotators did not label independently._

## 1. Integrity

- rows: 600 | unique blind_ids: 600 | duplicates: none
- join to key: 600/600 joined | not-joined: none
- labels ∈ {figurative,literal,incoherent}: invalid rows: none
- sanity: xlsx retained judge col vs key.judge_label mismatches: 0 (expect 0)
- **analysis N = 600**

GT vs Judge label distribution:

| label | gold | judge |
|---|---|---|
| figurative | 306 (51.0%) | 325 (54.2%) |
| literal | 150 (25.0%) | 156 (26.0%) |
| incoherent | 144 (24.0%) | 119 (19.8%) |

## 2. Judge vs Gold

### T1 — Overall
| accuracy | Wilson 95% CI | Cohen's κ | N |
|---|---|---|---|
| 0.932 | [0.909, 0.949] | 0.888 | 600 |

### T2 — Per model (headline)
| model | accuracy | Wilson 95% CI | κ | N |
|---|---|---|---|---|
| meta-llama/Llama-3.1-8B | 0.947 | [0.898, 0.973] | 0.917 | 150 |
| google/gemma-2-2b | 0.933 | [0.882, 0.963] | 0.892 | 150 |
| google/gemma-3-4b-pt | 0.927 | [0.873, 0.959] | 0.867 | 150 |
| google/gemma-3-12b-pt | 0.920 | [0.865, 0.954] | 0.868 | 150 |

### T3 — Per stratum
| stratum | accuracy | Wilson 95% CI | κ | N |
|---|---|---|---|---|
| baseline | 0.933 | [0.841, 0.974] | 0.753 | 60 |
| md_add_moderate | 0.980 | [0.930, 0.994] | 0.964 | 100 |
| md_add_extreme | 0.958 | [0.906, 0.982] | 0.934 | 120 |
| md_angular_extreme | 0.864 | [0.798, 0.911] | 0.773 | 140 |
| probe | 0.975 | [0.913, 0.993] | 0.932 | 80 |
| random | 0.910 | [0.838, 0.952] | 0.849 | 100 |

### T4 — Confusion matrices

**Overall** (rows=gold, cols=judge; N=600)

| gold\judge | figurative | literal | incoherent |
|---|---|---|---|
| figurative | 299 | 6 | 1 |
| literal | 6 | 143 | 1 |
| incoherent | 20 | 7 | 117 |

**md_angular_extreme** (rows=gold, cols=judge; N=140)

| gold\judge | figurative | literal | incoherent |
|---|---|---|---|
| figurative | 40 | 0 | 1 |
| literal | 0 | 17 | 1 |
| incoherent | 12 | 5 | 64 |

## 3. Failure modes

### Coherent/incoherent binary agreement
| scope | agreement | N |
|---|---|---|
| overall | 0.952 | 600 |
| baseline | 0.983 | 60 |
| md_add_moderate | 0.990 | 100 |
| md_add_extreme | 0.983 | 120 |
| md_angular_extreme | 0.864 | 140 |
| probe | 1.000 | 80 |
| random | 0.940 | 100 |

### Directional check per stratum

| stratum | judge=literal N | →gold=fig | →gold=incoh | gold=literal N | judge missed (≠lit) |
|---|---|---|---|---|---|
| baseline | 10 | 0.20 | 0.10 | 8 | 0.12 |
| md_add_moderate | 39 | 0.00 | 0.00 | 40 | 0.03 |
| md_add_extreme | 49 | 0.00 | 0.00 | 52 | 0.06 |
| md_angular_extreme | 22 | 0.00 | 0.23 | 18 | 0.06 |
| probe | 14 | 0.14 | 0.00 | 12 | 0.00 |
| random | 22 | 0.09 | 0.05 | 20 | 0.05 |

### Top-10 highest-impact disagreements

_Impact = disagreements that flip the literal call (judge=literal→gold≠literal, or gold=literal→judge≠literal), since literal-rate is the headline metric._

| blind_id | model | stratum | gold | judge | prefix → completion excerpt |
|---|---|---|---|---|---|
| hCqV4x | gemma-2-2b | md_add_extreme | literal | figurative | …The special ink of the new decree was alone enough to make c →  smoke.  The decree, issued on Tuesday, says that the government has the right to regulate… |
| ibqXEg | Llama-3.1-8B | md_angular_extreme | literal | incoherent | …The early version was functional, but it still had rough edg → not so smooth edges. It was hard to see the whole picture at once, but it was easy to get … |
| f3KEwf | gemma-3-12b-pt | md_angular_extreme | incoherent | literal | …If the timing had been better, the moment might have left th → ”¦  In the wake of a devastating loss, the women of the “Big Bang Theory” cast are coming … |
| m4Ug4Y | gemma-3-12b-pt | md_angular_extreme | incoherent | literal | …The weary traveler stood at the crossroads → 211225.1337.899546.852176.852176.852176.852176.852… |
| hjlskO | gemma-3-4b-pt | md_add_extreme | literal | figurative | …Until the next phase begins, the initiative will remain on i →  and snow, the next phase of the project will be a new ice rink in the centre of the park,… |
| BFSOZs | Llama-3.1-8B | md_angular_extreme | incoherent | literal | …As the temperature dropped at the lakeside camp, the scout l →  doing so, he was able to keep the scouts warm, but they were able to keep the scouts warm… |
| XXcwEk | gemma-2-2b | md_angular_extreme | incoherent | literal | …We checked the tracking app and saw the delivery was current →  <strong>air</strong>.  <strong>Hämärä</strong>  <strong>Hämärä</strong>  <strong>Hämärä</… |
| 4jLSGe | gemma-3-12b-pt | md_add_extreme | literal | figurative | …After the scandal broke, the actor decided to lie low →  down and leave.  The scandal about the relationship of the 48-year-old Hollywood actor Le… |
| vDj7b1 | gemma-3-12b-pt | md_angular_extreme | incoherent | literal | …She had learned to be cautious, in case someone pulled her l → 1400  into a trap. Her heart was a little faster, and she looked  around her to see who mi… |
| yZgSot | gemma-3-4b-pt | probe | figurative | literal | …The manager spent his entire career working behind bars →  and is now the only person in the world who has been a prison officer and prison governor… |

## 4. Corrected literal rates at headline MD-additive operating points

_Method: at each model's MD-additive best-literal condition (max judge-literal rate over mean_diff additive conditions in the FULL judged file), apply the nearest audited MD-additive stratum's P(gold|judge) to the judged label distribution: corrected P(gold=lit)=Σ_j P(judge=j)·P(gold=lit|judge=j)._

| model | headline cond (α_factor) | audit stratum used | raw literal | corrected literal |
|---|---|---|---|---|
| meta-llama/Llama-3.1-8B | mean_diff_mult_-2.00 (-2.00) | md_add_moderate | 0.512 | 0.519 |
| google/gemma-2-2b | mean_diff_alpha_-150 (-1.56) | md_add_moderate | 0.574 | 0.582 |
| google/gemma-3-4b-pt | mean_diff_mult_-1.5 (-1.50) | md_add_moderate | 0.362 | 0.373 |
| google/gemma-3-12b-pt | mean_diff_mult_-0.5 (-0.50) | md_add_moderate | 0.342 | 0.354 |

## 5. Summary

The v14 judge generalizes **across the four held-out models** — overall accuracy 0.932 (κ=0.888, N=600), ranging from 0.920 (gemma-3-12b-pt) to 0.947 (Llama-3.1-8B). It also holds **across method families**: best probe (0.975), weakest **MD-angular (0.864, N=140)** — the caveat cell: MD-angular shows the largest judge–gold gap (see T3/T4), so literal-rate claims in that regime should use the corrected rates in §4. Because first_pass_label ≡ judge_label, the human 'correction rate' and 'judge disagreement rate' are one and the same measurement. Gold was jointly adjudicated (Appendix J), so no independent inter-annotator κ is reported for this batch.
