# Concreteness word lists & templates

Two constructions of a concreteness contrast, both from the Brysbaert et al. (2014) norms.

- `concrete_250.txt` / `abstract_250.txt` : v_BRY lists. Nouns only (Brysbaert Dom_Pos == Noun),
  concreteness >= 4.5 (concrete) / <= 2.0 (abstract), <=4 subword pieces in all five model tokenizers,
  250 sampled per pool (seed 42).
- `gj_concrete_500.txt` / `gj_abstract_500.txt` : v_GJ recipe (Ghosh & Jiang). >= 4.0 / <= 2.0,
  NO POS filter, cap 500 per pool (seed 42).
- `templates.txt` : the 3 neutral carrier templates ({WORD} slot).
- `provenance.json` : exact pool sizes, coverage drops, thresholds, seed.

Brysbaert norms are NOT redistributed here; download from the source and cite:
  Brysbaert, Warriner & Kuperman (2014), Behavior Research Methods 46(3):904-911.
  https://doi.org/10.3758/s13428-013-0403-5

Note: the paper also reports a *deprecated* framed-v_GJ construction (GJ words inserted into the 3
templates). An audit found the framed abstract pool ~83% non-noun (ungrammatical carriers); the
released geometry uses the **bare-word** v_GJ (each word forwarded alone). See the paper's concreteness appendix.
