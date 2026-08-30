# IdioSteer prefixes

100 sentence prefixes across 20 idioms, each ending in a target idiom and validated to admit
BOTH a natural figurative and a natural literal continuation (dual-completion validation).

Files:
- `idiosteer_prefixes.csv` / `.jsonl` : item_id, idiom, variant, prefix, idiom_char_start,
  idiom_char_end, idiom_surface (character offsets of the target idiom span within `prefix`).
- `CONSTRUCTION_GUIDELINES.md` : how prefixes were written and validated (the dual-completion protocol).

Completion annotation guidelines (how model completions were labelled figurative/literal/incoherent)
are in `docs/ANNOTATION.md`. Per the paper's Ethics Statement, raw model completions are NOT released.
