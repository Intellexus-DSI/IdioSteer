# IdioLink subset (IDs only — IdioLink itself is NOT redistributed)

We do not redistribute IdioLink. Obtain it from the original release, then use
`retained_ids.csv` to select the rows we used.

- `retained_ids.csv` : id, idiom, type (figurative/literal), subject, split (train/val/test).
- `counts_per_idiom.csv`, `counts_per_category.csv` : per-idiom and per-(subject,type) counts.
- `filter_idiolink.py` : reproduces our selection from the original IdioLink CSV.

Original IdioLink: <link to the official IdioLink release>.
