"""Select the IdioLink rows used in this paper, given the original IdioLink CSV.
Usage: python filter_idiolink.py /path/to/original_idiolink.csv
Writes idiolink_used.csv containing only the retained ids."""
import sys, pandas as pd, os
ids = set(pd.read_csv(os.path.join(os.path.dirname(__file__), "retained_ids.csv"))["id"].astype(str))
df = pd.read_csv(sys.argv[1]); df = df[df["id"].astype(str).isin(ids)]
df.to_csv("idiolink_used.csv", index=False)
print(f"retained {len(df)} rows -> idiolink_used.csv")
