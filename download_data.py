"""Fetch external data not redistributed in this repo.

This release already ships the derived data it needs:
  data/idiosteer/       - 100 IdioSteer prefixes (20 idioms)
  data/concreteness/     - v_BRY / v_GJ word lists + provenance
  data/idiolink_subset/  - retained IdioLink ids + counts + filter script

Two external sources are NOT redistributed (license/size):
  1. Brysbaert et al. (2014) concreteness norms  -> see data/concreteness/README.md for the citation + URL.
  2. Full IdioLink dataset (HuggingFace: Intellexus/IdioLink) -> optional, only for re-deriving the subset.

Usage:
    python download_data.py --idiolink     # pull full IdioLink from HuggingFace into data/idiolink_full/
"""
import argparse, os

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idiolink", action="store_true", help="download full IdioLink dataset from HuggingFace")
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    if args.idiolink:
        from huggingface_hub import snapshot_download
        out = os.path.join(root, "data", "idiolink_full")
        print(f"Downloading Intellexus/IdioLink -> {out}")
        snapshot_download(repo_id="Intellexus/IdioLink", repo_type="dataset", local_dir=out)
        print("done.")
    else:
        print(__doc__)
        print("Nothing to download for the default paper-reproduction path; data/ is already populated.")

if __name__ == "__main__":
    main()
