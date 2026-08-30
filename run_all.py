"""Reproduce the paper from committed aggregates (the cheap path), and document the expensive steps.

    python run_all.py            # regenerate all figures/tables from assets/ (CPU, seconds)

The full generation grid (~106 GPU-h) and judge pass (~$0.66/2k items) are NOT run here; see docs/REPRODUCE.md.
"""
import os, subprocess, sys
ROOT=os.path.dirname(os.path.abspath(__file__))
def main():
    print("[1/1] Regenerating figures from assets/ ...")
    subprocess.run([sys.executable,os.path.join(ROOT,"run_figures.py")],check=False)
    print("\nDone. Table numbers are in assets/p1-p9 and results/. Full regeneration: docs/REPRODUCE.md.")
if __name__=="__main__": main()
