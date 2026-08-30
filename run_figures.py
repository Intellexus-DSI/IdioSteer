"""Regenerate the paper figures from committed aggregates in assets/ (CPU, no key).

Usage:
    python run_figures.py            # all four
    python run_figures.py --only layer_decodability
"""
import argparse, os, subprocess, sys
ROOT=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(ROOT,"figures")
JOBS={"layer_decodability":[["fig_layer_decodability.py","llama3b"],["fig_layer_decodability.py","gemma2_2b"]],
      "concreteness":[["fig_concreteness_per_idiom.py"]],
      "layer_sweep":[["fig4_layer_sweep_3panel.py"],["appM_layer_sweep_heatmap.py"]]}
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--only",choices=list(JOBS)); a=ap.parse_args()
    env={**os.environ,"IDIOSTEER_ROOT":ROOT}
    for k,jobs in JOBS.items():
        if a.only and k!=a.only: continue
        for j in jobs:
            print("+ figures/"+" ".join(j)); subprocess.run([sys.executable,os.path.join(FIG,j[0])]+j[1:],cwd=FIG,env=env,check=False)
if __name__=="__main__": main()
