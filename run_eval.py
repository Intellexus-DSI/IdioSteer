"""Generate steered completions for the IdioSteer prefixes (GPU; requires HF_TOKEN).
Wraps analysis/run_generation_conditions.py.

Usage:
    python run_eval.py --vector v_BRY
"""
import argparse, os, subprocess, sys
ROOT=os.path.dirname(os.path.abspath(__file__))
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--vector",default="v_BRY"); a=ap.parse_args()
    subprocess.run([sys.executable,os.path.join(ROOT,"analysis","run_generation_conditions.py"),a.vector],
                   cwd=ROOT,env={**os.environ,"IDIOSTEER_ROOT":ROOT},check=False)
if __name__=="__main__": main()
