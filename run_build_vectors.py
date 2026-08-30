"""Build steering vectors for one model (GPU; requires HF_TOKEN). Wraps analysis/build_concreteness_vectors.py.

Usage:
    python run_build_vectors.py --model llama3b
"""
import argparse, os, subprocess, sys
ROOT=os.path.dirname(os.path.abspath(__file__))
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--model",default="llama3b"); a=ap.parse_args()
    subprocess.run([sys.executable,os.path.join(ROOT,"analysis","build_concreteness_vectors.py"),a.model],
                   cwd=ROOT,env={**os.environ,"IDIOSTEER_ROOT":ROOT},check=False)
if __name__=="__main__": main()
