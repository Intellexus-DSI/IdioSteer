"""LLM-as-judge over generated completions (OpenRouter; requires OPENROUTER_API_KEY).
Wraps idiosteer/judge.py.

Usage:
    python run_judge.py --input gen.json --output judged.json --max-workers 5
"""
import argparse, os, subprocess, sys
ROOT=os.path.dirname(os.path.abspath(__file__))
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",required=True); ap.add_argument("--output",required=True); ap.add_argument("--max-workers",default="5")
    a=ap.parse_args()
    subprocess.run([sys.executable,os.path.join(ROOT,"idiosteer","judge.py"),
                    "--input",a.input,"--output",a.output,"--max-workers",str(a.max_workers)],
                   cwd=ROOT,env={**os.environ,"IDIOSTEER_ROOT":ROOT},check=False)
if __name__=="__main__": main()
