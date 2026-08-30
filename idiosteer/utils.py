"""Config + path helpers for IdioSteer."""
import os
try:
    import yaml
except ImportError:  # pyyaml optional at import time
    yaml = None

ROOT = os.environ.get("IDIOSTEER_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA    = os.path.join(ROOT, "data")
ASSETS  = os.path.join(ROOT, "assets")
RESULTS = os.path.join(ROOT, "results")

def load_config(path=None):
    """Load config.yaml (defaults to repo root)."""
    if yaml is None:
        raise ImportError("pyyaml required: pip install -r requirements.txt")
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

def get_key(name):
    """Resolve an API key: environment variable first, then keys.yaml. Never committed."""
    if os.environ.get(name):
        return os.environ[name]
    kp = os.path.join(ROOT, "keys.yaml")
    if yaml is not None and os.path.exists(kp):
        return (yaml.safe_load(open(kp)) or {}).get(name.lower())
    return None
