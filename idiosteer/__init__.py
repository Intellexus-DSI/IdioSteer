"""IdioSteer: activation steering for figurative-vs-literal idiom interpretation (EMNLP 2026).

Lightweight top-level import (no torch). For steering internals:
    from idiosteer.pipeline import MODELS, load_model_and_tokenizer
    from idiosteer.hooks import AdditiveSteeringHook, AngularSteeringHook
    from idiosteer.vectors import build_md_vector
    from idiosteer.judge import derive_label
"""
from .utils import load_config, get_key, ROOT, DATA, ASSETS, RESULTS
__all__ = ["load_config", "get_key", "ROOT", "DATA", "ASSETS", "RESULTS"]
