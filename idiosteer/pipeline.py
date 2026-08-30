"""
concreteness/pipeline.py
========================
Shared, REUSED activation-extraction + MD-construction logic for the
concreteness-direction experiments.

The extraction and mean-difference logic below is copied VERBATIM (semantics
preserved) from the canonical idiom pipeline:
    three_methods_evaluation_llama8b.py
        find_last_idiom_pos()          -> :123
        activation extraction loop     -> :155-166  (out.hidden_states[Ls][0,pos,:].cpu().float().numpy())
        MD construction                -> :176-178  (fig.mean(0)-lit.mean(0); raw_norm; /(raw_norm+1e-8))
We do NOT reimplement extraction; we only wrap the identical operations in
callables that take (model, tokenizer, layer) so they can be reused for target
NOUNS (concreteness) as well as idiom spans.
"""
import os
import numpy as np
import torch

# ── Per-model registry: HF id + canonical source layer Ls (hidden_states index) ──
# Ls values are the user-specified canonical layers (HF output_hidden_states
# convention, hidden_states[0] = embeddings). Llama-3.2-3B fixed to 14 per spec
# (the original script picked it dynamically; we hard-fix for consistency).
MODELS = {
    "llama3b":    {"hf": "meta-llama/Llama-3.2-3B", "Ls": 14},
    "llama8b":    {"hf": "meta-llama/Llama-3.1-8B", "Ls": 16},
    "gemma2_2b":  {"hf": "google/gemma-2-2b",       "Ls": 20},
    "gemma3_4b":  {"hf": "google/gemma-3-4b-pt",    "Ls": 17},
    "gemma3_12b": {"hf": "google/gemma-3-12b-pt",   "Ls": 24},
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(hf_id):
    """Load fp16 model + tokenizer exactly as the idiom pipeline does."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    return model, tok


# ── REUSED verbatim from three_methods_evaluation_llama8b.py:123 ────────────
def find_last_span_pos(tokenizer, sentence, target):
    """Return (input_ids, index_of_last_token_overlapping `target`) or (ids,None)/(None,None).
    Identical to find_last_idiom_pos() — locates the FIRST occurrence of `target`
    (case-insensitive) and returns the last token whose char-span overlaps it."""
    enc = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    idx = sentence.lower().find(target.lower())
    if idx == -1:
        return None, None
    end = idx + len(target)
    last = None
    for i, (s, e) in enumerate(offsets):
        if s < end and e > idx:
            last = i
    return ids, last


def last_token_at_charspan(tokenizer, sentence, idx, end):
    """Same last-token-overlap logic as find_last_span_pos, but given an EXACT
    char span [idx,end). Used for fixed frames where we insert the target word at
    a known position, avoiding substring false-matches (e.g. 'let' inside 'letter')."""
    enc = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    last = None
    for i, (s, e) in enumerate(offsets):
        if s < end and e > idx:
            last = i
    return ids, last


# ── REUSED verbatim from three_methods_evaluation_llama8b.py:155-166 ─────────
@torch.no_grad()
def collect_activations(model, tokenizer, samples, source_layer, log_every=500, tag=""):
    """samples: list of dicts each with 'ids' (list[int]) and 'pos' (int).
    Returns np.ndarray [n, d] of fp32 residual-stream vectors at hidden_states[source_layer], token `pos`."""
    acts = []
    for i, s in enumerate(samples):
        out = model(torch.tensor([s["ids"]], device=DEVICE), output_hidden_states=True)
        acts.append(out.hidden_states[source_layer][0, s["pos"], :].cpu().float().numpy())
        if log_every and (i + 1) % log_every == 0:
            print(f"    [{tag}] {i+1}/{len(samples)}", flush=True)
    return np.stack(acts)


# ── REUSED verbatim from three_methods_evaluation_llama8b.py:176-178 ─────────
def build_md_vector(pos_acts, neg_acts):
    """Class-mean difference (pos - neg), fp32, unit-normalized.
    Positive pole = the `pos_acts` class. Returns (unit_vec, raw_norm)."""
    mean_diff = pos_acts.mean(0) - neg_acts.mean(0)
    raw_norm = float(np.linalg.norm(mean_diff))
    unit = mean_diff / (raw_norm + 1e-8)
    return unit.astype(np.float32), raw_norm


# ── Data helpers ─────────────────────────────────────────────────────────────
REPO = os.path.expanduser(".")
IDIOLINK = os.path.join(REPO, "data_outputs", "idiolink_interpretability.csv")

# The 3 neutral frames (Stage 2c). {W} is the target-word slot.
FRAMES = [
    "She thought about the {W} for a while.",
    "They discussed the {W} during the meeting.",
    "He mentioned the {W} in his letter.",
]


def build_idiom_samples(tokenizer):
    """v_IDIOM source: IdioLink train split. Reuses find_last_span_pos with the
    `idiom` column (exactly as the canonical pipeline). Returns (samples, labels, n_drop).
    label 1 = figurative (positive pole), 0 = literal."""
    import pandas as pd
    df = pd.read_csv(IDIOLINK)
    train = df[df["split"] == "train"].copy()
    samples, labels, drop = [], [], 0
    for _, row in train.iterrows():
        ids, pos = find_last_span_pos(tokenizer, row["sentence"], row["idiom"])
        if ids is not None and pos is not None:
            samples.append({"ids": ids, "pos": pos})
            labels.append(1 if row["type"] == "figurative" else 0)
        else:
            drop += 1
    return samples, np.array(labels), drop


def build_frame_samples(tokenizer, words):
    """Put each word through all 3 frames; locate the word's final token at its
    exact insertion char-span. Returns (samples, meta, n_ambiguous_dropped).
    A sample is dropped+logged if the inserted word ALSO appears elsewhere in the
    frame (ambiguous) or tokenization yields no overlapping token."""
    samples, meta, ambiguous = [], [], 0
    for w in words:
        for fi, frame in enumerate(FRAMES):
            prefix = frame.split("{W}")[0]
            sentence = frame.format(W=w)
            idx = len(prefix)
            end = idx + len(w)
            # ambiguity guard: the word string appears more than once in the sentence
            if sentence.lower().count(w.lower()) > 1:
                ambiguous += 1
                continue
            ids, pos = last_token_at_charspan(tokenizer, sentence, idx, end)
            if ids is None or pos is None:
                ambiguous += 1
                continue
            samples.append({"ids": ids, "pos": pos})
            meta.append({"word": w, "frame": fi, "sentence": sentence})
    return samples, meta, ambiguous
