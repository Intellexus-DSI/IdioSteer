#!/usr/bin/env python3
"""LLM-as-a-Judge for idiom completions — v14 (clean, parallel).

Runs the v14 (`v14_coherence_strict`) prompt against `google/gemma-4-31b-it`
via OpenRouter for every (idiom, completion) pair in a generation results
file and writes an enriched JSON with the judge's structured output and a
derived label per completion.

Designed primary input: the `three_methods_eval_*.json` schema — top-level
`{ "metadata": {...}, "results": [ { "idiom", "variant", "sentence",
"completions": { "<condition>": ["<completion_string>", ... 5x ] } }, ... ] }`.

Within each (idiom, variant, condition) the 5 completions are submitted to the
API in parallel via a ThreadPoolExecutor (default 5 workers).

Output schema per completion (v14):
    {
        "label": "figurative" | "literal" | "incoherent",          # derived
        "is_coherent": bool,
        "literal_better_supported": bool,
        "figurative_meaning": str,
        "literal_meaning": str,
        "coherence_reasoning": str,
        "figurative_reading_reasoning": str,
        "literal_reading_reasoning": str,
        "comparison_reasoning": str,
        "confidence": "high" | "medium" | "low",
        "raw": str,            # raw model response (only on parse failure)
        "error": str | None,   # set if the call failed permanently
    }

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python annotations/judge_v14.py \\
        --input  data_outputs/idiom_evaluations/three_methods_eval_validated_latest.json \\
        --output data_outputs/idiom_evaluations/judge_v14/judged_<timestamp>.json \\
        --max-workers 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import requests
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it"

# Pricing in USD per 1M tokens (input / output) — used only for cost estimate.
MODEL_PRICING = {
    "google/gemma-4-31b-it": {"input": 0.14, "output": 0.39},
}


# =============================================================================
# V14 PROMPT
# =============================================================================

V14_SYSTEM_PROMPT = """You are an expert linguist analyzing idiom usage in model-generated text.

TASK:
Given an IDIOM and a FULL_TEXT, analyze the text and produce structured judgments. You do NOT need to produce a single final label — we will derive it from your boolean judgments.

SCOPE OF ANALYSIS:
- Focus primarily on the FIRST SENTENCE containing the idiom.
- If the first sentence alone is enough to decide, stop there.
- Only read beyond the first sentence when you need more context to disambiguate.
- Truncation at the end due to generation token limits is NOT incoherence.
- Repetition of the same sentence after the first one is a generation artifact — IGNORE IT when judging coherence.

CRITICAL DISTINCTION — FIGURATIVE vs LITERAL:
Figurative and literal refer to whether the IDIOM is used in its IDIOMATIC sense, NOT whether the described action is physical or abstract.
- FIGURATIVE means the idiom is used AS AN IDIOM — with its conventional, non-compositional meaning. Even if the idiomatic meaning itself involves a physical act (e.g., "take a bow" meaning to receive applause), it is still a FIGURATIVE use of the idiom.
- LITERAL means the individual words of the phrase are used with their compositional, word-by-word meaning OUTSIDE the idiomatic frame.

STEPS — follow in order:

STEP A — COHERENCE CHECK (reading-INDEPENDENT, grammar-STRICT):
Judge coherence as if the idiom's words were ordinary words in the language, WITHOUT regard to whether they form a standard idiomatic phrase. A sentence is coherent only if it is GRAMMATICAL and describes a plausible or imaginable situation under AT LEAST ONE reading (figurative OR literal).

Principle 1 — Reading-independent:
- DO NOT mark a sentence incoherent merely because the idiom's fixed form has been modified, extended, or re-purposed. Legitimate extensions of the idiom — inserting additional words, attaching modifiers, or re-using the words as a compositional/literal compound — are NOT coherence defects. They are a normal signal of LITERAL or creative usage, to be weighed later in STEP E.
- Do NOT require the phrase to preserve its canonical idiomatic structure. Treat the phrase's words as ordinary words when judging grammar and meaning.
- Pragmatic oddness (an unusual or absurd-but-imaginable scenario) is NOT incoherence.

Principle 2 — Grammaticality is required (strict):
Do NOT give benefit of the doubt to broken grammar just because you can guess the intended meaning. A sentence IS incoherent if it contains any of the following syntactic defects in the portion involving the idiom:
- A required argument is missing (e.g., a transitive verb or transitive preposition used without its object; a possessive whose head noun makes no sense; a pronoun with no reasonable antecedent).
- A word is attached to the idiom in a way that yields no valid English constituent (e.g., a dangling or stranded preposition/particle with no object or complement; a prepositional phrase whose attachment is syntactically impossible).
- The phrase contains invented non-words or malformed compounds that are not valid English words and cannot be parsed as a clean sequence of valid English words.
- A possessive or determiner is attached to a word that cannot take it under the intended structure.

Principle 3 — Do not "repair" the sentence:
- Coherence is judged on the sentence AS WRITTEN. Do not silently drop, add, or re-order words to make it work.
- "Awkward but parseable" is coherent. "Ungrammatical but guessable" is NOT coherent.

Mark "is_coherent" = false if the first sentence exhibits any of the syntactic defects above, OR if it fails to describe a plausible situation under BOTH the figurative AND the literal reading. Otherwise mark "is_coherent" = true.

Produce "is_coherent" (true | false) and a brief "coherence_reasoning". The reasoning MUST state both (a) whether the sentence is grammatical as written (and if not, which specific constituent is broken), and (b) whether at least one reading (figurative or literal) yields a plausible proposition.

STEP B — IDIOM UNDERSTANDING:
State your understanding of the idiom:
- "figurative_meaning": the conventional idiomatic sense (a short paraphrase).
- "literal_meaning": the compositional, word-by-word meaning outside the idiomatic frame (a short paraphrase).

STEP C — ANALYZE THE FIGURATIVE READING:
Read the text assuming the idiom is used in its FIGURATIVE (idiomatic) sense. Briefly describe:
- What the text would mean under the figurative reading.
- Which specific cues in the text support OR contradict this reading.
Produce: "figurative_reading_reasoning" (brief explanation with cues).

STEP D — ANALYZE THE LITERAL READING:
Read the text assuming the idiom is used in its LITERAL (compositional, word-by-word) sense. Briefly describe:
- What the text would mean under the literal reading.
- Which specific cues in the text support OR contradict this reading.
Produce: "literal_reading_reasoning" (brief explanation with cues).

STEP E — COMPARE THE TWO READINGS:
Directly compare the two readings and ask yourself:
"Is the LITERAL reading CLEARLY BETTER SUPPORTED by the text than the FIGURATIVE reading?"
- Set "literal_better_supported" = true ONLY if the literal reading is clearly and substantially more consistent with the text than the figurative reading. This requires specific cues in the text that force the literal reading and/or make the figurative reading implausible.
- If both readings are roughly equally supported, OR if the figurative reading is equally plausible or better supported, set "literal_better_supported" = false.
- The figurative (idiomatic) reading is the DEFAULT — only flip this boolean to true when the evidence clearly favors the literal reading.

Also provide:
- "comparison_reasoning": a brief explanation of which reading is better supported and why. Cite the decisive cues.
- "confidence": "high" | "medium" | "low" — how confident you are in the comparison.

Return ONLY a JSON object:
{"figurative_meaning": "short paraphrase", "literal_meaning": "short paraphrase", "is_coherent": true | false, "coherence_reasoning": "brief explanation", "figurative_reading_reasoning": "brief explanation with cues", "literal_reading_reasoning": "brief explanation with cues", "literal_better_supported": true | false, "comparison_reasoning": "brief explanation of which reading wins and why", "confidence": "high" | "medium" | "low"}"""


def build_messages(idiom: str, full_text: str) -> list[dict]:
    return [
        {"role": "system", "content": V14_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f'IDIOM: "{idiom}"\nFULL_TEXT: "{full_text}"',
        },
    ]


# =============================================================================
# TOKEN USAGE TRACKER (thread-safe)
# =============================================================================

@dataclass
class UsageTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    num_calls: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def add(self, usage: dict | None) -> None:
        if not usage:
            return
        with self._lock:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            self.num_calls += 1

    def cost(self, model: str) -> dict:
        rates = MODEL_PRICING.get(model, {"input": 1.0, "output": 3.0})
        ic = self.prompt_tokens / 1_000_000 * rates["input"]
        oc = self.completion_tokens / 1_000_000 * rates["output"]
        return {"input_cost_usd": round(ic, 4), "output_cost_usd": round(oc, 4),
                "total_cost_usd": round(ic + oc, 4)}

    def summary(self, model: str) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "num_calls": self.num_calls,
            **self.cost(model),
        }


# =============================================================================
# OPENROUTER CLIENT (with retries)
# =============================================================================

class OpenRouterClient:
    def __init__(self, api_key: str, model: str, max_retries: int = 4,
                 base_backoff: float = 2.0,
                 connect_timeout: float = 10.0, read_timeout: float = 45.0):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        # (connect, read) tuple — read timeout is "max seconds between bytes
        # received". Single-int timeouts have been observed to hang on stale
        # TLS sockets returned by urllib3's connection pool; the explicit tuple
        # plus `Connection: close` (below) avoids pool reuse and that bug.
        self.timeout = (connect_timeout, read_timeout)

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 700) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://msc-thesis-idiom-eval.local",
            "X-Title": "Idiom Judge v14",
            # Disable HTTP keep-alive: each call gets a fresh TCP+TLS socket.
            # Trades ~50ms per call for total robustness against stale sockets.
            "Connection": "close",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(OPENROUTER_API_URL, headers=headers,
                                  json=payload, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    wait = self.base_backoff ** attempt + 1
                    time.sleep(wait)
                    last_error = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                last_error = e
                time.sleep(self.base_backoff ** attempt)
            except requests.exceptions.HTTPError as e:
                # 4xx (other than 429) are not retryable
                if e.response is not None and 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise
                last_error = e
                time.sleep(self.base_backoff ** attempt)

        raise last_error or RuntimeError("Max retries exceeded")


# =============================================================================
# PARSING & LABEL DERIVATION
# =============================================================================

def _extract_json(content: str) -> dict | None:
    """Extract the first valid JSON object from a possibly-wrapped string."""
    s = content.strip()
    m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", s)
    if m:
        s = m.group(1).strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def derive_label(is_coherent: Any, literal_better: Any) -> str:
    """v14 label derivation:
        is_coherent == False        -> incoherent
        literal_better_supported    -> literal
        else                        -> figurative
    """
    if is_coherent is False:
        return "incoherent"
    if literal_better is True:
        return "literal"
    return "figurative"


def parse_v14(content: str) -> dict:
    data = _extract_json(content)
    if data is None:
        return {
            "label": "figurative",
            "is_coherent": None,
            "literal_better_supported": None,
            "figurative_meaning": "",
            "literal_meaning": "",
            "coherence_reasoning": "",
            "figurative_reading_reasoning": "",
            "literal_reading_reasoning": "",
            "comparison_reasoning": "",
            "confidence": "low",
            "raw": content[:1000],
            "parse_error": True,
        }

    is_coherent = data.get("is_coherent")
    literal_better = data.get("literal_better_supported")
    return {
        "label": derive_label(is_coherent, literal_better),
        "is_coherent": is_coherent,
        "literal_better_supported": literal_better,
        "figurative_meaning": data.get("figurative_meaning", ""),
        "literal_meaning": data.get("literal_meaning", ""),
        "coherence_reasoning": data.get("coherence_reasoning", ""),
        "figurative_reading_reasoning": data.get("figurative_reading_reasoning", ""),
        "literal_reading_reasoning": data.get("literal_reading_reasoning", ""),
        "comparison_reasoning": data.get("comparison_reasoning", ""),
        "confidence": data.get("confidence", "medium"),
        "parse_error": False,
    }


# =============================================================================
# SINGLE JUDGEMENT
# =============================================================================

def judge_one(client: OpenRouterClient, idiom: str, full_text: str,
              usage: UsageTracker) -> dict:
    """Run the v14 judge on one (idiom, full_text) pair. Never raises;
    on permanent failure returns a dict with `error` set."""
    try:
        resp = client.chat(build_messages(idiom, full_text))
    except Exception as e:
        return {
            "label": "figurative",
            "is_coherent": None,
            "literal_better_supported": None,
            "figurative_meaning": "",
            "literal_meaning": "",
            "coherence_reasoning": "",
            "figurative_reading_reasoning": "",
            "literal_reading_reasoning": "",
            "comparison_reasoning": "",
            "confidence": "low",
            "parse_error": False,
            "error": f"{type(e).__name__}: {e}",
        }
    usage.add(resp.get("usage"))
    content = resp["choices"][0]["message"]["content"]
    out = parse_v14(content)
    out["error"] = None
    return out


# =============================================================================
# DATASET ITERATION (three_methods_eval format)
# =============================================================================

@dataclass
class WorkItem:
    key: str           # stable id used for checkpointing
    idiom: str
    full_text: str
    # back-references for writing results back into the input structure:
    result_idx: int    # index into top-level "results"
    condition: str
    completion_idx: int


def iter_three_methods(data: dict) -> list[WorkItem]:
    items: list[WorkItem] = []
    for r_idx, r in enumerate(data.get("results", [])):
        idiom = r["idiom"]
        prefix = r["sentence"]
        completions = r.get("completions", {})
        variant = r.get("variant", 0)
        for cond, comps in completions.items():
            for c_idx, comp in enumerate(comps):
                full_text = (prefix or "") + (comp or "")
                key = f"{idiom}||v{variant}||{cond}||c{c_idx}"
                items.append(WorkItem(
                    key=key, idiom=idiom, full_text=full_text,
                    result_idx=r_idx, condition=cond, completion_idx=c_idx,
                ))
    return items


# =============================================================================
# CHECKPOINTING
# =============================================================================

def load_checkpoint(path: Path) -> dict:
    if path.exists():
        try:
            return json.load(open(path))
        except Exception:
            print(f"  Warning: could not read checkpoint at {path}; starting fresh")
    return {}


def save_checkpoint(path: Path, judgments: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    json.dump(judgments, open(tmp, "w"), ensure_ascii=False)
    tmp.replace(path)


# =============================================================================
# PARALLEL JUDGE
# =============================================================================

def run_parallel(
    client: OpenRouterClient,
    items: list[WorkItem],
    usage: UsageTracker,
    max_workers: int,
    checkpoint_path: Path | None,
    checkpoint_every: int = 25,
) -> dict[str, dict]:
    """Submit all items to a thread pool and return a dict {key: judgment}.

    Resumes from the checkpoint file if any items are already done.
    """
    judgments: dict[str, dict] = {}
    if checkpoint_path is not None:
        judgments = load_checkpoint(checkpoint_path)
        if judgments:
            print(f"  Resuming from checkpoint: {len(judgments)} items already done")

    todo = [it for it in items if it.key not in judgments]
    if not todo:
        return judgments

    completed_since_save = 0
    save_lock = Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(judge_one, client, it.idiom, it.full_text, usage): it
                   for it in todo}
        with tqdm(total=len(todo), desc=f"Judging (w={max_workers})") as bar:
            for fut in as_completed(futures):
                it = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"label": "figurative", "error": f"{type(e).__name__}: {e}",
                              "is_coherent": None, "literal_better_supported": None,
                              "confidence": "low", "parse_error": False}
                judgments[it.key] = result
                bar.update(1)

                if checkpoint_path is not None:
                    with save_lock:
                        completed_since_save += 1
                        if completed_since_save >= checkpoint_every:
                            save_checkpoint(checkpoint_path, judgments)
                            completed_since_save = 0

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, judgments)

    return judgments


# =============================================================================
# OUTPUT ASSEMBLY
# =============================================================================

def attach_judgments(input_data: dict, items: list[WorkItem],
                     judgments: dict[str, dict]) -> dict:
    """Return a copy of input_data with `judge_v14` attached to each result.

    Structure:
        results[i].judge_v14 = { "<condition>": [judgement_dict, ...5x] }
    """
    out = json.loads(json.dumps(input_data))  # deep copy
    # initialise judge_v14 mirror of completions structure
    for r in out.get("results", []):
        comps = r.get("completions", {})
        r["judge_v14"] = {cond: [None] * len(lst) for cond, lst in comps.items()}

    for it in items:
        j = judgments.get(it.key)
        if j is None:
            continue
        out["results"][it.result_idx]["judge_v14"][it.condition][it.completion_idx] = j

    return out


def aggregate_label_stats(judgments: dict[str, dict]) -> dict:
    counts = {"figurative": 0, "literal": 0, "incoherent": 0, "errors": 0}
    for j in judgments.values():
        if j.get("error"):
            counts["errors"] += 1
            continue
        counts[j.get("label", "figurative")] = counts.get(j.get("label", "figurative"), 0) + 1
    total = sum(counts.values())
    rates = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return {"counts": counts, "rates": rates, "total": total}


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="LLM-as-a-Judge (v14) — clean, parallel runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", "-i", type=Path, required=True,
                    help="Path to a three_methods_eval-style JSON file with "
                         "`results[*].completions[<condition>] = [str, ...]`")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Output JSON path (default: <input_dir>/judge_v14/judged_<stem>_<ts>.json)")
    ap.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                    help=f"OpenRouter model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--max-workers", "-w", type=int, default=5,
                    help="Concurrent in-flight requests (default: 5; one per completion in a 5-completion batch)")
    ap.add_argument("--limit", "-n", type=int, default=None,
                    help="Optional: judge only the first N work items (for sanity checks)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Checkpoint path (default: <output_dir>/.checkpoint_<stem>.json)")
    ap.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing")
    ap.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (or env OPENROUTER_API_KEY)")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set. Pass --api-key or export the env var.")

    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")
    print(f"Loading input: {args.input}")
    input_data = json.load(open(args.input))

    items = iter_three_methods(input_data)
    print(f"Found {len(items)} (idiom, completion) pairs to judge")
    if args.limit is not None:
        items = items[: args.limit]
        print(f"Limiting to first {len(items)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        out_dir = args.input.parent / "judge_v14"
        args.output = out_dir / f"judged_{args.input.stem}_{ts}.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.no_checkpoint:
        cp_path = None
    elif args.checkpoint is not None:
        cp_path = args.checkpoint
    else:
        cp_path = args.output.parent / f".checkpoint_{args.input.stem}.json"
    if cp_path:
        print(f"Checkpoint: {cp_path}")
    print(f"Output:     {args.output}")
    print(f"Model:      {args.model}    workers: {args.max_workers}")
    print()

    client = OpenRouterClient(api_key=api_key, model=args.model)
    usage = UsageTracker()

    t0 = time.time()
    judgments = run_parallel(
        client=client, items=items, usage=usage,
        max_workers=args.max_workers, checkpoint_path=cp_path,
    )
    elapsed = time.time() - t0

    enriched = attach_judgments(input_data, items, judgments)
    stats = aggregate_label_stats(judgments)
    enriched.setdefault("metadata", {})
    enriched["metadata"]["judge_v14"] = {
        "model": args.model,
        "prompt_version": "v14_coherence_strict",
        "max_workers": args.max_workers,
        "n_items": len(items),
        "elapsed_seconds": round(elapsed, 1),
        "label_stats": stats,
        "usage": usage.summary(args.model),
        "timestamp": ts,
        "input_file": str(args.input.resolve()),
    }

    json.dump(enriched, open(args.output, "w"), ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Items judged: {len(judgments)} / {len(items)}")
    print(f"  Label distribution:")
    for k, v in stats["counts"].items():
        print(f"    {k:11s}: {v:5d}  ({stats['rates'][k]*100:.1f}%)")
    cost = usage.cost(args.model)
    print(f"  Tokens: prompt={usage.prompt_tokens:,}  completion={usage.completion_tokens:,}  total={usage.total_tokens:,}")
    print(f"  Cost:   ${cost['total_cost_usd']:.4f}")
    print(f"  Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
