# IdioSteer — Sentence Construction & Validation Guidelines


## ISB Construction Guidelines

## Introduction
Hello, dear annotator, and thank you for your time helping us with our research!
Title: Idiom Steering Benchmark: Evaluating Activation Steering for Figurative vs. Literal Idiom Interpretation in LLMs
Core Research Question: Can activation steering methods shift a language model's default interpretation of idiomatic expressions from figurative to literal, and can this shift be reliably measured through open-ended text completion?

## Background
Idioms are multi-word expressions whose conventional meaning is non-compositional - that is, their figurative meaning cannot be derived from the literal meanings of their constituent words. For example, "spill the beans" conventionally means to reveal a secret, not to physically overturn a container of beans.
In this benchmark, each sentence ends with a target idiom and is designed to allow both figurative and literal interpretations. Your task is to validate that both interpretations are at least possible - not necessarily equally likely - for each sentence.

## Labels
For each sentence in the dataset, you will assign one of the following labels:
Valid (1) — At least one natural figurative completion AND at least one natural literal completion exist for the sentence. The sentence may lean toward one reading, but neither completion should require ignoring context or constructing an implausible scenario.
Invalid (0) — The sentence fails because no natural completion exists in at least one direction. Specifically if:
Every possible literal/figurative completion is forced or implausible.
The sentence is unnatural, grammatically awkward, or incoherent.
Important: A sentence is NOT invalid simply because it leans toward one reading. As long as a completion in the other direction exists and makes sense, the sentence is Valid (1).

## Examples

### Valid example — balanced:
Sentence: "The scientist put the project on ice"
Figurative completion: "...until the new funding cycle began." [natural]
Literal completion: "...the samples needed to be kept at near-freezing temperatures overnight." [natural]
Result: Both completions are natural. Label: Valid (1)

## Annotation Guidelines
Step 1 — Produce a figurative completion: Write a short, natural continuation interpreting the idiom figuratively.
Step 2 — Produce a literal completion: Write a short, natural continuation interpreting the idiom literally. If one reading feels forced, try alternatives (e.g., "take a bow" can refer to either a gesture or an archery bow).
Step 3 — Judge validity: Ask yourself: Does a natural completion exist in both directions?
Step 4 — If Invalid, suggest a fix: Revise the sentence to restore ambiguity (e.g., change the subject or setting).
Step 5 — Add a comment (optional): Flag as "figurative-cued", "literal-cued", or "borderline".

## Annotation Columns
Please fill in the following columns in the Google Sheet:
valid — 1 if valid, 0 if invalid.
correction — if invalid (0), your suggested revised sentence.
notes — free-style comments (e.g., "borderline").

## Additional Rules
Position: The target idiom must appear at the end of the sentence.
Bias: Do not penalize a sentence for leaning toward one reading.
Plausibility: A literal reading that requires a highly bizarre scenario does not count as valid.

### Is something off?
If a sentence is confusing, if you have a question about the guidelines, or if something seems wrong, please reach out directly rather than guessing. We would rather clarify than have uncertain annotations.
Contact: the IdioSteer maintainers (via the project GitHub repository)
Thank you very much for your contribution!
