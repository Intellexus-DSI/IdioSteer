# IdioSteer — Completion Annotation Guidelines (judge & human, Appendix J protocol)


## ISB Completion Annotation Guidelines

## Introduction
Hello, dear annotator, and thank you for your time helping us with our research!
Title: Idiom Steering Benchmark: Evaluating Activation Steering for Figurative vs. Literal Idiom Interpretation in LLMs
Core Research Question: Can activation steering methods shift a language model's default interpretation of idiomatic expressions from figurative to literal, and can this shift be reliably measured through open-ended text completion?

## Background
In this study, a language model was given sentences ending with a target idiom and asked to complete them. Each completion reveals whether the model interpreted the idiom figuratively (in its conventional meaning) or literally (based on the surface meanings of its words). 
Your task is to classify how the idiom is used in the context of the sentence.
For example, for the sentence: "The old man kicked the bucket", the model might complete it as:
"...peacefully in his sleep, surrounded by his loving family after a long and full life."figurative (died)
"...across the yard in a fit of frustration when he realized he had forgotten to fill it with water for the garden."literal (physical bucket)
Your task is to read each sentence and classify how the idiom was interpreted. This annotation is central to evaluating whether our steering methods successfully shifted the model's behavior.
The dataset contains 280 completions across 20 idioms. Each row will be annotated by two annotators and LLM independently to measure inter-annotator agreement.

## Labels
Assign exactly one of the following labels for each row:
figurative — The idiom is used with its conventional figurative meaning.
literal — The idiom is used literally, word-by-word meaning outside the idiomatic frame.
incoherent — The text is syntactically broken or fails to describe an imaginable situation.

## Annotation Steps

### Step 1 — Coherence Check (Grammar-Strict)

### Before looking for meaning, judge if the sentence is coherent.
Reading-Independent: Judge coherence as if the idiom's words were ordinary words. Do not penalize if the idiom’s form is modified or extended (e.g., "kicked the large bucket").
Grammaticality is Required: A sentence is incoherent if it contains syntactic defects such as:
Missing required arguments (e.g., a verb with no object).
Broken constituents (e.g., a dangling preposition like "the cake was of").
Invented non-words or malformed compounds.
Do Not "Repair": Judge the sentence exactly as written. "Awkward but parseable" is coherent. "Ungrammatical but guessable" is incoherent.
Action: If the first sentence exhibits these defects, label as incoherent and move to the next row.

### Step 2 — Understand the Idiom
Define the two potential paths:
Figurative Meaning: The conventional idiomatic sense (e.g., "kick the bucket" = to die).
Literal Meaning: The compositional, word-by-word meaning (e.g., "kick the bucket" = physically striking a pail).

### Step 3 — Analyze Both Readings

### Read the text twice:
Assume Figurative: Does the text contain cues (context, tone) that support the idiomatic sense?
Assume Literal: Does the text contain physical cues or specific details that force the word-by-word sense?

### Step 4 — Compare and Label

### Ask: "Is the literal reading clearly better supported by the text than the figurative reading?"
Label literal: Only if there are specific cues that make the figurative reading implausible.
Label figurative: This is the default. If both readings are equally plausible, or if the figurative reading is better supported, choose figurative.

### Important Principles
First Sentence Priority: Focus primarily on the first sentence containing the idiom. If that sentence alone is enough to decide, stop there.
Ignore Generation Artifacts: Truncation at the end of a block is not incoherence. Repetition of the same sentence after the first one is an artifact—ignore it when judging coherence.
The Figurative Default: The figurative (idiomatic) reading is the default. Only label as literal if the text provides clear cues that favor the literal meaning over the figurative one.

## Examples

### idiomatic example:
Idiom: piece of cake
Sentence: "I was nervous about the coding certification, but after all the studying I did, the exam was a piece of cake. I finished it with thirty minutes to spare."
Classification: figurative (means very easy).

### literal example:
Idiom: piece of cake
Sentence: "After the birthday girl blew out the candles, the host handed me a piece of cake on a small paper plate."
Classification: literal (physical cake).

### incoherent example:
Idiom: piece of cake
Completion: "The exam was a piece of cake cake cake..."
Classification: incoherent (the completion degenerates into repetition after the first clause and is not interpretable as a sentence).

## Annotation Columns
label — Choose one: figurative, literal, incoherent.
notes — (Optional) Record evidence, e.g., "substitution test passed" or "mentions physical ice".

### Notes & Rules
Labels: Choose exactly one: figurative, literal, or incoherent.
Notes Column: Briefly state your reasoning (e.g., "broken grammar" or "mentions physical objects"). You are also encouraged to note your Confidence (High, Medium, or Low) to help us align with the LLM's judgments.
something is off? If a sentence is confusing, or if you have a question about the guidelines, or if something seems wrong, please reach out directly rather than guessing.

### Quality & Contact
Workload: 280 completions across 20 idioms.
Agreement: Cohen's Kappa target is >= 0.8. Disagreements will be resolved jointly.
Contact: the IdioSteer maintainers (via the project GitHub repository)
Thank you very much for your contribution!
