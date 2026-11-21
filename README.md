# LLM Fine-Tuning (QLoRA) vs. RAG — A Head-to-Head Comparison

A LoRA/QLoRA fine-tune of Qwen2.5-1.5B-Instruct on my own lecture notes, evaluated head-to-head against the base model and against an existing RAG system built on the *same* source notes — to answer a practical question: **when should you fine-tune vs. retrieve?**

## Why this project

Fine-tuning and RAG are usually demonstrated as separate portfolio projects. Here, both were built on identical source material (ML/DL, databases, Python, transformers lecture notes), which makes the comparison a genuine experiment rather than an assumption. The goal wasn't to prove fine-tuning "works" — it was to find out, with real evidence, what each approach is actually good and bad at.

## Setup

**Model:** Qwen2.5-1.5B-Instruct, fine-tuned with QLoRA (4-bit base + LoRA adapters) on a free Colab T4 GPU.

**Dataset:** 150 instruction/output Q&A pairs (`data/qa_dataset.jsonl`), generated semi-automatically — an LLM drafted each pair from a note chunk, every pair was manually reviewed before being kept. Validated programmatically (`src/validate_dataset.py`): JSON structure, duplicate/near-duplicate detection, answer-length outliers.

**Training:** Two adapters were trained and kept, to see how behavior changes with scale:
- `models/lora_adapter_50ex/` — first pipeline-validation run, 50 examples
- `models/lora_adapter_150ex/` — full dataset, 150 examples

**Evaluation:** 18 held-out questions (`data/eval_questions.jsonl`) split into three categories, run through all three systems (base model, fine-tuned model, RAG):
- **in_training** (6) — topics directly covered in the notes
- **out_of_training** (6) — adjacent CS/ML topics never in the notes (sharding, GANs, async/await, sparse attention, replication, knowledge distillation)
- **general_knowledge** (6) — unrelated factual questions, to check for catastrophic forgetting

Every answer was scored 1–5 on correctness and (where meaningful) faithfulness to the source notes. Full results: `results/comparison_table_final.csv`.

## Key findings

### 1. Fine-tuning is faster — but not for the reason you'd expect
The fine-tuned model answered 17–45% faster than the base model across all three question categories. This isn't an architecture effect (LoRA adds negligible inference overhead) — it's a **length effect**: fine-tuning shifted the model toward shorter, more concise answers (matching the dataset's style), so it hit its stop token sooner. Confirmed by comparing raw answer lengths between the 50-example and 150-example adapters' outputs.

### 2. Fine-tuning learns *style* faster than it learns *facts*
At both 50 and 150 examples, neither adapter closely reproduced the exact trained answers verbatim — they generalized/paraphrased instead. That's a reasonable, non-overfit result at this dataset size, but it means the "fine-tuned model speaks in your notes' voice" story is more about tone and brevity than fact injection.

### 3. Fine-tuning hallucinates confidently — twice, documented
- Asked about `async/await` vs. multithreading (out-of-training), the fine-tuned model incorrectly claimed asyncio "runs each coroutine on its own thread." It doesn't — asyncio uses a single-threaded event loop. The base model did not make this error.
- Asked who wrote *Romeo and Juliet* (general knowledge — a forgetting check), the fine-tuned model got the author right but fabricated a detail: that it "premiered at the Globe Theatre in 1594" — impossible, since the Globe wasn't built until 1599.

Both are clean, reproducible examples of confident fabrication introduced by fine-tuning at small scale, even on topics with zero relation to the training data.

### 4. RAG's failure mode is safer — but it has a real, fixable retrieval bug
RAG never hallucinated once across all 18 questions — on anything outside its document store, it correctly said it didn't have enough information. But it also refused a question it *should* have been able to answer: "What is the difference between a Python decorator and a context manager?" — despite the source notes explicitly covering both.

**Root cause, measured directly:** the combined query's embedding diluted both concepts. The Context Managers chunk scored 0.292 against the combined query (just under the 0.3 retrieval threshold) but 0.41 on its own dedicated query — the classic embedding-averaging problem with multi-part questions.

**Fix, implemented and verified (commit `77e7a78`):**
- **Query decomposition** — comparison-style questions ("difference between X and Y") are now split into sub-queries, retrieved separately, merged and deduplicated.
- **Heading-aware chunk prefixing** — chunks are now prefixed with their section heading rather than the full document title (the old title-based prefix was accidentally boosting every chunk in `python_advanced.txt` because the title itself contains the word "Decorators," crowding out the actually-relevant chunk).

Verified before/after: the target chunk moved from 0.292 (refused) to 0.407 (answered), a +0.107 margin over the threshold and +0.081 margin over the nearest unrelated chunk. Confirmed end-to-end through the live app, not just the retrieval script.

### 5. One case where fine-tuning was *more* accurate than the base model
On "what does self-attention compute for each word?", the base model incorrectly described self-attention as only looking at *previous* tokens — that's causal/masked attention (GPT-style decoders), not general self-attention. The fine-tuned model got this right, matching the notes' Query/Key/Value framing precisely. Fine-tuning isn't strictly worse — on topics it was actually trained on, it can correct base-model imprecision.

## Conclusion: when to use which

| | RAG | Fine-tuning |
|---|---|---|
| On in-scope topics | Accurate, faithful to source | Accurate, more concise, sometimes corrects base model errors |
| On out-of-scope topics | Refuses safely (no hallucination) | Answers confidently — sometimes wrong |
| Coverage gaps | Possible on multi-part questions (found and fixed here) | None — always attempts an answer |
| Inference cost | Retrieval + generation | Generation only (faster) |
| Updating knowledge | Add/edit documents | Requires retraining |

**For a small, fixed knowledge base where correctness matters more than fluency, RAG is the safer default** — its retrieval-gating naturally prevents fabrication, at the cost of occasional coverage gaps that are usually fixable (as shown here). **Fine-tuning is better suited to shifting tone/format/behavior** consistently across all inputs, but at small scale (150 examples) it does not reliably inject new facts, and it inherits the risk of confidently answering outside its actual knowledge — a risk RAG is structurally immune to.

## Repository structure

```
data/               qa_dataset.jsonl (150 pairs), eval_questions.jsonl (18), raw_notes/
notebooks/          01 data generation notes, 02 QLoRA training (Colab), 03 evaluation (Colab)
src/                validate_dataset.py, evaluate.py
models/             lora_adapter_50ex/, lora_adapter_150ex/ (LoRA weights, gitignored)
results/            model_outputs.csv, comparison_table_final.csv
```

## Limitations & honest notes

- 150 examples is enough to validate the pipeline and observe real behavioral effects (speed, style transfer, hallucination risk), but too small to expect reliable fact injection from fine-tuning.
- The RAG fix's shared-qualifier heuristic (`_share_context`) only fires on capitalized words (e.g. "Python") — a fully general context-sharing mechanism across arbitrary shared terms isn't implemented.
- Correctness/faithfulness scoring throughout is manual, not automated — there's no ground-truth answer key, and automated text-similarity metrics don't reliably catch hallucination, so human judgment against the source notes was used instead.

## What I'd do with more time

- Scale the dataset toward 300 pairs and retrain, to see whether fact-injection improves with more data at the same model size.
- Add a lightweight LoRA rank ablation (r=8 vs r=16) to see its effect on the style-vs-fact tradeoff observed here.
- Push the LoRA adapter to Hugging Face Hub as a public, loadable artifact.
