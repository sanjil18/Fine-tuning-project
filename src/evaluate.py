"""
Combine model_outputs.csv (from notebooks/03_evaluation.ipynb, run in Colab)
with your RAG system's answers, and set up a manual scoring table.

This does NOT run any models itself — it's a local, no-GPU-needed step that
prepares results/comparison_table.csv for you to fill in by hand.

Why manual scoring instead of an automated metric: there's no ground-truth
answer key for these questions, and automated text-similarity metrics (BLEU,
ROUGE, etc.) don't reliably capture factual correctness or hallucination.
Manually judging each answer against your source notes is more honest and
more defensible in a write-up than a fabricated automatic score.

Usage:
    python src/evaluate.py results/model_outputs.csv
"""

import sys
import csv
from pathlib import Path


def build_comparison_table(model_outputs_path: str, output_path: str = "results/comparison_table.csv"):
    with open(model_outputs_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fieldnames = [
        "question",
        "category",
        "base_model_answer",
        "base_model_latency_sec",
        "base_correctness",       # 1-5, fill in manually
        "base_faithfulness",      # 1-5, fill in manually
        "finetuned_answer",
        "finetuned_latency_sec",
        "finetuned_correctness",  # 1-5, fill in manually
        "finetuned_faithfulness", # 1-5, fill in manually
        "rag_answer",             # fill in manually: run the same question through your RAG app
        "rag_latency_sec",        # fill in manually
        "rag_correctness",        # 1-5, fill in manually
        "rag_faithfulness",       # 1-5, fill in manually
        "notes",                  # free text: anything worth flagging (hallucination, hedging, etc.)
    ]

    Path("results").mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(out_row)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print("\nNext steps (manual):")
    print("  1. Run each 'question' through your RAG project and fill in rag_answer / rag_latency_sec.")
    print("  2. Score every answer 1-5 on:")
    print("     - correctness: is the answer factually right?")
    print("     - faithfulness: for in_training questions, does it match what your notes actually say?")
    print("       (faithfulness is less meaningful for out_of_training/general_knowledge rows)")
    print("  3. Use the 'notes' column to flag anything interesting: confident hallucination,")
    print("     hedging/uncertainty, refusal, or a notably strong/weak answer.")
    print("  4. Once scored, compute average scores per category per model for your write-up table.")

    # Quick latency summary, since that part needs no manual scoring
    categories = {}
    for row in rows:
        cat = row.get("category", "unknown")
        categories.setdefault(cat, {"base": [], "finetuned": []})
        try:
            categories[cat]["base"].append(float(row["base_model_latency_sec"]))
            categories[cat]["finetuned"].append(float(row["finetuned_latency_sec"]))
        except (ValueError, KeyError):
            pass

    print("\nAverage latency by category (base vs fine-tuned):")
    for cat, vals in categories.items():
        if vals["base"] and vals["finetuned"]:
            avg_base = sum(vals["base"]) / len(vals["base"])
            avg_ft = sum(vals["finetuned"]) / len(vals["finetuned"])
            print(f"  {cat}: base={avg_base:.2f}s, finetuned={avg_ft:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <path_to_model_outputs.csv>")
        sys.exit(2)
    build_comparison_table(sys.argv[1])
