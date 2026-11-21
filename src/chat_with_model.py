"""
Standalone local chat with your fine-tuned model (base model + LoRA adapter).

Runs on CPU -- no GPU needed, but expect each response to take anywhere from
10 seconds to a couple of minutes depending on your machine. This is for
casual testing/demo purposes, not fast interactive chat.

Usage:
    python src/chat_with_model.py
    python src/chat_with_model.py --adapter models/lora_adapter_50ex   (to test the other adapter)
    python src/chat_with_model.py --base-only                          (to compare against the base model, no adapter)

Requirements (install once):
    pip install transformers peft torch accelerate
"""

import argparse
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER = "models/lora_adapter_150ex"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Path to the LoRA adapter folder")
    parser.add_argument("--base-only", action="store_true", help="Skip the adapter, chat with the raw base model")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading base model ({MODEL_NAME}) on CPU... this can take a minute.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")

    if not args.base_only:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {args.adapter} ...")
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"Ready -- chatting with FINE-TUNED model ({args.adapter}).")
    else:
        print("Ready -- chatting with BASE model (no adapter).")

    model.eval()
    print("Type your question and press Enter. Type 'quit' or 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")

        start = time.time()
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.time() - start

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nModel ({elapsed:.1f}s): {response.strip()}\n")


if __name__ == "__main__":
    main()
