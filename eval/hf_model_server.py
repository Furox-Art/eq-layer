"""Persistent Hugging Face chat-model backend for A/B pilot generation.

Protocol: read one JSON object per stdin line and emit one JSON object per
stdout line. The model is loaded once at process start.

Example:
  python eval/hf_model_server.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --max-new-tokens 96
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help=(
            "Allow non-deterministic kernels. Faster, but identical inputs stop "
            "producing identical outputs across runs."
        ),
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    torch.set_num_threads(max(1, args.cpu_threads))

    # Greedy decoding is only deterministic given identical logits, and CPU
    # matmul reduction order varies between runs. Two pilot runs at temperature 0
    # with the same per-case seed produced identical text for only 8 of 12 EQ
    # arms and 7 of 12 baseline arms, which quietly weakens any paired
    # comparison. Single-threaded deterministic kernels cost some throughput and
    # buy reproducibility, which is what a measurement harness needs.
    if not args.nondeterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        trust_remote_code=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        payload = json.loads(raw)
        requested_model = str(payload.get("model_id") or "")
        if requested_model != args.model_id:
            print(
                json.dumps(
                    {
                        "error": (
                            f"model_id mismatch: requested={requested_model!r} "
                            f"loaded={args.model_id!r}"
                        )
                    }
                ),
                flush=True,
            )
            continue

        messages = payload.get("messages") or []
        seed = int(payload.get("seed", 0))
        temperature = float(payload.get("temperature", 0.0))

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                }
            )
        else:
            generate_kwargs["do_sample"] = False

        with torch.inference_mode():
            generated = model.generate(**encoded, **generate_kwargs)

        prompt_tokens = encoded["input_ids"].shape[1]
        new_tokens = generated[0, prompt_tokens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(json.dumps({"text": text}, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
