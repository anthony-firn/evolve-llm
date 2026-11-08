"""Generate reference fixtures from independent implementations.

* Dequantization: gguf-py's NumPy dequantizers on random blocks of every supported format.
* Tokenizers: Hugging Face `tokenizers` with each model's original tokenizer.json.
* Chat templates: transformers' Jinja renderer with the template stored in the GGUF file.
* Logits: transformers' PyTorch model loaded from the same GGUF file (dequantized to float32).

Usage: python tools/make_fixtures.py --models ~/.cache/evolve-llm/models --tokenizers DIR [--skip-logits]
Requires: numpy, torch, transformers 4.57, tokenizers, gguf.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as Q, GGUFReader
from gguf.quants import dequantize

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

MODELS = {
    "stories15m": ("llama2.c-stories15M.Q8_0.gguf", "stories"),
    "smollm2": ("SmolLM2-135M-Instruct-Q8_0.gguf", "smollm2"),
    "qwen25": ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen25"),
    "qwen3": ("Qwen3-0.6B-Q8_0.gguf", "qwen3"),
    "llama32": ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "llama32"),
}

TEXTS = [
    "Hello world",
    "  leading spaces and trailing   ",
    "The year 2026 had 365 days; 1234567 is a number.",
    "naïve café, résumé — “quotes” and ‘apostrophes’",
    "I'm, you're, they've, we'll, she'd, IT'S, I'D",
    "日本語のテキストと中文混合",
    "emoji 🚀🔥 and 👩‍👩‍👧‍👦 family",
    "def f(x):\n    return x**2  # comment\n\n\n\tTabbed",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "é decomposed accent and Å vs Å",
    "multiple\n\n\nnewlines\r\nand CRLF",
    "   ",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "http://example.com/path?query=1&b=2#frag",
    "Ünïcödé ÀÉÎÕÜ ß ǅ",
    "123 4567 89012 3.14159 1,000,000",
    "",
]

CONVERSATION = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "What is WebGPU?"},
    {"role": "assistant", "content": "A browser API for GPU graphics and compute."},
    {"role": "user", "content": "Can it run language models?"},
]

# Format id -> bytes per block, values per block, offset of the f16 scale fields.
BLOCKS = {
    Q.Q4_0: (18, [0]), Q.Q4_1: (20, [0, 2]), Q.Q5_0: (22, [0]), Q.Q5_1: (24, [0, 2]), Q.Q8_0: (34, [0]),
    Q.Q2_K: (84, [80, 82]), Q.Q3_K: (110, [108]), Q.Q4_K: (144, [0, 2]), Q.Q5_K: (176, [0, 2]),
    Q.Q6_K: (210, [208]), Q.IQ4_NL: (18, [0]), Q.IQ4_XS: (136, [0]),
}


def quant_fixtures(rng):
    cases = []
    for t, (size, halves) in BLOCKS.items():
        raw = bytearray(rng.integers(0, 256, size=size * 3, dtype=np.uint8).tobytes())
        for b in range(3):
            for h in halves:
                v = np.float16(rng.uniform(-0.05, 0.05) if b else 0.01)
                raw[b * size + h:b * size + h + 2] = v.tobytes()
        values = dequantize(np.frombuffer(bytes(raw), dtype=np.uint8), t).astype(np.float32).reshape(-1)
        cases.append({"type": int(t), "name": t.name, "bytes": base64.b64encode(bytes(raw)).decode(),
                      "values": [float(v) for v in values]})
    floats = rng.normal(size=64).astype(np.float32)
    cases.append({"type": 1, "name": "F16", "bytes": base64.b64encode(floats.astype(np.float16).tobytes()).decode(),
                  "values": [float(v) for v in floats.astype(np.float16).astype(np.float32)]})
    bf = (floats.view(np.uint32) >> 16).astype(np.uint16)
    cases.append({"type": 30, "name": "BF16", "bytes": base64.b64encode(bf.tobytes()).decode(),
                  "values": [float(v) for v in (bf.astype(np.uint32) << 16).view(np.float32)]})
    return cases


def gguf_template(path):
    r = GGUFReader(str(path))
    f = r.fields.get("tokenizer.chat_template")
    return bytes(f.parts[-1]).decode() if f else None


def model_fixture(key, models, tokdir, logits):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    fname, tdir = MODELS[key]
    path = models / fname
    tok = Tokenizer.from_file(str(tokdir / tdir / "tokenizer.json"))
    fx = {"file": fname, "tokenizer": [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in TEXTS]}
    template = gguf_template(path)
    hf = AutoTokenizer.from_pretrained(str(tokdir / tdir))
    if template:
        chats = []
        variants = [{}] if key != "qwen3" else [{"enable_thinking": True}, {"enable_thinking": False}]
        for extra in variants:
            text = hf.apply_chat_template(CONVERSATION, chat_template=template, tokenize=False,
                                          add_generation_prompt=True, date_string="08 Nov 2026", **extra)
            chats.append({"messages": CONVERSATION, "options": {"date": [2026, 11, 8], **extra}, "text": text})
        fx["chat"] = chats
        prompt = chats[-1]["text"]
    else:
        prompt = "Once upon a time, a little robot"
    ids = tok.encode(prompt, add_special_tokens=False).ids
    if key == "stories15m":
        ids = [1] + ids
    fx["prompt_ids"] = ids
    if logits:
        import torch
        from transformers import AutoModelForCausalLM
        device = "cuda" if torch.cuda.is_available() and key == "llama32" else "cpu"
        model = AutoModelForCausalLM.from_pretrained(str(models), gguf_file=fname, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(torch.tensor([ids], device=device)).logits[0, -1].float().cpu().numpy()
        top = np.argsort(-out)[:20]
        fx["logits"] = {"top": [[int(i), float(out[i])] for i in top], "max": float(out.max()), "min": float(out.min()),
                        "mean": float(out.mean()), "std": float(out.std())}
        del model
    return fx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--tokenizers", type=Path, required=True)
    ap.add_argument("--skip-logits", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    (OUT / "quant.json").write_text(json.dumps(quant_fixtures(rng)))
    print("quant.json")
    for key in a.only or MODELS:
        fx = model_fixture(key, a.models, a.tokenizers, not a.skip_logits)
        (OUT / f"{key}.json").write_text(json.dumps(fx, ensure_ascii=False, indent=1))
        print(f"{key}.json", "with logits" if "logits" in fx else "")


if __name__ == "__main__":
    main()
