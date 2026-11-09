"""Build tiny random-weight models of each mixture-of-experts architecture and their reference logits.

Each model is created with transformers (weights rounded to f16 so the GGUF holds them exactly),
converted to GGUF by llama.cpp's convert_hf_to_gguf.py, and run in float32 PyTorch. The GGUF files
go to the models directory; the logits go to tests/fixtures/tiny-<name>.json for tests/tiny.rs.

Usage:
  python3 tools/make_tiny_models.py --llama-cpp /path/to/llama.cpp --tokenizer /path/to/smollm2 \
      --models ~/.cache/evolve-llm/models
The tokenizer directory needs tokenizer.json and tokenizer_config.json of SmolLM2 (49152 tokens).
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VOCAB = 49152
TOKENS = [1, 504, 2381, 314, 260, 3519, 30, 1650, 7, 2210, 422, 19, 9087, 33, 1184, 260, 4920, 25, 351, 12]

YARN = {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64, "beta_fast": 32, "beta_slow": 1,
        "mscale": 1.0, "mscale_all_dim": 1.0}

SPECS = {
    "qwen3moe": ("Qwen3MoeConfig", "Qwen3MoeForCausalLM", dict(
        hidden_size=128, intermediate_size=256, moe_intermediate_size=64, num_experts=8, num_experts_per_tok=2,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32, norm_topk_prob=True,
        decoder_sparse_step=1, mlp_only_layers=[], tie_word_embeddings=True, rope_theta=1e6)),
    "mixtral": ("MixtralConfig", "MixtralForCausalLM", dict(
        hidden_size=128, intermediate_size=64, num_local_experts=4, num_experts_per_tok=2, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True, rope_theta=1e6, sliding_window=None)),
    # DeepSeek V3 / R1 / Kimi K2 layout: query compression, sigmoid routing with a selection bias,
    # group-limited choice, a shared expert, a leading dense layer and YaRN.
    "deepseek3": ("DeepseekV3Config", "DeepseekV3ForCausalLM", dict(
        hidden_size=128, intermediate_size=256, moe_intermediate_size=64, num_hidden_layers=3,
        first_k_dense_replace=1, n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2, n_group=4,
        topk_group=2, routed_scaling_factor=2.5, norm_topk_prob=True, num_attention_heads=4,
        num_key_value_heads=4, q_lora_rank=64, kv_lora_rank=64, qk_nope_head_dim=32, qk_rope_head_dim=32,
        v_head_dim=64, rope_scaling=YARN, max_position_embeddings=256, tie_word_embeddings=False,
        rope_theta=10000.0, rope_interleave=True,
        # Saved to config.json as DeepSeek's own checkpoints have it, so the GGUF records sigmoid routing.
        scoring_func="sigmoid", topk_method="noaux_tc")),
    # DeepSeek V2 Lite layout: no query compression, softmax routing, two shared experts.
    "deepseek2lite": ("DeepseekV2Config", "DeepseekV2ForCausalLM", dict(
        hidden_size=128, intermediate_size=256, moe_intermediate_size=64, num_hidden_layers=3,
        first_k_dense_replace=1, n_routed_experts=8, n_shared_experts=2, num_experts_per_tok=2, n_group=1,
        topk_group=1, topk_method="greedy", routed_scaling_factor=1.0, norm_topk_prob=False,
        num_attention_heads=4, num_key_value_heads=4, q_lora_rank=None, kv_lora_rank=64, qk_nope_head_dim=32,
        qk_rope_head_dim=32, v_head_dim=64, max_position_embeddings=256, tie_word_embeddings=False,
        rope_theta=10000.0)),
}


def build(name, tokenizer, out_dir):
    import transformers
    cfg_cls, model_cls, kw = SPECS[name]
    cfg = getattr(transformers, cfg_cls)(vocab_size=VOCAB, initializer_range=0.08, rms_norm_eps=1e-6,
                                         bos_token_id=1, eos_token_id=2, **kw)
    torch.manual_seed(2026)
    model = getattr(transformers, model_cls)(cfg).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            if n.endswith("e_score_correction_bias"):
                p.copy_(0.05 * torch.rand_like(p))
            p.copy_(p.half().float())
        for n, b in model.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_((0.05 * torch.rand_like(b)).half().float())
    model.save_pretrained(out_dir, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(tokenizer / f, out_dir / f)
    with torch.no_grad():
        logits = model(torch.tensor([TOKENS])).logits[0].float().numpy()
    return logits


def pack(v):
    return base64.b64encode(np.asarray(v, dtype=np.float32).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    a.models.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(a.llama_cpp / "gguf-py")}
    for name in a.only or SPECS:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            logits = build(name, a.tokenizer, tmp)
            out = a.models / f"tiny-{name}.gguf"
            # Also an 8-bit copy, to run the expert kernels on quantized weights.
            for kind, path in (("f16", out), ("q8_0", a.models / f"tiny-{name}-q8_0.gguf")):
                subprocess.run([sys.executable, str(a.llama_cpp / "convert_hf_to_gguf.py"), str(tmp), "--outtype",
                                kind, "--outfile", str(path)], check=True, env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        fx = {"file": out.name, "tokens": TOKENS, "positions": [3, len(TOKENS) - 1],
              "logits": [pack(logits[3]), pack(logits[-1])],
              "range": float(logits.max() - logits.min())}
        (ROOT / "tests/fixtures" / f"tiny-{name}.json").write_text(json.dumps(fx))
        print(f"tiny-{name}: {out.stat().st_size / 1e6:.1f} MB, logit range {fx['range']:.3f}")


if __name__ == "__main__":
    main()
