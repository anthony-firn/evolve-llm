#!/usr/bin/env python3
"""Make the vision encoder fixtures: tiny Qwen2.5-VL and Qwen3-VL vision towers with random weights.

Writes tests/fixtures/tiny-qwen25vl-vision.json holding a GGUF laid out as llama.cpp's mmproj files are
(clip.projector_type qwen2.5vl_merger, v.blk.N.attn_q/k/v, v.patch_embd.weight and .weight.1 for
the two temporal slices, v.post_ln, mm.0, mm.2; base64, 600 KB, so the test runs in CI), a test
image (RGB bytes, 168x140: a multiple of 28, so no resizing) and the embeddings that
Hugging Face transformers' Qwen2_5_VisionTransformerPretrainedModel gives for it.

Needs torch, transformers (with Qwen2.5-VL) and gguf:
    python3 tools/make_vision_fixture.py
"""
import base64
import json
from pathlib import Path

import numpy as np
import torch
import gguf
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures"
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
W, H = 168, 140
PATCH, MERGE, TEMPORAL = 14, 2, 2


def main():
    torch.manual_seed(0)
    cfg = Qwen2_5_VLVisionConfig(depth=4, hidden_size=64, intermediate_size=96, num_heads=4, out_hidden_size=48, patch_size=PATCH,
                                 spatial_merge_size=MERGE, temporal_patch_size=TEMPORAL, window_size=112, fullatt_block_indexes=[1, 3],
                                 in_channels=3, hidden_act="silu")
    cfg._attn_implementation = "eager"
    model = Qwen2_5_VisionTransformerPretrainedModel(cfg).eval().float()
    with torch.no_grad():  # random weights large enough to matter, small enough to stay finite
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.08)

    # A test image with structure in both directions.
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([(xx * 255 // (W - 1)), (yy * 255 // (H - 1)), ((xx // 14 + yy // 14) % 2) * 200 + 20], axis=-1).astype(np.uint8)
    # The image processor's patches: rescale, normalize, repeat for the two frames, then
    # (grid_t, T, C, gh/m, m, P, gw/m, m, P) -> (grid_t, gh/m, gw/m, m, m, C, T, P, P).
    x = (img.astype(np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    x = np.repeat(x.transpose(2, 0, 1)[None], TEMPORAL, axis=0)
    gh, gw = H // PATCH, W // PATCH
    x = x.reshape(1, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH)
    with torch.no_grad():
        out = model(torch.from_numpy(np.ascontiguousarray(x)), grid_thw=torch.tensor([[1, gh, gw]])).float().numpy()

    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "tiny-qwen25vl-vision.gguf"
    w = gguf.GGUFWriter(str(tmp), "clip")
    w.add_string("general.type", "clip-vision")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "qwen2.5vl_merger")
    w.add_uint32("clip.vision.image_size", 560)
    w.add_uint32("clip.vision.patch_size", PATCH)
    w.add_uint32("clip.vision.embedding_length", cfg.hidden_size)
    w.add_uint32("clip.vision.feed_forward_length", cfg.intermediate_size)
    w.add_uint32("clip.vision.block_count", cfg.depth)
    w.add_uint32("clip.vision.attention.head_count", cfg.num_heads)
    w.add_uint32("clip.vision.projection_dim", cfg.out_hidden_size)
    w.add_uint32("clip.vision.n_wa_pattern", 2)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    w.add_array("clip.vision.image_mean", MEAN)
    w.add_array("clip.vision.image_std", STD)
    w.add_bool("clip.use_silu", True)
    sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
    f16 = lambda a: a.astype(np.float16)
    pe = sd["patch_embed.proj.weight"]  # (dim, C, T, P, P)
    w.add_tensor("v.patch_embd.weight", f16(pe[:, :, 0]))
    w.add_tensor("v.patch_embd.weight.1", f16(pe[:, :, 1]))
    d = cfg.hidden_size
    for i in range(cfg.depth):
        b = f"blocks.{i}."
        qkv, qkv_b = sd[b + "attn.qkv.weight"], sd[b + "attn.qkv.bias"]
        for j, n in enumerate("qkv"):
            w.add_tensor(f"v.blk.{i}.attn_{n}.weight", f16(qkv[j * d:(j + 1) * d]))
            w.add_tensor(f"v.blk.{i}.attn_{n}.bias", qkv_b[j * d:(j + 1) * d])
        w.add_tensor(f"v.blk.{i}.attn_out.weight", f16(sd[b + "attn.proj.weight"]))
        w.add_tensor(f"v.blk.{i}.attn_out.bias", sd[b + "attn.proj.bias"])
        w.add_tensor(f"v.blk.{i}.ln1.weight", sd[b + "norm1.weight"])
        w.add_tensor(f"v.blk.{i}.ln2.weight", sd[b + "norm2.weight"])
        for n in ("gate", "up", "down"):
            w.add_tensor(f"v.blk.{i}.ffn_{n}.weight", f16(sd[b + f"mlp.{n}_proj.weight"]))
            w.add_tensor(f"v.blk.{i}.ffn_{n}.bias", sd[b + f"mlp.{n}_proj.bias"])
    w.add_tensor("v.post_ln.weight", sd["merger.ln_q.weight"])
    w.add_tensor("mm.0.weight", f16(sd["merger.mlp.0.weight"]))
    w.add_tensor("mm.0.bias", sd["merger.mlp.0.bias"])
    w.add_tensor("mm.2.weight", f16(sd["merger.mlp.2.weight"]))
    w.add_tensor("mm.2.bias", sd["merger.mlp.2.bias"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    # The reference ran with the float32 weights; the fixture stores matrices in float16, so the
    # engine is compared against a run with the float16-rounded weights.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() > 1:
                p.copy_(p.half().float())
        out16 = model(torch.from_numpy(np.ascontiguousarray(x)), grid_thw=torch.tensor([[1, gh, gw]])).float().numpy()
    (OUT / "tiny-qwen25vl-vision.json").write_text(json.dumps({
        "gguf": base64.b64encode(tmp.read_bytes()).decode(),
        "width": W, "height": H, "rgb": base64.b64encode(img.tobytes()).decode(),
        "grid": [1, gh, gw], "tokens": int(out16.shape[0]), "dim": int(out16.shape[1]),
        "embeddings": [round(float(v), 6) for v in out16.reshape(-1)],
        "embeddings_f32_weights_max_diff": float(np.abs(out16 - out).max()),
    }))
    print(f"wrote tests/fixtures/tiny-qwen25vl-vision.json: {out16.shape[0]} image tokens x {out16.shape[1]}, |f16 - f32 weights| <= {np.abs(out16 - out).max():.2e}")


def qwen3vl():
    """tests/fixtures/tiny-qwen3vl-vision.json: the Qwen3-VL encoder (16-pixel patches, a learned
    position table interpolated to the image, LayerNorm, tanh-GELU MLP, whole-image attention,
    and deepstack mergers after two blocks), laid out as llama.cpp's qwen3vl_merger mmproj."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    torch.manual_seed(1)
    p16, w_, h_ = 16, 160, 128
    cfg = Qwen3VLVisionConfig(depth=4, hidden_size=64, intermediate_size=96, num_heads=4, out_hidden_size=48, patch_size=p16,
                              spatial_merge_size=MERGE, temporal_patch_size=TEMPORAL, num_position_embeddings=16,
                              deepstack_visual_indexes=[1, 2], hidden_act="gelu_pytorch_tanh", in_channels=3)
    cfg._attn_implementation = "eager"
    model = Qwen3VLVisionModel(cfg).eval().float()
    with torch.no_grad():
        for name, prm in model.named_parameters():
            # Norm weights near 1 as in trained models, and larger first-MLP weights, put the
            # activations where tanh-GELU and exact GELU differ.
            if "norm" in name and name.endswith("weight"):
                prm.copy_(1.0 + 0.1 * torch.randn_like(prm))
            else:
                prm.copy_(torch.randn_like(prm) * (0.5 if "pos_embed" in name else 0.3 if "mlp.linear_fc1" in name else 0.08))
            if prm.dim() > 1 and "pos_embed" not in name:
                prm.copy_(prm.half().float())  # matrices are stored in f16
    yy, xx = np.mgrid[0:h_, 0:w_]
    img = np.stack([(xx * 255 // (w_ - 1)), (yy * 255 // (h_ - 1)), ((xx // 16 + yy // 16) % 2) * 200 + 20], axis=-1).astype(np.uint8)
    mean = std = [0.5, 0.5, 0.5]
    x = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
    x = np.repeat(x.transpose(2, 0, 1)[None], TEMPORAL, axis=0)
    gh, gw = h_ // p16, w_ // p16
    x = x.reshape(1, TEMPORAL, 3, gh // MERGE, MERGE, p16, gw // MERGE, MERGE, p16)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, 3 * TEMPORAL * p16 * p16)
    with torch.no_grad():
        out, deep = model(torch.from_numpy(np.ascontiguousarray(x)), grid_thw=torch.tensor([[1, gh, gw]]))
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "tiny-qwen3vl-vision.gguf"
    w = gguf.GGUFWriter(str(tmp), "clip")
    w.add_string("general.type", "mmproj")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "qwen3vl_merger")
    w.add_uint32("clip.vision.image_size", 768)
    w.add_uint32("clip.vision.patch_size", p16)
    w.add_uint32("clip.vision.embedding_length", cfg.hidden_size)
    w.add_uint32("clip.vision.feed_forward_length", cfg.intermediate_size)
    w.add_uint32("clip.vision.block_count", cfg.depth)
    w.add_uint32("clip.vision.attention.head_count", cfg.num_heads)
    w.add_uint32("clip.vision.projection_dim", cfg.out_hidden_size)
    w.add_uint32("clip.vision.spatial_merge_size", MERGE)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    w.add_array("clip.vision.image_mean", mean)
    w.add_array("clip.vision.image_std", std)
    w.add_bool("clip.use_gelu", True)
    w.add_array("clip.vision.is_deepstack_layers", [i in cfg.deepstack_visual_indexes for i in range(cfg.depth)])
    sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
    f16 = lambda a: a.astype(np.float16)
    pe = sd["patch_embed.proj.weight"]
    w.add_tensor("v.patch_embd.weight", f16(pe[:, :, 0]))
    w.add_tensor("v.patch_embd.weight.1", f16(pe[:, :, 1]))
    w.add_tensor("v.patch_embd.bias", sd["patch_embed.proj.bias"])
    w.add_tensor("v.position_embd.weight", sd["pos_embed.weight"])
    for i in range(cfg.depth):
        b = f"blocks.{i}."
        w.add_tensor(f"v.blk.{i}.attn_qkv.weight", f16(sd[b + "attn.qkv.weight"]))
        w.add_tensor(f"v.blk.{i}.attn_qkv.bias", sd[b + "attn.qkv.bias"])
        w.add_tensor(f"v.blk.{i}.attn_out.weight", f16(sd[b + "attn.proj.weight"]))
        w.add_tensor(f"v.blk.{i}.attn_out.bias", sd[b + "attn.proj.bias"])
        for n in ("1", "2"):
            w.add_tensor(f"v.blk.{i}.ln{n}.weight", sd[b + f"norm{n}.weight"])
            w.add_tensor(f"v.blk.{i}.ln{n}.bias", sd[b + f"norm{n}.bias"])
        w.add_tensor(f"v.blk.{i}.ffn_up.weight", f16(sd[b + "mlp.linear_fc1.weight"]))
        w.add_tensor(f"v.blk.{i}.ffn_up.bias", sd[b + "mlp.linear_fc1.bias"])
        w.add_tensor(f"v.blk.{i}.ffn_down.weight", f16(sd[b + "mlp.linear_fc2.weight"]))
        w.add_tensor(f"v.blk.{i}.ffn_down.bias", sd[b + "mlp.linear_fc2.bias"])
    w.add_tensor("v.post_ln.weight", sd["merger.norm.weight"])
    w.add_tensor("v.post_ln.bias", sd["merger.norm.bias"])
    w.add_tensor("mm.0.weight", f16(sd["merger.linear_fc1.weight"]))
    w.add_tensor("mm.0.bias", sd["merger.linear_fc1.bias"])
    w.add_tensor("mm.2.weight", f16(sd["merger.linear_fc2.weight"]))
    w.add_tensor("mm.2.bias", sd["merger.linear_fc2.bias"])
    for j, layer in enumerate(cfg.deepstack_visual_indexes):
        m = f"deepstack_merger_list.{j}."
        w.add_tensor(f"v.deepstack.{layer}.norm.weight", sd[m + "norm.weight"])
        w.add_tensor(f"v.deepstack.{layer}.norm.bias", sd[m + "norm.bias"])
        w.add_tensor(f"v.deepstack.{layer}.fc1.weight", f16(sd[m + "linear_fc1.weight"]))
        w.add_tensor(f"v.deepstack.{layer}.fc1.bias", sd[m + "linear_fc1.bias"])
        w.add_tensor(f"v.deepstack.{layer}.fc2.weight", f16(sd[m + "linear_fc2.weight"]))
        w.add_tensor(f"v.deepstack.{layer}.fc2.bias", sd[m + "linear_fc2.bias"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out = out.float().numpy()
    deep = [d.float().numpy() for d in deep]
    (OUT / "tiny-qwen3vl-vision.json").write_text(json.dumps({
        "gguf": base64.b64encode(tmp.read_bytes()).decode(),
        "width": w_, "height": h_, "rgb": base64.b64encode(img.tobytes()).decode(),
        "grid": [1, gh, gw], "tokens": int(out.shape[0]), "dim": int(out.shape[1]),
        "embeddings": [round(float(v), 6) for v in out.reshape(-1)],
        "deepstack": [[round(float(v), 6) for v in d.reshape(-1)] for d in deep],
    }))
    print(f"wrote tests/fixtures/tiny-qwen3vl-vision.json: {out.shape[0]} image tokens x {out.shape[1]}, {len(deep)} deepstack features")


if __name__ == "__main__":
    main()
    qwen3vl()
