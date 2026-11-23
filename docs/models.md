# Which models to run, and what to build next

Research of 2026-11-22: model cards, Hugging Face GGUF headers (read with range requests),
and public leaderboards. Marks:
- **[I]** independent: a leaderboard or Artificial Analysis (AA);
- **[X]** measured by another model maker;
- **[S]** self-reported by the model's maker.

Which models the app offers is decided by the frontier (`web/model-manager.mjs`): Artificial
Analysis's Coding Index and output tokens per model (`modes` in `web/models.js`) against speed on
this engine, so only quality is taken from outside and only speed is measured here. Our own
`tests/agent_bench.py` checks that the models the frontier picks really do the work.

## Summary

1. **Qwen3.5's architecture (`qwen35`, `qwen35moe`) opens the most.**
   - **What it is:** Gated DeltaNet linear attention in three of four layers, gated attention in the rest.
   - **What it opens:**
     - the best small text models: Qwen3.5-4B/9B (AA 27/32 vs 18 for Qwen3-4B-2507 [I]), FrogNano-4B, MiMo-V2.6-Distill-9B and Ornith-1.5-9B;
     - the best small computer-use models: Holo3.1-4B and Fara1.5-4B;
     - Qwen3.6-35B-A3B.
   - **Memory:** the key-value cache shrinks 4–5×, about 32 KB per token against 144 KB for Qwen3-4B.
   - **Kernel:** our Kimi delta attention (`KDA`) is the same recurrence with a decay per key channel; Gated DeltaNet uses one decay per head.
   - **Next in line:** `qwen3next` (Qwen3-Coder-Next) uses the same block.
2. **Vision: the Qwen3-VL encoder (`qwen3vl_merger`) comes next.**
   - **What it opens:** about 20 strong 2026 GUI models, on two bases:
     - Qwen3.5, without deepstack: Holo3.1, Fara1.5 and UI-Venus-2;
     - Qwen3-VL, with deepstack: GUI-Owl-1.5, Holo2, MAI-UI and UI-Venus-1.5.
   - **Already built:** the Qwen2.5-VL encoder (`src/vision.rs`, `src/gpu/vision.rs`) covers UI-TARS-1.5-7B, Fara-7B and Holo1.5.
3. **Usable now, no engine work:**
   - **Nanbeige4.1-3B** (`llama`, 2.44 GB). Its predecessor tops BFCL v4 among models of 8B or smaller at 51.4 [I], against 35.7 for Qwen3-4B-Instruct-2507 [I].
   - **Qwen3-4B-Thinking-2507** (`qwen3`): BFCL v3 71.2 vs 61.9 [S].
   - **The 3B-active MoE models in the catalog:** Qwen3-30B-A3B Instruct and Coder.
   - **GLM-4.7-Flash** (`deepseek2`, about 3B active, SWE-bench Verified 59.2 [S]) needs the `glm4` pre-tokenizer and GLM's tool template.
4. **Law:** no small legal fine-tune is worth adopting. SaulLM-7B (2024) scores about 62% on LegalBench, and Lawma is a classifier. Use the best general model with retrieval of primary sources and citation checks; on LegalBench, gpt-oss-20b scores 70.9% [I].

## Computer use (GUI agents)

The shortlist for a browser agent that sees screenshots:

| Model | Base | License | Grounding | Agent | Coordinates | GGUF (Q4 + mmproj) |
|---|---|---|---|---|---|---|
| Holo3.1-4B (Jun 2026) | Qwen3.5-4B | Apache | ScreenSpot-Pro 66.5 [S] | OSWorld 75.8 (H harness) [S] | function call, 0–1000 | 3.07 + 0.37 GB |
| Fara1.5-4B (May 2026) | Qwen3.5-4B | MIT | ScreenSpot-v2 92.3, Pro 59.9 [S] | WebVoyager 80.8, Online-Mind2Web 57.3 [S] | `computer_use` tool call, 1000×1000 | 2.71 + 0.37 GB |
| GUI-Owl-1.5-4B (Feb 2026) | Qwen3-VL-4B | MIT | ScreenSpot-Pro 66.8 [S] | OSWorld-Verified 48.2 [S] | tool call JSON, 0–1000 | 2.50 + 0.45 GB |
| Fara1.5-9B | Qwen3.5-9B | MIT | ScreenSpot-Pro 66.2 [S] | WebVoyager 86.6 [S] | as Fara1.5-4B | 5.91 + 0.92 GB |
| Holo3.1-0.8B (phones) | Qwen3.5-0.8B | Apache | ScreenSpot-Pro 54.3 [S] | OSWorld 34.6 (H harness) [S] | as Holo3.1-4B | 0.67 + 0.12 GB |
| UI-TARS-1.5-7B (2025) | Qwen2.5-VL | Apache | ScreenSpot-Pro 49.6 [S] | OSWorld 27.5 [S] | pixels | 4.68 + 0.86 GB |

UI-Venus-2-9B scores higher still (WebVoyager 90.8 [S]), but its weights have no license yet.

**Qwen3-VL encoder details:**
- Images are cut into 16-pixel patches with 2×2 merging, so each language-model token covers 32×32 pixels.
- Positions come from a learned 48×48 table, interpolated bilinearly, plus 2D RoPE.
- Blocks use LayerNorm with bias and full attention in every layer, and an MLP with tanh-approximated GELU.
- Deepstack, in Qwen3-VL only: three layers each feed their own merger, and the results are added to the image tokens after the language model's first three layers.

**Text side:** interleaved multimodal RoPE (implemented): pair *i* takes the height position if i % 3 == 1, the width position if i % 3 == 2, otherwise time.

**Cost:** a 1280×800 screenshot is about 1,000 image tokens; with a 4B model that's about 11 TFLOP. Use 720p screenshots and keep only the last 1–3 of them.

## Text models by kind of work

| Kind | Runs today | After the Qwen3.5 port |
|---|---|---|
| Agent / tool use | Nanbeige4.1-3B, Qwen3-4B-Thinking-2507, Qwen3-30B-A3B-2507 | Qwen3.6-35B-A3B, Qwen3.5-9B, Qwen3.5-4B |
| Coding | Qwen3-Coder-30B-A3B (SWE-bench Verified 51.6 [I]), GLM-4.7-Flash* | Qwen3.6-35B-A3B, FrogNano-4B, MiMo-V2.6-Distill-9B |
| Research | Tongyi-DeepResearch-30B-A3B, MiroThinker-1.7-mini, DR-Venus-4B | Qwen3.6-35B-A3B, Ornith-1.5-9B |
| Law | Qwen3-30B-A3B-Thinking-2507 or GLM-4.7-Flash*, with retrieval | Qwen3.6-35B-A3B, Qwen3.5-9B |

\* After the `glm4` pre-tokenizer and template are added.

The research fine-tunes were trained on their own tool names (search, visit), so map `web_search` and `open_url` to them.

## Storage: would more disk run better models?

Mostly no: the limit is read speed, not disk space.

- **Dense models** read every weight for every token. A 14B model at Q4 manages about 3 tokens/s split across both GPUs; a 27B model about 2.
- **3B-active MoE models** at 3–4 bits (12–22 GB each) are the practical ceiling. Estimated speeds, as upper bounds from memory bandwidth:
  - 11–26 tokens/s with a fast read path from NVMe plus a GPU expert cache;
  - about 1–2.4 tokens/s with today's streaming in Chrome.
- **Fix first: streaming in Chrome.**
  - **The symptom:** the README's own numbers imply about 30 ms extra per expert layer per token in Chrome, against about 6 ms natively.
  - **Possible causes:** that looks like per-layer readbacks or read latency, not disk bandwidth.
  - **Possible fixes:** keep experts on the GPU, overlap or predict routing, and store each expert as one contiguous block.
- **Disk plans:**
  - **30 GB (today):** two MoE models at IQ3 plus a 3B model.
  - **64 GB:** three 3B-active models at Q4.
  - **128 GB:** adds Qwen3-Coder-Next (80B-A3B). gpt-oss-120b and other 120B-class models stay below about 2.5 tokens/s.

## Build order

1. Qwen3.5 (`qwen35`, `qwen35moe`), then `qwen3next`.
2. The Qwen3-VL encoder and deepstack, images in the chat, and computer-use actions in the in-app browser.
3. Faster MoE streaming in Chrome.
4. GLM-4.7-Flash (`glm4` pre-tokenizer, template); Nanbeige4.2's repeated layers; `granite`.
5. For phones: `lfm3` (LFM3-1.7B), Qwen3.5-0.8B and 2B.

## Sources

- [BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [Artificial Analysis on the Qwen3.5 small models](https://artificialanalysis.ai/articles/qwen3-5-small-models)
- [Vals LegalBench](https://www.vals.ai/benchmarks/legal_bench)
- [SWE-bench](https://github.com/SWE-bench/swe-bench.github.io)
- [Holo3.1-4B](https://huggingface.co/Hcompany/Holo-3.1-4B)
- [Fara1.5-4B](https://huggingface.co/microsoft/Fara1.5-4B)
- [GUI-Owl-1.5-4B](https://huggingface.co/mPLUG/GUI-Owl-1.5-4B-Instruct)
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B)
- [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
- [FrogNano-4B](https://huggingface.co/microsoft/FrogNano-4B-2609)
- [Tongyi-DeepResearch-30B-A3B](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)
- [Cacheable by Design? (expert caching)](https://arxiv.org/abs/2608.18261)
- [SGLang SSD expert pack](https://lmsys.org/blog/2026-08-29-sglang-ssd-expert-pack/)
- [A WGSL Gated DeltaNet](https://huggingface.co/borkiss/qwen35-2b-webgpu)
- llama.cpp `tools/mtmd/models/qwen3vl.cpp` and `src/models/qwen35.cpp`
- transformers `models/qwen3_vl`, `models/qwen3_5` and `models/qwen2_5_vl`
