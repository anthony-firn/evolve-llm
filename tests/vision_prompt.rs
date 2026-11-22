//! The path a prompt with images takes through the language model: token embeddings read on the
//! CPU and given as hidden states (the embedding op skipped), with per-token rotary positions,
//! and the rotary offset kept for the tokens after an image. With plain positions it must give
//! the logits an ordinary pass gives. Needs a GPU and Qwen3 0.6B in the models directory.

use evolve_llm::gguf;
use evolve_llm::gpu::{self, engine::{Engine, Options}};
use evolve_llm::quant::Layout;
use evolve_llm::source::MemorySource;
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    std::env::var("EVOLVE_LLM_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/evolve-llm/models"))
}

fn engine(file: &str) -> Option<(Engine, Vec<u8>)> {
    let bytes = std::fs::read(models_dir().join(file)).ok()?;
    let g = gguf::parse(&bytes).unwrap();
    let gpu = pollster::block_on(gpu::init()).ok()?;
    let mut e = Engine::with_options(gpu, &g, Options { binding_limit: None, layout: Layout::Fast, budget: None, stage: None }).unwrap();
    for p in e.plan().to_vec().iter().filter(|p| p.resident) {
        e.upload(&p.name, 0, &bytes[p.offset as usize..(p.offset + p.bytes) as usize]).unwrap();
    }
    e.set_source(Box::new(MemorySource(vec![bytes.clone()])));
    e.finish(512).unwrap();
    Some((e, bytes))
}

fn worst(a: &[f32], b: &[f32]) -> (f32, f32) {
    let scale = b.iter().fold(0f32, |m, x| m.max(x.abs()));
    (a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max), scale)
}

#[test]
fn hidden_states_with_positions_match_tokens() {
    let file = "Qwen3-0.6B-Q8_0.gguf";
    let Some((e, _)) = engine(file) else {
        eprintln!("skipping: {file} or a GPU is missing");
        return;
    };
    let tokens: Vec<u32> = vec![151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198];
    let n = tokens.len();
    let want = pollster::block_on(e.forward(&tokens, 0)).unwrap();
    let hidden = e.embed_tokens(&tokens).unwrap();
    let plain: Vec<[u32; 3]> = (0..n as u32).map(|p| [p, p, p]).collect();
    let got = pollster::block_on(e.forward_hidden_positioned(&hidden, n, 0, &plain, &[])).unwrap();
    let (d, s) = worst(&got, &want);
    assert!(d <= 1e-3 * s.max(1.0), "hidden states with positions differ from tokens by {d} (logits up to {s})");

    // A rotary offset (the tokens after an image) is the same as positions shifted by it.
    e.set_rope_delta(5);
    let shifted = pollster::block_on(e.forward(&tokens, 0)).unwrap();
    e.set_rope_delta(0);
    let five: Vec<[u32; 3]> = (5..5 + n as u32).map(|p| [p, p, p]).collect();
    let explicit = pollster::block_on(e.forward_hidden_positioned(&hidden, n, 0, &five, &[])).unwrap();
    let (d, s) = worst(&shifted, &explicit);
    assert!(d <= 1e-3 * s.max(1.0), "an offset of 5 differs from positions shifted by 5 by {d}");
    let (moved, _) = worst(&shifted, &want);
    assert!(moved > 1e-2, "shifting the positions should change the logits ({moved})");
}

/// A real vision-language model sees: Qwen3-VL-2B (its vision encoder on the GPU, deepstack,
/// interleaved multimodal positions) names the colour of a square, red and then blue.
#[test]
fn qwen3vl_names_the_colour_it_sees() {
    use evolve_llm::gpu::vision::VisionGpu;
    use evolve_llm::sampler::SamplerConfig;
    use evolve_llm::session::{ImageInput, Session};
    let (text, proj) = ("Qwen3VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf");
    let (Ok(proj_bytes), Some((e, bytes))) = (std::fs::read(models_dir().join(proj)), engine(text)) else {
        eprintln!("skipping: {text} and {proj}, or a GPU, are missing");
        return;
    };
    let g = gguf::parse(&bytes).unwrap();
    let mut s = Session::new(e, &g).unwrap();
    s.set_sampler(SamplerConfig { temperature: 0.0, ..SamplerConfig::default() });
    let vgpu = pollster::block_on(gpu::init()).unwrap();
    let vision = VisionGpu::load(&vgpu, &proj_bytes).unwrap();
    let pad = s.tok.encode("<|image_pad|>", false, true);
    assert_eq!(pad.len(), 1, "one token for the image placeholder");
    for (colour, rgb) in [("red", [220u8, 30, 30]), ("blue", [30u8, 60, 220])] {
        // A square in the middle of a white 256x256 image.
        let (w, h) = (256usize, 256usize);
        let mut img = vec![255u8; w * h * 3];
        for y in 64..192 {
            for x in 64..192 {
                img[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&rgb);
            }
        }
        let cpu_cfg = vision.cfg.clone();
        let px = evolve_llm::vision::patches_for(&cpu_cfg, &img, w, h).unwrap();
        let enc = pollster::block_on(vision.encode(&px)).unwrap();
        let (rows, cols) = (px.grid_h / 2, px.grid_w / 2);
        let prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What color is the square? Answer with one word.<|im_end|>\n<|im_start|>assistant\n";
        let mut tokens = Vec::new();
        for t in s.encode_prompt(prompt) {
            if t == pad[0] {
                tokens.extend(std::iter::repeat_n(pad[0], rows * cols));
            } else {
                tokens.push(t);
            }
        }
        let image = ImageInput { embeddings: enc.embeddings, rows, cols, deepstack: enc.deepstack };
        pollster::block_on(s.prefill_images(&tokens, pad[0], &[image])).unwrap();
        let mut out = Vec::new();
        for _ in 0..6 {
            let t = pollster::block_on(s.next()).unwrap();
            if s.tok.is_end_of_generation(t) {
                break;
            }
            out.push(t);
        }
        let answer = s.tok.decode(&out);
        eprintln!("{colour} square: {answer:?}");
        assert!(answer.to_lowercase().contains(colour), "asked about a {colour} square, the model said {answer:?}");
        s.reset();
    }
}
