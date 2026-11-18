//! Each tool-capable model family's chat template (tests/fixtures/templates) renders a
//! conversation with a tool call and its result. The call it renders for history is the syntax
//! the model writes; it is kept in tests/fixtures/tool-call-renders.json, which
//! tests/tool-formats.test.mjs parses back with the page's parser (web/tools.js).
//!
//! `EVOLVE_LLM_WRITE_FIXTURES=1 cargo test --test tool_templates` rewrites that file.

use evolve_llm::chat::{ChatTemplate, Message, Options};
use serde_json::{json, Value};
use std::path::PathBuf;

/// (fixture name, BOS text, EOS text)
const FAMILIES: &[(&str, &str, &str)] = &[
    ("qwen2.5", "", "<|im_end|>"),
    ("qwen3", "", "<|im_end|>"),
    ("qwen3-coder", "", "<|im_end|>"),
    ("llama3.2", "<|begin_of_text|>", "<|eot_id|>"),
    ("deepseek-v3.1", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"),
    ("kimi-k2", "[BOS]", "[EOS]"),
    ("kimi-k3.1", "[BOS]", "[EOS]"),
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn tools() -> Value {
    json!([{ "type": "function", "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": { "type": "object", "properties": {
            "query": { "type": "string", "description": "What to search for" },
            "limit": { "type": "integer", "description": "Results to return" }
        }, "required": ["query"] }
    } }])
}

fn message(v: Value) -> Message {
    serde_json::from_value(v).unwrap()
}

fn call(args: Value) -> Message {
    message(json!({ "role": "assistant", "content": "", "tool_calls": [
        { "id": "functions.web_search:0", "type": "function", "function": { "name": "web_search", "arguments": args } }
    ] }))
}

#[test]
fn templates_render_tool_calls_and_results() {
    let user = Message::new("user", "What is WebGPU?");
    let result = message(json!({ "role": "tool", "tool_call_id": "functions.web_search:0", "name": "web_search", "content": "RESULT TEXT" }));
    let opts = Options { tools: Some(tools()), enable_thinking: Some(false), ..Default::default() };
    let args = json!({ "query": "WebGPU spec", "limit": 3 });
    let mut renders = serde_json::Map::new();
    for (name, bos, eos) in FAMILIES {
        let source = std::fs::read_to_string(root().join(format!("tests/fixtures/templates/{name}.jinja"))).unwrap();
        let t = ChatTemplate::from_source(&source, bos, eos);
        // Arguments as an object, or as JSON text for templates that concatenate them (DeepSeek).
        let (asst, full) = match t.render(&[user.clone(), call(args.clone()), result.clone()], true, &opts) {
            Ok(full) => (call(args.clone()), full),
            Err(_) => {
                let a = call(Value::String(args.to_string()));
                let full = t
                    .render(&[user.clone(), a.clone(), result.clone()], true, &opts)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                (a, full)
            }
        };
        assert!(full.contains("RESULT TEXT"), "{name}: the tool result is missing:\n{full}");
        assert!(full.contains("web_search"), "{name}: the tools are missing:\n{full}");
        // The call as the template writes it: what the assistant turn adds to the prompt.
        let before = t.render(std::slice::from_ref(&user), true, &opts).unwrap();
        let with = t.render(&[user.clone(), asst], false, &opts).unwrap();
        let common = before.chars().zip(with.chars()).take_while(|(a, b)| a == b).count();
        let segment: String = with.chars().skip(common).collect();
        assert!(segment.contains("WebGPU spec"), "{name}: the call is missing:\n{segment}");
        renders.insert(name.to_string(), Value::String(segment));
    }
    let path = root().join("tests/fixtures/tool-call-renders.json");
    let got = serde_json::to_string_pretty(&Value::Object(renders)).unwrap() + "\n";
    if std::env::var("EVOLVE_LLM_WRITE_FIXTURES").is_ok() {
        std::fs::write(&path, &got).unwrap();
    }
    assert_eq!(std::fs::read_to_string(&path).unwrap_or_default(), got, "rerun with EVOLVE_LLM_WRITE_FIXTURES=1");
}
