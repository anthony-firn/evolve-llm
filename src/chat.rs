//! Chat prompts from the model's own Jinja template, rendered the way Hugging
//! Face renders them (trim_blocks, lstrip_blocks, Python string methods).

use crate::gguf::Gguf;
use crate::tokenizer::Tokenizer;
use minijinja::syntax::SyntaxConfig;
use minijinja::value::Serde;
use minijinja::{Environment, Error, ErrorKind, Value};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: String,
    /// Anything else the template may read (OpenAI-style `tool_calls`, `tool_call_id`, `name`),
    /// passed through unchanged.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Message {
        Message {
            role: role.into(),
            content: content.into(),
            extra: Default::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Options {
    /// Today's date as (year, month 1-12, day), for templates that print it.
    #[serde(default)]
    pub date: Option<(i32, u32, u32)>,
    /// Qwen3-style switch; passed to the template when set.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// How much a model that takes an effort setting should think (Kimi K3: low, high, max).
    #[serde(default)]
    pub thinking_effort: Option<String>,
    /// Function schemas (OpenAI style) offered to models whose templates support tools.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
}

/// ChatML, used when a model ships without a template.
const CHATML: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

pub struct ChatTemplate {
    pub source: String,
    pub from_model: bool,
    bos: String,
    eos: String,
}

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

fn strftime(fmt: &str, (y, m, d): (i32, u32, u32)) -> String {
    let month = MONTHS[(m.clamp(1, 12) - 1) as usize];
    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y % 100)),
            Some('B') => out.push_str(month),
            Some('b') => out.push_str(&month[..3]),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

/// `tojson` as transformers defines it for chat templates: Python's `json.dumps` with
/// `ensure_ascii=False` by default, `indent`, `separators` and `sort_keys`, and no HTML escaping
/// (minijinja's own escapes `<`, which would show models their tool arguments altered).
fn tojson(value: Value, kwargs: minijinja::value::Kwargs) -> Result<Value, Error> {
    let indent: Option<Value> = kwargs.get("indent")?;
    let ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let sort: Option<bool> = kwargs.get("sort_keys")?;
    let separators: Option<Vec<String>> = kwargs.get("separators")?;
    kwargs.assert_all_used()?;
    let indent = match indent {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => Some(usize::try_from(v)?),
    };
    let (item, key) = match separators.as_deref() {
        Some([a, b]) => (a.clone(), b.clone()),
        _ => ((if indent.is_some() { "," } else { ", " }).to_string(), ": ".to_string()),
    };
    let json: serde_json::Value = serde_json::to_value(&value)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("cannot serialize to JSON: {e}")))?;
    let mut out = String::new();
    py_json(&json, &PyJson { indent, item, key, sort: sort.unwrap_or(false), ascii: ascii.unwrap_or(false) }, 0, &mut out);
    Ok(Value::from_safe_string(out))
}

struct PyJson {
    indent: Option<usize>,
    item: String,
    key: String,
    sort: bool,
    ascii: bool,
}

fn py_json(v: &serde_json::Value, f: &PyJson, level: usize, out: &mut String) {
    use serde_json::Value as J;
    let string = |s: &str, out: &mut String| {
        let quoted = serde_json::to_string(s).unwrap();
        if f.ascii {
            for c in quoted.chars() {
                if c.is_ascii() {
                    out.push(c);
                } else {
                    let mut buf = [0u16; 2];
                    for unit in c.encode_utf16(&mut buf) {
                        out.push_str(&format!("\\u{unit:04x}"));
                    }
                }
            }
        } else {
            out.push_str(&quoted);
        }
    };
    let newline = |level: usize, out: &mut String| {
        if let Some(n) = f.indent {
            out.push('\n');
            out.push_str(&" ".repeat(n * level));
        }
    };
    match v {
        J::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            let mut keys: Vec<&String> = map.keys().collect();
            if f.sort {
                keys.sort();
            }
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(&f.item);
                }
                newline(level + 1, out);
                string(k, out);
                out.push_str(&f.key);
                py_json(&map[*k], f, level + 1, out);
            }
            newline(level, out);
            out.push('}');
        }
        J::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&f.item);
                }
                newline(level + 1, out);
                py_json(item, f, level + 1, out);
            }
            newline(level, out);
            out.push(']');
        }
        J::String(s) => string(s, out),
        other => out.push_str(&other.to_string()),
    }
}

impl ChatTemplate {
    pub fn from_gguf(g: &Gguf, tok: &Tokenizer) -> ChatTemplate {
        let text = |id: Option<u32>| {
            id.map(|i| tok.token_text(i).to_string())
                .unwrap_or_default()
        };
        let (source, from_model) = match g.str("tokenizer.chat_template") {
            Some(s) if !s.trim().is_empty() => (s.to_string(), true),
            _ => (CHATML.to_string(), false),
        };
        ChatTemplate {
            source,
            from_model,
            bos: text(tok.bos),
            eos: text(tok.eos),
        }
    }

    /// A template from its source text (tests and tools).
    pub fn from_source(source: &str, bos: &str, eos: &str) -> ChatTemplate {
        ChatTemplate {
            source: source.into(),
            from_model: true,
            bos: bos.into(),
            eos: eos.into(),
        }
    }

    pub fn render(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
        opts: &Options,
    ) -> Result<String, String> {
        let mut env = Environment::new();
        env.set_syntax(
            SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|e| e.to_string())?,
        );
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        let date = opts.date.unwrap_or((2026, 1, 1));
        env.add_function("strftime_now", move |fmt: String| strftime(&fmt, date));
        env.add_filter("tojson", tojson);
        env.add_template("chat", &self.source)
            .map_err(|e| format!("chat template: {e}"))?;
        let tmpl = env.get_template("chat").unwrap();
        let mut ctx = serde_json::json!({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": self.bos,
            "eos_token": self.eos,
        });
        if let Some(t) = opts.enable_thinking {
            ctx["enable_thinking"] = t.into();
            // Kimi K3 templates call it `thinking`.
            ctx["thinking"] = t.into();
        }
        if let Some(e) = &opts.thinking_effort {
            ctx["thinking_effort"] = e.clone().into();
        }
        if let Some(t) = &opts.tools {
            ctx["tools"] = t.clone();
        }
        tmpl.render(Serde(&ctx)).map_err(|e| {
            let mut msg = format!("chat template: {e}");
            if let Some(detail) = e.detail() {
                msg.push_str(&format!(" ({detail})"));
            }
            msg
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_dates_like_strftime() {
        assert_eq!(strftime("%d %b %Y", (2026, 11, 8)), "08 Nov 2026");
        assert_eq!(strftime("%B %-d", (2026, 3, 1)), "March %-d");
    }

    #[test]
    fn tojson_matches_python_json_dumps() {
        let t = ChatTemplate {
            source: "{{ x|tojson }}|{{ x|tojson(ensure_ascii=false) }}|{{ x|tojson(indent=2) }}|{{ x|tojson(separators=(',', ':'), sort_keys=true) }}|{{ 'é'|tojson(ensure_ascii=true) }}".into(),
            from_model: true,
            bos: String::new(),
            eos: String::new(),
        };
        // The value comes in as a message field.
        let t = ChatTemplate { source: t.source.replace("x|", "messages[0].extra|"), ..t };
        let m = vec![Message {
            role: "user".into(),
            content: String::new(),
            extra: serde_json::from_str(r#"{"extra": {"b": [1, "<a> é"], "a": {"c": true}}}"#).unwrap(),
        }];
        let got = t.render(&m, false, &Options::default()).unwrap();
        assert_eq!(
            got,
            "{\"b\": [1, \"<a> é\"], \"a\": {\"c\": true}}|{\"b\": [1, \"<a> é\"], \"a\": {\"c\": true}}|{\n  \"b\": [\n    1,\n    \"<a> é\"\n  ],\n  \"a\": {\n    \"c\": true\n  }\n}|{\"a\":{\"c\":true},\"b\":[1,\"<a> é\"]}|\"\\u00e9\""
        );
    }

    #[test]
    fn chatml_fallback_renders() {
        let t = ChatTemplate {
            source: CHATML.into(),
            from_model: false,
            bos: String::new(),
            eos: String::new(),
        };
        let m = vec![Message::new("user", "hi")];
        assert_eq!(
            t.render(&m, true, &Options::default()).unwrap(),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
