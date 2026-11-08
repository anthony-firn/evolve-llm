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
    pub content: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Options {
    /// Today's date as (year, month 1-12, day), for templates that print it.
    #[serde(default)]
    pub date: Option<(i32, u32, u32)>,
    /// Qwen3-style switch; passed to the template when set.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
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
    fn chatml_fallback_renders() {
        let t = ChatTemplate {
            source: CHATML.into(),
            from_model: false,
            bos: String::new(),
            eos: String::new(),
        };
        let m = vec![Message {
            role: "user".into(),
            content: "hi".into(),
        }];
        assert_eq!(
            t.render(&m, true, &Options::default()).unwrap(),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
