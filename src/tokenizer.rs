//! Tokenizers read from GGUF metadata: byte-level BPE (GPT-2 style, used by
//! Llama 3, Qwen and SmolLM) and SentencePiece-style BPE (Llama 2, TinyLlama).

use crate::gguf::{Gguf, Value};
use fancy_regex::Regex;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use unicode_normalization::UnicodeNormalization;

const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

const NORMAL: i32 = 1;
const UNKNOWN: i32 = 2;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const BYTE: i32 = 6;

/// Texts that end a turn when they are special tokens.
const END_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|eom_id|>",
    "<|im_end|>",
    "<|end|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<EOT>",
    "</s>",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Bpe,
    Spm,
}

pub struct Tokenizer {
    pub kind: Kind,
    tokens: Vec<String>,
    types: Vec<i32>,
    scores: Vec<f32>,
    ids: HashMap<String, u32>,
    merges: HashMap<(u32, u32), (u32, u32)>,
    specials: Vec<(String, u32)>,
    special_first: [bool; 256],
    pre: Vec<Regex>,
    nfc: bool,
    ignore_merges: bool,
    add_space_prefix: bool,
    byte_char: [char; 256],
    char_byte: HashMap<char, u8>,
    byte_token: [Option<u32>; 256],
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub add_bos: bool,
    eog: HashSet<u32>,
}

fn string_array<'a>(g: &'a Gguf, key: &str) -> Option<Vec<&'a str>> {
    g.get(key)?.as_array()?.iter().map(Value::as_str).collect()
}

fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
    }
    table
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, String> {
        let model = g
            .str("tokenizer.ggml.model")
            .ok_or("missing tokenizer.ggml.model")?;
        let kind = match model {
            "gpt2" => Kind::Bpe,
            "llama" => Kind::Spm,
            other => return Err(format!("the {other} tokenizer is not supported yet")),
        };
        let tokens: Vec<String> = string_array(g, "tokenizer.ggml.tokens")
            .ok_or("missing tokenizer.ggml.tokens")?
            .into_iter()
            .map(str::to_owned)
            .collect();
        let n = tokens.len();
        let types: Vec<i32> = match g.get("tokenizer.ggml.token_type").and_then(Value::as_array) {
            Some(a) => a.iter().map(|v| v.as_f64().unwrap_or(1.0) as i32).collect(),
            None => vec![NORMAL; n],
        };
        let scores: Vec<f32> = match g.get("tokenizer.ggml.scores").and_then(Value::as_array) {
            Some(a) => a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect(),
            None => vec![0.0; n],
        };
        if types.len() != n || scores.len() != n {
            return Err("tokenizer arrays have different lengths".into());
        }
        let mut ids = HashMap::with_capacity(n);
        for (i, t) in tokens.iter().enumerate() {
            ids.entry(t.clone()).or_insert(i as u32);
        }
        let mut merges = HashMap::new();
        if kind == Kind::Bpe {
            let list =
                string_array(g, "tokenizer.ggml.merges").ok_or("missing tokenizer.ggml.merges")?;
            for (rank, m) in list.iter().enumerate() {
                let Some((a, b)) = m.split_once(' ') else {
                    continue;
                };
                let (Some(&ia), Some(&ib)) = (ids.get(a), ids.get(b)) else {
                    continue;
                };
                let Some(&merged) = ids.get(&format!("{a}{b}")) else {
                    continue;
                };
                merges.entry((ia, ib)).or_insert((rank as u32, merged));
            }
        }
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| matches!(types[*i], CONTROL | USER_DEFINED) && !t.is_empty())
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        let mut special_first = [false; 256];
        for (t, _) in &specials {
            special_first[t.as_bytes()[0] as usize] = true;
        }

        let pre_name = g.str("tokenizer.ggml.pre").unwrap_or("default");
        let (patterns, nfc, ignore_merges): (Vec<&str>, bool, bool) = match pre_name {
            "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "pixtral" => {
                (vec![LLAMA3], false, true)
            }
            "qwen2" | "deepseek-r1-qwen" | "qwen35" => (vec![QWEN2], true, false),
            "smollm" => (vec![r"\p{N}", GPT2], false, false),
            _ => (vec![GPT2], false, false),
        };
        let pre = if kind == Kind::Bpe {
            patterns
                .iter()
                .map(|p| Regex::new(p).map_err(|e| e.to_string()))
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };
        let byte_char = bytes_to_unicode();
        let char_byte = byte_char
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        let mut byte_token = [None; 256];
        if kind == Kind::Spm {
            for b in 0..256usize {
                byte_token[b] = ids.get(&format!("<0x{b:02X}>")).copied();
            }
        }
        let id = |k: &str| g.u64(k).map(|v| v as u32).filter(|&v| (v as usize) < n);
        let bos = id("tokenizer.ggml.bos_token_id");
        let eos = id("tokenizer.ggml.eos_token_id");
        let default_bos = kind == Kind::Spm || ignore_merges;
        let add_bos = g
            .bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(default_bos)
            && bos.is_some();
        let mut eog: HashSet<u32> = [
            eos,
            id("tokenizer.ggml.eot_token_id"),
            id("tokenizer.ggml.eom_token_id"),
        ]
        .into_iter()
        .flatten()
        .collect();
        for (text, i) in &specials {
            if END_TEXTS.contains(&text.as_str()) {
                eog.insert(*i);
            }
        }
        Ok(Tokenizer {
            kind,
            tokens,
            types,
            scores,
            ids,
            merges,
            specials,
            special_first,
            pre,
            nfc,
            ignore_merges,
            add_space_prefix: g.bool("tokenizer.ggml.add_space_prefix").unwrap_or(true),
            byte_char,
            char_byte,
            byte_token,
            bos,
            eos,
            add_bos,
            eog,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_text(&self, id: u32) -> &str {
        self.tokens
            .get(id as usize)
            .map(String::as_str)
            .unwrap_or("")
    }

    pub fn is_end_of_generation(&self, id: u32) -> bool {
        self.eog.contains(&id)
    }

    pub fn end_tokens(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.eog.iter().copied().collect();
        v.sort();
        v
    }

    /// Encode text. Special-token texts such as `<|im_start|>` become their
    /// single token when `parse_special` is set.
    pub fn encode(&self, text: &str, add_bos: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos {
            if let Some(b) = self.bos {
                out.push(b);
            }
        }
        // SentencePiece models prefix a space at the start and after each special token.
        let mut after_special = true;
        for (segment, special) in self.split_specials(text, parse_special) {
            match special {
                Some(id) => out.push(id),
                None => match self.kind {
                    Kind::Bpe => self.encode_bpe(segment, &mut out),
                    Kind::Spm => self.encode_spm(segment, after_special, &mut out),
                },
            }
            after_special = special.is_some();
        }
        out
    }

    fn split_specials<'a>(&self, text: &'a str, parse: bool) -> Vec<(&'a str, Option<u32>)> {
        if !parse || self.specials.is_empty() {
            return vec![(text, None)];
        }
        let mut parts = Vec::new();
        let mut start = 0;
        let mut i = 0;
        let bytes = text.as_bytes();
        while i < bytes.len() {
            if !text.is_char_boundary(i) {
                i += 1;
                continue;
            }
            if !self.special_first[bytes[i] as usize] {
                i += 1;
                continue;
            }
            let rest = &text[i..];
            if let Some((tok, id)) = self
                .specials
                .iter()
                .find(|(t, _)| rest.starts_with(t.as_str()))
            {
                if start < i {
                    parts.push((&text[start..i], None));
                }
                parts.push((&text[i..i + tok.len()], Some(*id)));
                i += tok.len();
                start = i;
            } else {
                i += 1;
            }
        }
        if start < bytes.len() {
            parts.push((&text[start..], None));
        }
        parts
    }

    fn encode_bpe(&self, text: &str, out: &mut Vec<u32>) {
        let normalized: String;
        let text = if self.nfc {
            normalized = text.nfc().collect();
            &normalized
        } else {
            text
        };
        let mut pieces = vec![text];
        for re in &self.pre {
            let mut next = Vec::new();
            for piece in pieces {
                let mut last = 0;
                for m in re.find_iter(piece).flatten() {
                    if m.start() > last {
                        next.push(&piece[last..m.start()]);
                    }
                    if m.end() > m.start() {
                        next.push(m.as_str());
                    }
                    last = m.end();
                }
                if last < piece.len() {
                    next.push(&piece[last..]);
                }
            }
            pieces = next;
        }
        for piece in pieces {
            let mapped: String = piece.bytes().map(|b| self.byte_char[b as usize]).collect();
            if self.ignore_merges {
                if let Some(&id) = self.ids.get(&mapped) {
                    out.push(id);
                    continue;
                }
            }
            self.bpe_merge(&mapped, out);
        }
    }

    fn bpe_merge(&self, mapped: &str, out: &mut Vec<u32>) {
        struct Sym {
            id: u32,
            prev: isize,
            next: isize,
            alive: bool,
        }
        let mut syms: Vec<Sym> = Vec::new();
        for c in mapped.chars() {
            let mut buf = [0u8; 4];
            match self.ids.get(c.encode_utf8(&mut buf) as &str) {
                Some(&id) => {
                    let i = syms.len() as isize;
                    syms.push(Sym {
                        id,
                        prev: i - 1,
                        next: i + 1,
                        alive: true,
                    });
                }
                None => {
                    // Byte-level vocabularies contain all 256 byte symbols; skip anything else.
                    continue;
                }
            }
        }
        if syms.is_empty() {
            return;
        }
        let last = syms.len() - 1;
        syms[last].next = -1;
        // Min-heap over (rank, left position).
        let mut heap: BinaryHeap<Reverse<(u32, usize, u32, u32)>> = BinaryHeap::new();
        for i in 0..syms.len() - 1 {
            if let Some(&(rank, _)) = self.merges.get(&(syms[i].id, syms[i + 1].id)) {
                heap.push(Reverse((rank, i, syms[i].id, syms[i + 1].id)));
            }
        }
        while let Some(Reverse((_, left, a, b))) = heap.pop() {
            if !syms[left].alive || syms[left].id != a || syms[left].next < 0 {
                continue;
            }
            let right = syms[left].next as usize;
            if !syms[right].alive || syms[right].id != b {
                continue;
            }
            let (_, merged) = self.merges[&(a, b)];
            syms[left].id = merged;
            syms[right].alive = false;
            let after = syms[right].next;
            syms[left].next = after;
            if after >= 0 {
                syms[after as usize].prev = left as isize;
                if let Some(&(rank, _)) = self.merges.get(&(merged, syms[after as usize].id)) {
                    heap.push(Reverse((rank, left, merged, syms[after as usize].id)));
                }
            }
            let before = syms[left].prev;
            if before >= 0 {
                if let Some(&(rank, _)) = self.merges.get(&(syms[before as usize].id, merged)) {
                    heap.push(Reverse((
                        rank,
                        before as usize,
                        syms[before as usize].id,
                        merged,
                    )));
                }
            }
        }
        let mut i = 0isize;
        while i >= 0 {
            out.push(syms[i as usize].id);
            i = syms[i as usize].next;
        }
    }

    fn encode_spm(&self, text: &str, first: bool, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let mut s = String::with_capacity(text.len() + 3);
        if self.add_space_prefix && first {
            s.push('\u{2581}');
        }
        for c in text.chars() {
            s.push(if c == ' ' { '\u{2581}' } else { c });
        }
        struct Sym {
            start: usize,
            len: usize,
            prev: isize,
            next: isize,
        }
        let mut syms: Vec<Sym> = Vec::new();
        for (i, (pos, c)) in s.char_indices().enumerate() {
            syms.push(Sym {
                start: pos,
                len: c.len_utf8(),
                prev: i as isize - 1,
                next: i as isize + 1,
            });
        }
        let last = syms.len() - 1;
        syms[last].next = -1;

        #[derive(PartialEq)]
        struct Bigram {
            score: f32,
            left: usize,
            right: usize,
            size: usize,
        }
        impl Eq for Bigram {}
        impl PartialOrd for Bigram {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Bigram {
            fn cmp(&self, o: &Self) -> Ordering {
                self.score.total_cmp(&o.score).then(o.left.cmp(&self.left))
            }
        }
        let mut heap = BinaryHeap::new();
        let mut rev: HashMap<(usize, usize), (usize, usize, usize, usize)> = HashMap::new();
        let try_add = |syms: &Vec<Sym>,
                       heap: &mut BinaryHeap<Bigram>,
                       rev: &mut HashMap<_, _>,
                       l: isize,
                       r: isize| {
            if l < 0 || r < 0 {
                return;
            }
            let (l, r) = (l as usize, r as usize);
            let start = syms[l].start;
            let size = syms[l].len + syms[r].len;
            let piece = &s[start..start + size];
            if let Some(&id) = self.ids.get(piece) {
                heap.push(Bigram {
                    score: self.scores[id as usize],
                    left: l,
                    right: r,
                    size,
                });
                rev.insert(
                    (start, size),
                    (syms[l].start, syms[l].len, syms[r].start, syms[r].len),
                );
            }
        };
        for i in 1..syms.len() {
            try_add(&syms, &mut heap, &mut rev, i as isize - 1, i as isize);
        }
        while let Some(bg) = heap.pop() {
            let (l, r) = (bg.left, bg.right);
            if syms[l].len == 0 || syms[r].len == 0 || syms[l].len + syms[r].len != bg.size {
                continue;
            }
            syms[l].len += syms[r].len;
            syms[r].len = 0;
            syms[l].next = syms[r].next;
            if syms[r].next >= 0 {
                let n = syms[r].next as usize;
                syms[n].prev = l as isize;
            }
            let (p, nx) = (syms[l].prev, syms[l].next);
            try_add(&syms, &mut heap, &mut rev, p, l as isize);
            try_add(&syms, &mut heap, &mut rev, l as isize, nx);
        }
        let mut i = 0isize;
        while i >= 0 {
            let sym = &syms[i as usize];
            self.spm_resegment(&s, sym.start, sym.len, &rev, out);
            i = sym.next;
        }
    }

    fn spm_resegment(
        &self,
        s: &str,
        start: usize,
        len: usize,
        rev: &HashMap<(usize, usize), (usize, usize, usize, usize)>,
        out: &mut Vec<u32>,
    ) {
        let piece = &s[start..start + len];
        if let Some(&id) = self.ids.get(piece) {
            out.push(id);
            return;
        }
        if let Some(&(ls, ll, rs, rl)) = rev.get(&(start, len)) {
            self.spm_resegment(s, ls, ll, rev, out);
            self.spm_resegment(s, rs, rl, rev, out);
            return;
        }
        for b in piece.bytes() {
            if let Some(id) = self.byte_token[b as usize] {
                out.push(id);
            }
        }
    }

    /// Bytes a token contributes to the output text.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(text) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        let ty = self.types[id as usize];
        match (self.kind, ty) {
            (_, CONTROL) => Vec::new(),
            (_, USER_DEFINED) => text.as_bytes().to_vec(),
            (Kind::Bpe, _) => text
                .chars()
                .filter_map(|c| self.char_byte.get(&c).copied())
                .collect(),
            (Kind::Spm, BYTE) => {
                let hex = text.trim_start_matches("<0x").trim_end_matches('>');
                u8::from_str_radix(hex, 16)
                    .map(|b| vec![b])
                    .unwrap_or_default()
            }
            (Kind::Spm, UNKNOWN) => "\u{fffd}".as_bytes().to_vec(),
            (Kind::Spm, _) => text.replace('\u{2581}', " ").into_bytes(),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| self.token_bytes(i)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Turns a token stream into text, holding back incomplete UTF-8 sequences.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        self.pending.extend(tok.token_bytes(id));
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) => {
                if e.error_len().is_some() {
                    // Invalid bytes, not just a truncated sequence: emit them replaced.
                    let s = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    return s;
                }
                e.valid_up_to()
            }
        };
        let out = String::from_utf8(self.pending[..valid].to_vec()).unwrap();
        self.pending.drain(..valid);
        out
    }

    pub fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}
