//! GGUF container parsing: header, metadata and tensor directory.
//!
//! The parser works on a prefix of the file. When the prefix is too short it
//! reports how many bytes it needs, so a browser can read a model's header
//! without loading gigabytes of weights into memory.

use std::collections::BTreeMap;
use std::fmt;

const MAGIC: u32 = 0x4655_4747; // "GGUF" little endian
const MAX_STRING: u64 = 1 << 26;
const MAX_ARRAY: u64 = 1 << 26;
const MAX_TENSORS: u64 = 1 << 20;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => self.as_u64().map(|v| v != 0),
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// GGML order: `dims[0]` is the contiguous (row length) dimension.
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Offset relative to the start of the tensor data section.
    pub offset: u64,
    /// File of a split model that holds the tensor (0 for single files).
    pub file: usize,
    /// Start of that file's tensor data section.
    pub base: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Byte offset of the tensor within its file.
    pub fn start(&self) -> u64 {
        self.base + self.offset
    }
}

#[derive(Debug, Clone)]
pub struct Gguf {
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    /// Absolute file offset where tensor data begins.
    pub data_offset: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// The prefix ended early; read at least this many bytes and try again.
    NeedMore(u64),
    Invalid(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::NeedMore(n) => write!(f, "GGUF header needs {n} bytes"),
            ParseError::Invalid(msg) => write!(f, "invalid GGUF file: {msg}"),
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: u64) -> Result<&'a [u8], ParseError> {
        let end = (self.pos as u64)
            .checked_add(n)
            .ok_or_else(|| ParseError::Invalid("length overflow".into()))?;
        if end > self.bytes.len() as u64 {
            // Ask for the missing part plus headroom, so large vocabularies take few retries.
            let want = end.max(self.bytes.len() as u64 * 2).max(1 << 20);
            return Err(ParseError::NeedMore(want));
        }
        let start = self.pos;
        self.pos = end as usize;
        Ok(&self.bytes[start..end as usize])
    }

    fn u8(&mut self) -> Result<u8, ParseError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, ParseError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, ParseError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, ParseError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, ParseError> {
        let len = self.u64()?;
        if len > MAX_STRING {
            return Err(ParseError::Invalid(format!("string of {len} bytes")));
        }
        let raw = self.take(len)?;
        // Tokenizer vocabularies occasionally contain invalid UTF-8; keep them lossless enough to load.
        Ok(match std::str::from_utf8(raw) {
            Ok(s) => s.to_owned(),
            Err(_) => String::from_utf8_lossy(raw).into_owned(),
        })
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<Value, ParseError> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                if depth > 2 {
                    return Err(ParseError::Invalid("nested arrays too deep".into()));
                }
                let inner = self.u32()?;
                let len = self.u64()?;
                if len > MAX_ARRAY {
                    return Err(ParseError::Invalid(format!("array of {len} items")));
                }
                let mut items = Vec::with_capacity(len.min(1 << 20) as usize);
                for _ in 0..len {
                    items.push(self.value(inner, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => {
                return Err(ParseError::Invalid(format!(
                    "unknown metadata type {other}"
                )))
            }
        })
    }
}

/// Parse a GGUF header from the start of a file.
pub fn parse(bytes: &[u8]) -> Result<Gguf, ParseError> {
    let mut r = Reader { bytes, pos: 0 };
    if r.u32()? != MAGIC {
        return Err(ParseError::Invalid("not a GGUF file (bad magic)".into()));
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(ParseError::Invalid(format!(
            "unsupported GGUF version {version}"
        )));
    }
    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;
    if tensor_count > MAX_TENSORS || kv_count > MAX_TENSORS {
        return Err(ParseError::Invalid("implausible header counts".into()));
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = r.string()?;
        let ty = r.u32()?;
        let value = r.value(ty, 0)?;
        metadata.insert(key, value);
    }
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()?;
        if n_dims == 0 || n_dims > 4 {
            return Err(ParseError::Invalid(format!(
                "tensor {name} has {n_dims} dimensions"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let ggml_type = r.u32()?;
        let offset = r.u64()?;
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
            file: 0,
            base: 0,
        });
    }
    let alignment = metadata
        .get("general.alignment")
        .and_then(Value::as_u64)
        .unwrap_or(32);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(ParseError::Invalid(format!("bad alignment {alignment}")));
    }
    let pos = r.pos as u64;
    let data_offset = pos.div_ceil(alignment) * alignment;
    for t in &mut tensors {
        t.base = data_offset;
    }
    Ok(Gguf {
        version,
        metadata,
        tensors,
        alignment,
        data_offset,
    })
}

impl Gguf {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }
    pub fn u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }
    pub fn f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(Value::as_f64)
    }
    pub fn str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }
    pub fn bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(Value::as_bool)
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn tensor_index(&self, name: &str) -> Option<usize> {
        self.tensors.iter().position(|t| t.name == name)
    }

    /// Number of files of a split model (`model-00001-of-00004.gguf`), from the first file.
    pub fn split_count(&self) -> usize {
        self.u64("split.count").unwrap_or(1).max(1) as usize
    }

    /// Join the headers of every file of a split model, in order, into one directory whose
    /// tensors remember their file. Metadata comes from the first file.
    pub fn merge(parts: Vec<Gguf>) -> Result<Gguf, String> {
        let mut it = parts.into_iter();
        let mut first = it.next().ok_or("no files")?;
        let count = first.split_count();
        for (i, part) in it.enumerate() {
            let file = i + 1;
            if let Some(n) = part.u64("split.no") {
                if n as usize != file {
                    return Err(format!("file {} is split part {} of the model", file + 1, n + 1));
                }
            }
            for mut t in part.tensors {
                t.file = file;
                first.tensors.push(t);
            }
        }
        if count > 1 {
            let have = first.tensors.iter().map(|t| t.file).max().unwrap_or(0) + 1;
            if have != count {
                return Err(format!("the model has {count} parts but {have} were given"));
            }
        }
        Ok(first)
    }
}

/// For `name-00001-of-00004.gguf`, the names of all parts; None for a single file.
pub fn split_names(name: &str) -> Option<Vec<String>> {
    let stem = name.strip_suffix(".gguf")?;
    let (head, total) = stem.rsplit_once("-of-")?;
    let (prefix, index) = head.rsplit_once('-')?;
    let n: usize = total.parse().ok()?;
    if index.len() != total.len() || index.parse::<usize>().ok()? == 0 || n < 2 {
        return None;
    }
    let w = total.len();
    Some((1..=n).map(|i| format!("{prefix}-{i:0w$}-of-{total}.gguf")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_names_follow_llama_cpp() {
        let n = split_names("Kimi-K2-UD-IQ1_S-00002-of-00005.gguf").unwrap();
        assert_eq!(n.len(), 5);
        assert_eq!(n[0], "Kimi-K2-UD-IQ1_S-00001-of-00005.gguf");
        assert_eq!(n[4], "Kimi-K2-UD-IQ1_S-00005-of-00005.gguf");
        assert!(split_names("model.gguf").is_none());
        assert!(split_names("qwen-of-the-year.gguf").is_none());
    }

    fn push_str(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    fn sample() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend(MAGIC.to_le_bytes());
        b.extend(3u32.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        push_str(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        push_str(&mut b, "llama");
        push_str(&mut b, "list");
        b.extend(9u32.to_le_bytes());
        b.extend(5u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend((-3i32).to_le_bytes());
        b.extend(7i32.to_le_bytes());
        push_str(&mut b, "w");
        b.extend(2u32.to_le_bytes());
        b.extend(32u64.to_le_bytes());
        b.extend(4u64.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b
    }

    #[test]
    fn parses_header_and_alignment() {
        let bytes = sample();
        let g = parse(&bytes).unwrap();
        assert_eq!(g.str("general.architecture"), Some("llama"));
        assert_eq!(
            g.get("list").unwrap().as_array().unwrap()[0],
            Value::I32(-3)
        );
        assert_eq!(g.tensors[0].dims, vec![32, 4]);
        assert_eq!(g.data_offset % 32, 0);
        assert!(g.data_offset >= bytes.len() as u64);
    }

    #[test]
    fn short_prefix_asks_for_more() {
        let bytes = sample();
        match parse(&bytes[..bytes.len() - 3]) {
            Err(ParseError::NeedMore(n)) => assert!(n >= bytes.len() as u64),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_other_files() {
        assert!(matches!(
            parse(b"PK\x03\x04aaaaaaaaaaaaaaaaaaaaaaaa"),
            Err(ParseError::Invalid(_))
        ));
    }
}
