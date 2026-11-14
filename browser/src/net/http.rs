//! HTTP/1.1 wire format: request serialization, response head parsing, body framing and
//! content decoding. Everything here is pure so it can be unit tested natively.

use std::io::Read;

pub const MAX_HEAD: usize = 256 * 1024;
pub const MAX_BODY: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct ResponseHead {
    pub status: u16,
    pub reason: String,
    pub headers: Vec<(String, String)>,
}

impl ResponseHead {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    pub fn all<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.headers
            .iter()
            .filter(move |(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Serialize a request. `target` is the origin-form path and query.
pub fn build_request(method: &str, target: &str, host: &str, headers: &[(String, String)], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(512 + body.len());
    out.extend_from_slice(method.as_bytes());
    out.push(b' ');
    out.extend_from_slice(if target.is_empty() { b"/" } else { target.as_bytes() });
    out.extend_from_slice(b" HTTP/1.1\r\nHost: ");
    out.extend_from_slice(host.as_bytes());
    out.extend_from_slice(b"\r\n");
    let mut has_length = false;
    for (name, value) in headers {
        if name.eq_ignore_ascii_case("host") || name.contains(['\r', '\n', ':']) || value.contains(['\r', '\n']) {
            continue;
        }
        if name.eq_ignore_ascii_case("content-length") {
            has_length = true;
        }
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b": ");
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(b"\r\n");
    }
    let needs_length = !body.is_empty() || matches!(method, "POST" | "PUT" | "PATCH");
    if needs_length && !has_length {
        out.extend_from_slice(format!("Content-Length: {}\r\n", body.len()).as_bytes());
    }
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(body);
    out
}

/// Parse a response head from the start of `buf`. Returns the head and its length, or `None`
/// when more bytes are needed.
pub fn parse_head(buf: &[u8]) -> Result<Option<(ResponseHead, usize)>, String> {
    let mut headers = vec![httparse::EMPTY_HEADER; 256];
    let mut response = httparse::Response::new(&mut headers);
    match response.parse(buf) {
        Ok(httparse::Status::Complete(len)) => {
            let head = ResponseHead {
                status: response.code.unwrap_or(0),
                reason: response.reason.unwrap_or("").to_string(),
                headers: response
                    .headers
                    .iter()
                    .map(|h| (h.name.to_string(), String::from_utf8_lossy(h.value).into_owned()))
                    .collect(),
            };
            Ok(Some((head, len)))
        }
        Ok(httparse::Status::Partial) if buf.len() > MAX_HEAD => Err("Response headers are too large".into()),
        Ok(httparse::Status::Partial) => Ok(None),
        Err(err) => Err(format!("Invalid HTTP response: {err}")),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Framing {
    /// No body (HEAD, 1xx, 204, 304)
    None,
    Length(usize),
    Chunked,
    /// Read until the server closes the connection
    Close,
}

pub fn framing(method: &str, head: &ResponseHead) -> Result<Framing, String> {
    if method == "HEAD" || (100..200).contains(&head.status) || head.status == 204 || head.status == 304 {
        return Ok(Framing::None);
    }
    if head
        .header("transfer-encoding")
        .is_some_and(|v| v.split(',').any(|t| t.trim().eq_ignore_ascii_case("chunked")))
    {
        return Ok(Framing::Chunked);
    }
    if let Some(len) = head.header("content-length") {
        let len: usize = len.trim().parse().map_err(|_| "Invalid Content-Length".to_string())?;
        if len > MAX_BODY {
            return Err("Response body is too large".into());
        }
        return Ok(Framing::Length(len));
    }
    Ok(Framing::Close)
}

/// Whether the connection may be reused after this response.
pub fn keep_alive(head: &ResponseHead, framing: Framing) -> bool {
    if framing == Framing::Close {
        return false;
    }
    !head
        .header("connection")
        .is_some_and(|v| v.split(',').any(|t| t.trim().eq_ignore_ascii_case("close")))
}

/// Incremental decoder for `Transfer-Encoding: chunked`.
#[derive(Debug, Default)]
pub struct ChunkedDecoder {
    state: ChunkState,
    remaining: usize,
    line: Vec<u8>,
    pub done: bool,
}

#[derive(Debug, Default, PartialEq)]
enum ChunkState {
    #[default]
    Size,
    Data,
    DataEnd,
    Trailer,
}

impl ChunkedDecoder {
    /// Feed bytes. Decoded body bytes are appended to `out`. Returns the number of input bytes
    /// consumed; any bytes after the final chunk are left unconsumed.
    pub fn feed(&mut self, mut input: &[u8], out: &mut Vec<u8>) -> Result<usize, String> {
        let start = input.len();
        while !input.is_empty() && !self.done {
            match self.state {
                ChunkState::Size | ChunkState::Trailer | ChunkState::DataEnd => {
                    let Some(pos) = input.iter().position(|&b| b == b'\n') else {
                        self.line.extend_from_slice(input);
                        input = &[];
                        if self.line.len() > 8192 {
                            return Err("Chunk header is too long".into());
                        }
                        break;
                    };
                    self.line.extend_from_slice(&input[..pos]);
                    input = &input[pos + 1..];
                    let line = std::mem::take(&mut self.line);
                    let line = String::from_utf8_lossy(&line);
                    let line = line.trim_end_matches('\r');
                    match self.state {
                        ChunkState::Size => {
                            let size = line.split(';').next().unwrap_or("").trim();
                            let size = usize::from_str_radix(size, 16).map_err(|_| format!("Invalid chunk size {size:?}"))?;
                            if size == 0 {
                                self.state = ChunkState::Trailer;
                            } else {
                                if out.len() + size > MAX_BODY {
                                    return Err("Response body is too large".into());
                                }
                                self.remaining = size;
                                self.state = ChunkState::Data;
                            }
                        }
                        ChunkState::DataEnd => self.state = ChunkState::Size,
                        ChunkState::Trailer => {
                            if line.is_empty() {
                                self.done = true;
                            }
                        }
                        ChunkState::Data => unreachable!(),
                    }
                }
                ChunkState::Data => {
                    let take = self.remaining.min(input.len());
                    out.extend_from_slice(&input[..take]);
                    input = &input[take..];
                    self.remaining -= take;
                    if self.remaining == 0 {
                        self.state = ChunkState::DataEnd;
                    }
                }
            }
        }
        Ok(start - input.len())
    }
}

/// Decode a body according to its `Content-Encoding`.
pub fn decode_content(encoding: Option<&str>, body: Vec<u8>) -> Result<Vec<u8>, String> {
    let Some(encoding) = encoding else { return Ok(body) };
    let mut data = body;
    // Encodings are listed in the order they were applied.
    for coding in encoding.split(',').map(|c| c.trim().to_ascii_lowercase()).rev() {
        data = match coding.as_str() {
            "" | "identity" => data,
            "gzip" | "x-gzip" => read_all(flate2::read::MultiGzDecoder::new(&data[..]))?,
            "deflate" => read_all(flate2::read::ZlibDecoder::new(&data[..]))
                .or_else(|_| read_all(flate2::read::DeflateDecoder::new(&data[..])))?,
            "br" => read_all(brotli_decompressor::Decompressor::new(&data[..], 4096))?,
            other => return Err(format!("Unsupported content encoding {other}")),
        };
    }
    Ok(data)
}

fn read_all(mut reader: impl Read) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    reader
        .by_ref()
        .take(MAX_BODY as u64 + 1)
        .read_to_end(&mut out)
        .map_err(|e| format!("Could not decode the response body: {e}"))?;
    if out.len() > MAX_BODY {
        return Err("Decoded response body is too large".into());
    }
    Ok(out)
}

/// Parse a `data:` URL into its media type and bytes.
pub fn parse_data_url(url: &str) -> Result<(String, Vec<u8>), String> {
    let rest = url.strip_prefix("data:").ok_or("Not a data URL")?;
    let (meta, data) = rest.split_once(',').ok_or("Invalid data URL")?;
    let data = data.split('#').next().unwrap_or("");
    let is_base64 = meta.to_ascii_lowercase().ends_with(";base64");
    let mime = meta.trim_end_matches(";base64").trim_end_matches(";BASE64");
    let mime = if mime.is_empty() { "text/plain;charset=US-ASCII".to_string() } else { mime.to_string() };
    let decoded: Vec<u8> = percent_encoding::percent_decode_str(data).collect();
    if is_base64 {
        let text: String = String::from_utf8_lossy(&decoded).chars().filter(|c| !c.is_whitespace()).collect();
        Ok((mime, base64_decode(&text)?))
    } else {
        Ok((mime, decoded))
    }
}

pub fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        } as u32)
    }
    let bytes: Vec<u8> = input.bytes().filter(|&b| b != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut acc = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            acc |= val(c).ok_or("Invalid base64")? << (18 - 6 * i);
        }
        out.push((acc >> 16) as u8);
        if chunk.len() > 2 {
            out.push((acc >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(acc as u8);
        }
        if chunk.len() == 1 {
            return Err("Invalid base64".into());
        }
    }
    Ok(out)
}

pub fn base64_encode(input: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_requests_without_header_injection() {
        let req = build_request(
            "POST",
            "/login?x=1",
            "example.com",
            &[("Content-Type".into(), "application/x-www-form-urlencoded".into()), ("Bad\r\nX".into(), "y".into()), ("X-Ok".into(), "a\r\nb".into())],
            b"a=1",
        );
        let text = String::from_utf8(req).unwrap();
        assert!(text.starts_with("POST /login?x=1 HTTP/1.1\r\nHost: example.com\r\n"));
        assert!(text.contains("Content-Length: 3\r\n"));
        assert!(!text.contains("Bad"));
        assert!(!text.contains("X-Ok"));
        assert!(text.ends_with("\r\n\r\na=1"));
    }

    #[test]
    fn parses_heads_incrementally() {
        let raw = b"HTTP/1.1 302 Found\r\nLocation: /next\r\nSet-Cookie: a=1\r\nSet-Cookie: b=2\r\nContent-Length: 0\r\n\r\nrest";
        assert_eq!(parse_head(&raw[..20]).unwrap(), None);
        let (head, len) = parse_head(raw).unwrap().unwrap();
        assert_eq!(head.status, 302);
        assert_eq!(head.header("location"), Some("/next"));
        assert_eq!(head.all("set-cookie").collect::<Vec<_>>(), vec!["a=1", "b=2"]);
        assert_eq!(&raw[len..], b"rest");
        assert_eq!(framing("GET", &head).unwrap(), Framing::Length(0));
        assert!(keep_alive(&head, Framing::Length(0)));
    }

    #[test]
    fn decodes_chunked_bodies_split_anywhere() {
        let raw = b"4\r\nWiki\r\n5;ext=1\r\npedia\r\nE\r\n in\r\n\r\nchunks.\r\n0\r\nTrailer: x\r\n\r\nNEXT";
        for split in 0..raw.len() {
            let mut dec = ChunkedDecoder::default();
            let mut out = Vec::new();
            let a = dec.feed(&raw[..split], &mut out).unwrap();
            let (start, b) = if dec.done {
                (a, 0)
            } else {
                assert_eq!(a, split);
                (split, dec.feed(&raw[split..], &mut out).unwrap())
            };
            assert!(dec.done, "split {split}");
            assert_eq!(out, b"Wikipedia in\r\n\r\nchunks.");
            assert_eq!(&raw[start + b..], b"NEXT");
        }
    }

    #[test]
    fn decodes_gzip_deflate_and_brotli() {
        use std::io::Write;
        let text = b"hello hello hello world".to_vec();
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gz.write_all(&text).unwrap();
        assert_eq!(decode_content(Some("gzip"), gz.finish().unwrap()).unwrap(), text);
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(&text).unwrap();
        assert_eq!(decode_content(Some("deflate"), z.finish().unwrap()).unwrap(), text);
        // "hello" compressed with brotli quality 11
        let br = [0x0b, 0x02, 0x80, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x03];
        assert_eq!(decode_content(Some("br"), br.to_vec()).unwrap(), b"hello");
        assert!(decode_content(Some("zstd"), text.clone()).is_err());
    }

    #[test]
    fn parses_data_urls_and_base64() {
        assert_eq!(parse_data_url("data:text/html,%3Cb%3Ehi").unwrap(), ("text/html".into(), b"<b>hi".to_vec()));
        assert_eq!(parse_data_url("data:image/png;base64,aGVs bG8=").unwrap().1, b"hello");
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_decode("aGVsbG8").unwrap(), b"hello");
    }
}
