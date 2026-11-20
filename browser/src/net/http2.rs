//! HTTP/2 framing, HPACK and flow control use the maintained h2 implementation.
//! Cookie, redirect and CORS decisions remain in Client, shared with HTTP/1.1.

use super::http::{self, ResponseHead};
use super::FetchRequest;
use bytes::Bytes;
use futures::future::{AbortHandle, Abortable};
use std::future::Future;
use std::pin::pin;
use std::rc::Rc;
use std::task::Poll;
use tokio::io::{AsyncRead, AsyncWrite};

#[derive(Debug)]
pub enum Error {
    BeforeSend(String),
    AfterSend(String),
    Invalid(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::BeforeSend(s) | Self::AfterSend(s) | Self::Invalid(s) => write!(f, "HTTP/2: {s}") }
    }
}

struct Inner {
    sender: h2::client::SendRequest<Bytes>,
    abort: AbortHandle,
    firefox: bool,
}

impl Drop for Inner {
    fn drop(&mut self) { self.abort.abort(); }
}

#[derive(Clone)]
pub struct Session(Rc<Inner>);

/// Connection settings. Firefox's are SETTINGS 1=65536, 2=0, 4=131072, 5=16384, a connection
/// WINDOW_UPDATE to 12 MiB, and client streams from 3 (Firefox reserves 1 for h2c upgrades).
fn builder(firefox: bool) -> h2::client::Builder {
    let mut builder = h2::client::Builder::new();
    builder.header_table_size(65536).enable_push(false).initial_window_size(super::headers::FIREFOX_INITIAL_WINDOW)
        .max_frame_size(16384).max_send_buffer_size(64 * 1024);
    if firefox {
        builder.initial_connection_window_size(super::headers::FIREFOX_CONNECTION_WINDOW).initial_stream_id(3);
    } else {
        builder.initial_connection_window_size(12 * 1024 * 1024).max_header_list_size(http::MAX_HEAD as u32);
    }
    builder
}

pub async fn connect<T>(io: T, firefox: bool) -> Result<(Session, impl Future<Output = ()>), String>
where T: AsyncRead + AsyncWrite + Unpin + 'static {
    let (sender, connection) = builder(firefox).handshake(io).await.map_err(|e| format!("HTTP/2 handshake: {e}"))?;
    let (abort, registration) = AbortHandle::new_pair();
    let session = Session(Rc::new(Inner { sender, abort, firefox }));
    let driver = async move { let _ = Abortable::new(connection, registration).await; };
    Ok((session, driver))
}

impl Session {
    pub fn same_connection(&self, other: &Self) -> bool { Rc::ptr_eq(&self.0, &other.0) }

    pub async fn request(&self, req: &FetchRequest, headers: &[(String, String)], framing: h2::ext::RequestFraming) -> Result<(ResponseHead, Vec<u8>), Error> {
        let mut request = request_head(req, headers, self.0.firefox).map_err(Error::Invalid)?;
        request.extensions_mut().insert(framing);
        let mut sender = self.0.sender.clone().ready().await.map_err(|e| Error::BeforeSend(e.to_string()))?;
        let (response, mut send) = sender.send_request(request, req.body.is_empty()).map_err(|e| Error::BeforeSend(e.to_string()))?;
        let mut response = pin!(response);
        let mut upload = pin!(send_body(&mut send, &req.body));
        let mut upload_done = req.body.is_empty();
        let response = futures::future::poll_fn(|cx| {
            // A final early response can arrive while upload flow control is blocked.
            if let Poll::Ready(result) = response.as_mut().poll(cx) { return Poll::Ready(result.map_err(|e| Error::AfterSend(e.to_string()))); }
            if !upload_done {
                match upload.as_mut().poll(cx) {
                    Poll::Ready(Ok(())) => upload_done = true,
                    Poll::Ready(Err(e)) => return Poll::Ready(Err(Error::AfterSend(e))),
                    Poll::Pending => {},
                }
            }
            Poll::Pending
        }).await?;
        let (parts, mut stream) = response.into_parts();
        let mut size = 0;
        let mut fields = Vec::new();
        for (name, value) in &parts.headers {
            size += name.as_str().len() + value.as_bytes().len() + 32;
            if size > http::MAX_HEAD { return Err(Error::AfterSend("Response headers are too large".into())); }
            let value = value.to_str().map_err(|_| Error::AfterSend("Invalid response header value".into()))?;
            fields.push((name.to_string(), value.to_string()));
        }
        let head = ResponseHead { status: parts.status.as_u16(), reason: parts.status.canonical_reason().unwrap_or("").into(), headers: fields };
        if req.method != "HEAD" && head.header("content-length").and_then(|v| v.parse::<usize>().ok()).is_some_and(|n| n > http::MAX_BODY) {
            return Err(Error::AfterSend("Response body is too large".into()));
        }
        let mut body = Vec::new();
        loop {
            let data = futures::future::poll_fn(|cx| {
                let data = stream.poll_data(cx);
                if data.is_ready() { return data; }
                if !upload_done && upload.as_mut().poll(cx).is_ready() { upload_done = true; }
                Poll::Pending
            }).await;
            let Some(data) = data else { break };
            let data = data.map_err(|e| Error::AfterSend(e.to_string()))?;
            if body.len().saturating_add(data.len()) > http::MAX_BODY { return Err(Error::AfterSend("Response body is too large".into())); }
            if req.method == "HEAD" && !data.is_empty() { return Err(Error::AfterSend("HEAD response contained DATA".into())); }
            body.extend_from_slice(&data);
            stream.flow_control().release_capacity(data.len()).map_err(|e| Error::AfterSend(e.to_string()))?;
        }
        let body = http::decode_content(head.header("content-encoding"), body).map_err(Error::AfterSend)?;
        Ok((head, body))
    }
}

pub(crate) fn request_head(req: &FetchRequest, headers: &[(String, String)], firefox: bool) -> Result<::http::Request<()>, String> {
    let mut url = req.url.clone();
    url.set_fragment(None);
    let _ = url.set_username("");
    let _ = url.set_password(None);
    let mut request = ::http::Request::builder().method(req.method.as_str()).uri(url.as_str()).version(::http::Version::HTTP_2).body(()).map_err(|e| e.to_string())?;
    let length = (!req.body.is_empty() || matches!(req.method.as_str(), "POST" | "PUT" | "PATCH")).then(|| req.body.len().to_string());
    let fields = request.headers_mut();
    let mut add = |name: &str, value: &str, sensitive: bool| -> Result<(), String> {
        let mut value = ::http::HeaderValue::from_str(value).map_err(|e| e.to_string())?;
        value.set_sensitive(sensitive);
        fields.append(::http::HeaderName::from_bytes(name.as_bytes()).map_err(|e| e.to_string())?, value);
        Ok(())
    };
    let mut wrote_length = false;
    for (name, value) in headers {
        let name = name.to_ascii_lowercase();
        if matches!(name.as_str(), "host" | "connection" | "proxy-connection" | "keep-alive" | "transfer-encoding" | "upgrade") { continue; }
        if name == "te" && !value.eq_ignore_ascii_case("trailers") { continue; }
        if name == "content-length" {
            // The body decides the length; keep the field where the caller placed it.
            if let Some(length) = &length { if !wrote_length { add("content-length", length, false)?; wrote_length = true; } }
            continue;
        }
        if name == "cookie" && firefox {
            // Firefox sends each cookie as its own field (RFC 9113 §8.2.3) so HPACK can index
            // them, except short ones, which are never indexed to resist guessing (RFC 7541 §7.1.3).
            for crumb in value.split("; ").filter(|c| !c.is_empty()) { add("cookie", crumb, crumb.len() < 20)?; }
            continue;
        }
        add(&name, value, matches!(name.as_str(), "cookie" | "authorization" | "proxy-authorization"))?;
    }
    if let (Some(length), false) = (&length, wrote_length) { add("content-length", length, false)?; }
    Ok(request)
}

async fn send_body(send: &mut h2::SendStream<Bytes>, body: &[u8]) -> Result<(), String> {
    let mut offset = 0;
    while offset < body.len() {
        send.reserve_capacity((body.len() - offset).min(16 * 1024));
        let capacity = futures::future::poll_fn(|cx| send.poll_capacity(cx)).await
            .ok_or("Request upload closed")?.map_err(|e| e.to_string())?;
        let n = capacity.min(body.len() - offset).min(16 * 1024);
        if n == 0 { continue; }
        send.send_data(Bytes::copy_from_slice(&body[offset..offset + n]), offset + n == body.len()).map_err(|e| e.to_string())?;
        offset += n;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::{Credentials, Mode, Redirect};
    use futures::task::LocalSpawnExt;
    use std::cell::RefCell;
    use std::pin::Pin;
    use std::task::Context;
    use tokio::io::ReadBuf;

    /// Records the bytes the client writes.
    struct Tap<T> {
        io: T,
        sent: Rc<RefCell<Vec<u8>>>,
    }
    impl<T: AsyncRead + Unpin> AsyncRead for Tap<T> {
        fn poll_read(self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &mut ReadBuf<'_>) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.get_mut().io).poll_read(cx, buf)
        }
    }
    impl<T: AsyncWrite + Unpin> AsyncWrite for Tap<T> {
        fn poll_write(self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &[u8]) -> Poll<std::io::Result<usize>> {
            let this = self.get_mut();
            let result = Pin::new(&mut this.io).poll_write(cx, buf);
            if let Poll::Ready(Ok(n)) = result {
                this.sent.borrow_mut().extend_from_slice(&buf[..n]);
            }
            result
        }
        fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.get_mut().io).poll_flush(cx)
        }
        fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.get_mut().io).poll_shutdown(cx)
        }
    }

    struct Exchange {
        sent: Vec<u8>,
        head: ResponseHead,
        body: Vec<u8>,
        /// Request field names and values in the order the server decoded them, and the body
        server_fields: Vec<(String, String)>,
        server_body: Vec<u8>,
    }

    /// One request from our client to h2's server over an in-memory connection.
    fn exchange(firefox: bool, req: FetchRequest, headers: Vec<(String, String)>, framing: h2::ext::RequestFraming) -> Exchange {
        let (client_io, server_io) = tokio::io::duplex(1 << 20);
        let sent = Rc::new(RefCell::new(Vec::new()));
        let mut pool = futures::executor::LocalPool::new();
        let seen = Rc::new(RefCell::new((Vec::new(), Vec::new())));
        let server_seen = seen.clone();
        let handlers = pool.spawner();
        // The server connection keeps being polled while a separate task reads the request.
        pool.spawner().spawn_local(async move {
            let mut conn = h2::server::handshake(server_io).await.unwrap();
            while let Some(Ok((request, mut respond))) = conn.accept().await {
                let server_seen = server_seen.clone();
                handlers.spawn_local(async move {
                    let (parts, mut body) = request.into_parts();
                    let mut data = Vec::new();
                    while let Some(chunk) = body.data().await {
                        let chunk = chunk.unwrap();
                        data.extend_from_slice(&chunk);
                        body.flow_control().release_capacity(chunk.len()).unwrap();
                    }
                    let fields = parts.headers.iter().map(|(k, v)| (k.to_string(), v.to_str().unwrap().to_string())).collect();
                    *server_seen.borrow_mut() = (fields, data);
                    let large = parts.uri.path() == "/large";
                    let response = ::http::Response::builder().status(200).header("content-type", "text/plain").body(()).unwrap();
                    let mut send = respond.send_response(response, false).unwrap();
                    let body = if large { Bytes::from(vec![b'x'; 3_000_000]) } else { Bytes::from_static(b"ok") };
                    send.send_data(body, true).unwrap();
                }).unwrap();
            }
        }).unwrap();
        let spawner = pool.spawner();
        let tap = Tap { io: client_io, sent: sent.clone() };
        let (head, body) = pool.run_until(async move {
            let (session, driver) = connect(tap, firefox).await.unwrap();
            spawner.spawn_local(driver).unwrap();
            session.request(&req, &headers, framing).await.unwrap()
        });
        let (server_fields, server_body) = seen.take();
        let sent = sent.take();
        Exchange { sent, head, body, server_fields, server_body }
    }

    /// (type, flags, stream, payload) for each client frame after the connection preface
    fn frames(bytes: &[u8]) -> Vec<(u8, u8, u32, Vec<u8>)> {
        const PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
        assert!(bytes.starts_with(PREFACE));
        let mut out = Vec::new();
        let mut i = PREFACE.len();
        while i + 9 <= bytes.len() {
            let len = u32::from_be_bytes([0, bytes[i], bytes[i + 1], bytes[i + 2]]) as usize;
            let stream = u32::from_be_bytes(bytes[i + 5..i + 9].try_into().unwrap()) & 0x7fff_ffff;
            out.push((bytes[i + 3], bytes[i + 4], stream, bytes[i + 9..i + 9 + len].to_vec()));
            i += 9 + len;
        }
        out
    }

    /// Names of the first four fields of a fresh HPACK block, from their static-table indexes.
    fn pseudo_names(block: &[u8]) -> Vec<&'static str> {
        fn int(block: &[u8], i: &mut usize, prefix: u32) -> usize {
            let mask = (1usize << prefix) - 1;
            let mut value = block[*i] as usize & mask;
            *i += 1;
            if value == mask {
                let mut shift = 0;
                loop {
                    let b = block[*i];
                    *i += 1;
                    value += ((b & 0x7f) as usize) << shift;
                    shift += 7;
                    if b & 0x80 == 0 { break; }
                }
            }
            value
        }
        let name = |index| match index { 1 => ":authority", 2 | 3 => ":method", 4 | 5 => ":path", 6 | 7 => ":scheme", _ => "other" };
        let (mut i, mut out) = (0, Vec::new());
        while out.len() < 4 {
            let b = block[i];
            let index = if b & 0x80 != 0 {
                int(block, &mut i, 7)
            } else if b & 0xe0 == 0x20 {
                int(block, &mut i, 5); // dynamic table size update
                continue;
            } else {
                let index = int(block, &mut i, if b & 0x40 != 0 { 6 } else { 4 });
                let len = int(block, &mut i, 7);
                i += len;
                index
            };
            out.push(name(index));
        }
        out
    }

    fn settings(payload: &[u8]) -> Vec<(u16, u32)> {
        payload.chunks(6).map(|c| (u16::from_be_bytes([c[0], c[1]]), u32::from_be_bytes([c[2], c[3], c[4], c[5]]))).collect()
    }

    fn window_increment(payload: &[u8]) -> u32 {
        u32::from_be_bytes(payload.try_into().unwrap()) & 0x7fff_ffff
    }

    fn request(method: &str, url: &str, dest: &'static str) -> FetchRequest {
        FetchRequest {
            url: url::Url::parse(url).unwrap(),
            method: method.into(),
            headers: vec![],
            body: vec![],
            initiator: Some(url::Url::parse("https://accounts.google.com/v3/signin/identifier").unwrap()),
            mode: Mode::Cors,
            credentials: Credentials::Include,
            redirect: Redirect::Follow,
            dest,
            user_activated: false,
        }
    }

    /// The connection opens like Firefox ESR's capture: SETTINGS 1,2,4,5; WINDOW_UPDATE to
    /// 12 MiB; HEADERS on stream 3 with priority and :method :path :authority :scheme; then
    /// the stream's WINDOW_UPDATE to 12 MiB.
    #[test]
    fn firefox_connection_matches_reference_framing() {
        let req = request("GET", "https://accounts.google.com/v3/signin/identifier?hl=en", "document");
        let headers = vec![("User-Agent".into(), "UA".into()), ("Cookie".into(), "NID=511=abcdefghijklmnopqrstuvwxyz; a=1".into()), ("TE".into(), "trailers".into())];
        let framing = crate::net::headers::h2_framing(crate::identity::Identity::FirefoxEsr, "document");
        let ex = exchange(true, req, headers, framing);
        assert_eq!((ex.head.status, ex.body.as_slice()), (200, &b"ok"[..]));

        let fixture: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/firefox-esr-153.7-protocol.json")).unwrap();
        let reference: Vec<(u16, u32)> = fixture["http2"]["client_settings"][0]["settings"].as_array().unwrap().iter()
            .map(|s| (s["id"].as_u64().unwrap() as u16, s["value"].as_u64().unwrap() as u32)).collect();
        let reference_increment = fixture["http2"]["initial_connection_window_increments"][0].as_u64().unwrap() as u32;

        // Leave out SETTINGS acknowledgements, whose position depends on the server.
        let frames: Vec<_> = frames(&ex.sent).into_iter().filter(|f| !(f.0 == 4 && f.1 & 1 == 1)).collect();
        assert_eq!((frames[0].0, frames[0].2), (4, 0));
        assert_eq!(settings(&frames[0].3), reference);
        assert_eq!((frames[1].0, frames[1].2, window_increment(&frames[1].3)), (8, 0, reference_increment));
        let (kind, flags, stream, payload) = &frames[2];
        assert_eq!((*kind, *flags, *stream), (1, 0x25, 3), "HEADERS: END_STREAM | END_HEADERS | PRIORITY on stream 3");
        assert_eq!(&payload[..5], &[0, 0, 0, 0, 41], "non-exclusive, depends on 0, weight 42");
        assert_eq!(pseudo_names(&payload[5..]), [":method", ":path", ":authority", ":scheme"]);
        assert_eq!((frames[3].0, frames[3].2, window_increment(&frames[3].3)), (8, 3, 12451840));

        let cookies: Vec<_> = ex.server_fields.iter().filter(|(k, _)| k == "cookie").map(|(_, v)| v.as_str()).collect();
        assert_eq!(cookies, ["NID=511=abcdefghijklmnopqrstuvwxyz", "a=1"]);
        assert!(ex.server_fields.contains(&("te".into(), "trailers".into())));
    }

    #[test]
    fn firefox_post_keeps_field_order_and_uploads_body() {
        let mut req = request("POST", "https://accounts.google.com/v3/signin/_/AccountsSignInUi/data/batchexecute", "empty");
        req.body = b"f.req=%5B%5D&at=token".to_vec();
        req.headers = vec![("Content-Type".into(), "application/x-www-form-urlencoded;charset=utf-8".into()), ("X-Same-Domain".into(), "1".into())];
        let identity = crate::identity::Identity::FirefoxEsr;
        let headers = crate::net::headers::request_headers(&req, identity, "UA", "en-US,en;q=0.9", Some("NID=1".into()), Some("https://accounts.google.com"));
        let ex = exchange(true, req.clone(), headers.clone(), crate::net::headers::h2_framing(identity, "empty"));
        assert_eq!(ex.server_body, req.body);
        let sent: Vec<String> = ex.server_fields.iter().map(|(k, _)| k.clone()).collect();
        let expected: Vec<String> = headers.iter().map(|(k, _)| k.to_ascii_lowercase()).collect();
        assert_eq!(sent, expected);
        assert!(ex.server_fields.contains(&("content-length".into(), req.body.len().to_string())));
        let frames = frames(&ex.sent);
        let headers_frame = frames.iter().find(|f| f.0 == 1).unwrap();
        assert_eq!((headers_frame.1, headers_frame.2, headers_frame.3[4]), (0x24, 3, 21), "no END_STREAM before the body; weight 22");
    }

    /// A stream window raised before our SETTINGS are acknowledged stays consistent with the
    /// server's view: a response larger than both initial windows arrives without a
    /// flow-control error.
    #[test]
    fn firefox_stream_window_accepts_large_response() {
        let req = request("GET", "https://example.com/large", "document");
        let framing = crate::net::headers::h2_framing(crate::identity::Identity::FirefoxEsr, "document");
        let ex = exchange(true, req, vec![], framing);
        assert_eq!(ex.head.status, 200);
        assert_eq!(ex.body.len(), 3_000_000);
    }

    #[test]
    fn post_uploads_body_with_default_framing() {
        let mut req = request("POST", "https://example.com/upload", "empty");
        req.body = vec![7; 100_000];
        let headers = vec![("Content-Type".into(), "application/octet-stream".into())];
        let ex = exchange(false, req.clone(), headers, h2::ext::RequestFraming::default());
        assert_eq!(ex.server_body, req.body);
        assert_eq!(ex.body, b"ok");
    }

    #[test]
    fn other_identities_keep_h2_default_framing() {
        let req = request("GET", "https://example.com/a", "document");
        let ex = exchange(false, req, vec![("Cookie".into(), "a=1; b=2".into())], h2::ext::RequestFraming::default());
        assert_eq!(ex.body, b"ok");
        let frames = frames(&ex.sent);
        let headers_frame = frames.iter().find(|f| f.0 == 1).unwrap();
        assert_eq!((headers_frame.1, headers_frame.2), (0x05, 1));
        assert_eq!(pseudo_names(&headers_frame.3), [":method", ":scheme", ":authority", ":path"]);
        assert!(!frames.iter().any(|f| f.0 == 8 && f.2 != 0), "no stream WINDOW_UPDATE before data");
        assert_eq!(ex.server_fields.iter().filter(|(k, _)| k == "cookie").count(), 1);
    }
}
