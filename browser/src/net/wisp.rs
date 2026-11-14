//! Wisp v1: many TCP streams multiplexed over one WebSocket.
//! <https://github.com/MercuryWorkshop/wisp-protocol>
//!
//! Packets are `[type: u8][stream id: u32 LE][payload]`. The relay only forwards bytes; TLS is
//! negotiated end to end inside the engine, so the relay never sees page contents or cookies.

pub const CONNECT: u8 = 0x01;
pub const DATA: u8 = 0x02;
pub const CONTINUE: u8 = 0x03;
pub const CLOSE: u8 = 0x04;
pub const STREAM_TCP: u8 = 0x01;

pub const CLOSE_VOLUNTARY: u8 = 0x02;

#[derive(Debug, Clone, PartialEq)]
pub enum Packet {
    Connect { id: u32, kind: u8, port: u16, host: String },
    Data { id: u32, payload: Vec<u8> },
    Continue { id: u32, buffer: u32 },
    Close { id: u32, reason: u8 },
}

pub fn encode(packet: &Packet) -> Vec<u8> {
    let mut out = Vec::new();
    match packet {
        Packet::Connect { id, kind, port, host } => {
            out.push(CONNECT);
            out.extend_from_slice(&id.to_le_bytes());
            out.push(*kind);
            out.extend_from_slice(&port.to_le_bytes());
            out.extend_from_slice(host.as_bytes());
        }
        Packet::Data { id, payload } => {
            out.reserve(5 + payload.len());
            out.push(DATA);
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(payload);
        }
        Packet::Continue { id, buffer } => {
            out.push(CONTINUE);
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&buffer.to_le_bytes());
        }
        Packet::Close { id, reason } => {
            out.push(CLOSE);
            out.extend_from_slice(&id.to_le_bytes());
            out.push(*reason);
        }
    }
    out
}

pub fn decode(buf: &[u8]) -> Result<Packet, String> {
    if buf.len() < 5 {
        return Err("Wisp packet is too short".into());
    }
    let id = u32::from_le_bytes(buf[1..5].try_into().unwrap());
    let body = &buf[5..];
    match buf[0] {
        CONNECT if body.len() >= 3 => Ok(Packet::Connect {
            id,
            kind: body[0],
            port: u16::from_le_bytes([body[1], body[2]]),
            host: String::from_utf8(body[3..].to_vec()).map_err(|_| "Invalid host")?,
        }),
        DATA => Ok(Packet::Data { id, payload: body.to_vec() }),
        CONTINUE if body.len() >= 4 => Ok(Packet::Continue { id, buffer: u32::from_le_bytes(body[..4].try_into().unwrap()) }),
        CLOSE if !body.is_empty() => Ok(Packet::Close { id, reason: body[0] }),
        kind => Err(format!("Unknown or truncated Wisp packet type {kind:#x}")),
    }
}

pub fn close_reason(reason: u8) -> &'static str {
    match reason {
        0x01 | 0x02 => "the connection closed",
        0x03 => "the network failed",
        0x41 => "the relay refused this stream type",
        0x42 => "the server could not be reached or its name could not be resolved",
        0x43 => "the connection timed out",
        0x44 => "the server refused the connection",
        0x47 => "the connection stopped sending data",
        0x48 => "the relay blocks this address",
        0x49 => "the relay is limiting connections",
        _ => "the relay closed the connection",
    }
}

#[cfg(target_arch = "wasm32")]
pub use mux::{Mux, WispStream};

#[cfg(target_arch = "wasm32")]
mod mux {
    use super::*;
    use std::cell::RefCell;
    use std::collections::{HashMap, VecDeque};
    use std::future::poll_fn;
    use std::rc::Rc;
    use std::task::{Poll, Waker};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;
    use web_sys::{BinaryType, MessageEvent, WebSocket};

    const MAX_PAYLOAD: usize = 32 * 1024;

    #[derive(Default)]
    struct Slot {
        rx: VecDeque<Vec<u8>>,
        closed: Option<u8>,
        credit: u32,
        read_waker: Option<Waker>,
        write_waker: Option<Waker>,
    }

    enum State {
        Connecting,
        Open,
        Failed(String),
    }

    struct Inner {
        url: String,
        ws: Option<WebSocket>,
        state: State,
        initial_credit: u32,
        got_initial: bool,
        next_id: u32,
        slots: HashMap<u32, Slot>,
        open_wakers: Vec<Waker>,
        _closures: Vec<Closure<dyn FnMut(JsValue)>>,
    }

    /// A shared WebSocket connection to a Wisp relay.
    #[derive(Clone)]
    pub struct Mux(Rc<RefCell<Inner>>);

    impl Mux {
        pub fn new(url: &str) -> Self {
            Mux(Rc::new(RefCell::new(Inner {
                url: url.to_string(),
                ws: None,
                state: State::Failed("Not connected".into()),
                initial_credit: 128,
                got_initial: false,
                next_id: 1,
                slots: HashMap::new(),
                open_wakers: vec![],
                _closures: vec![],
            })))
        }

        pub fn url(&self) -> String {
            self.0.borrow().url.clone()
        }

        pub fn is_open(&self) -> bool {
            matches!(self.0.borrow().state, State::Open)
        }

        fn ensure_socket(&self) -> Result<(), String> {
            let mut inner = self.0.borrow_mut();
            if matches!(inner.state, State::Open | State::Connecting) {
                return Ok(());
            }
            let ws = WebSocket::new(&inner.url).map_err(|e| format!("Could not open the relay WebSocket: {e:?}"))?;
            ws.set_binary_type(BinaryType::Arraybuffer);
            let weak = Rc::downgrade(&self.0);
            let on_open = Closure::<dyn FnMut(JsValue)>::new({
                let weak = weak.clone();
                move |_| {
                    if let Some(inner) = weak.upgrade() {
                        let mut inner = inner.borrow_mut();
                        inner.state = State::Open;
                        // Wisp v1 servers send CONTINUE for stream 0 first; wait briefly for it,
                        // but don't block if a relay omits it.
                        for w in inner.open_wakers.drain(..) {
                            w.wake();
                        }
                    }
                }
            });
            let on_message = Closure::<dyn FnMut(JsValue)>::new({
                let weak = weak.clone();
                move |event: JsValue| {
                    let Some(inner) = weak.upgrade() else { return };
                    let event: MessageEvent = event.unchecked_into();
                    let Ok(buf) = event.data().dyn_into::<js_sys::ArrayBuffer>() else { return };
                    let bytes = js_sys::Uint8Array::new(&buf).to_vec();
                    let Ok(packet) = decode(&bytes) else { return };
                    let mut inner = inner.borrow_mut();
                    match packet {
                        Packet::Continue { id: 0, buffer } => {
                            inner.initial_credit = buffer.max(1);
                            inner.got_initial = true;
                            for w in inner.open_wakers.drain(..) {
                                w.wake();
                            }
                        }
                        Packet::Continue { id, buffer } => {
                            if let Some(slot) = inner.slots.get_mut(&id) {
                                slot.credit = buffer;
                                if let Some(w) = slot.write_waker.take() {
                                    w.wake();
                                }
                            }
                        }
                        Packet::Data { id, payload } => {
                            if let Some(slot) = inner.slots.get_mut(&id) {
                                slot.rx.push_back(payload);
                                if let Some(w) = slot.read_waker.take() {
                                    w.wake();
                                }
                            }
                        }
                        Packet::Close { id, reason } => {
                            if let Some(slot) = inner.slots.get_mut(&id) {
                                slot.closed = Some(reason);
                                if let Some(w) = slot.read_waker.take() {
                                    w.wake();
                                }
                                if let Some(w) = slot.write_waker.take() {
                                    w.wake();
                                }
                            }
                        }
                        Packet::Connect { .. } => {}
                    }
                }
            });
            let on_close = Closure::<dyn FnMut(JsValue)>::new({
                let weak = weak.clone();
                move |_| {
                    if let Some(inner) = weak.upgrade() {
                        let mut inner = inner.borrow_mut();
                        inner.state = State::Failed("The relay connection closed".into());
                        inner.ws = None;
                        inner.got_initial = false;
                        for slot in inner.slots.values_mut() {
                            slot.closed.get_or_insert(0x03);
                            if let Some(w) = slot.read_waker.take() {
                                w.wake();
                            }
                            if let Some(w) = slot.write_waker.take() {
                                w.wake();
                            }
                        }
                        for w in inner.open_wakers.drain(..) {
                            w.wake();
                        }
                    }
                }
            });
            ws.set_onopen(Some(on_open.as_ref().unchecked_ref()));
            ws.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
            ws.set_onclose(Some(on_close.as_ref().unchecked_ref()));
            ws.set_onerror(Some(on_close.as_ref().unchecked_ref()));
            inner._closures = vec![on_open, on_message, on_close];
            inner.ws = Some(ws);
            inner.state = State::Connecting;
            Ok(())
        }

        async fn wait_open(&self) -> Result<(), String> {
            self.ensure_socket()?;
            let deadline = js_sys::Date::now() + 300.0;
            poll_fn(|cx| {
                let mut inner = self.0.borrow_mut();
                match &inner.state {
                    State::Open if inner.got_initial || js_sys::Date::now() > deadline => Poll::Ready(Ok(())),
                    State::Failed(reason) => Poll::Ready(Err(format!(
                        "Could not reach the relay at {} ({reason}). Start it with `evolve-relay`, or set another relay in the browser settings.",
                        inner.url
                    ))),
                    State::Open => {
                        // Re-poll shortly in case the relay never sends the initial CONTINUE.
                        let waker = cx.waker().clone();
                        let cb = Closure::once_into_js(move || waker.wake());
                        set_timeout(&cb, 50);
                        inner.open_wakers.push(cx.waker().clone());
                        Poll::Pending
                    }
                    State::Connecting => {
                        inner.open_wakers.push(cx.waker().clone());
                        Poll::Pending
                    }
                }
            })
            .await
        }

        pub async fn connect(&self, host: &str, port: u16) -> Result<WispStream, String> {
            self.wait_open().await?;
            let mut inner = self.0.borrow_mut();
            let id = inner.next_id;
            inner.next_id = inner.next_id.wrapping_add(1).max(1);
            let credit = inner.initial_credit;
            inner.slots.insert(id, Slot { credit, ..Default::default() });
            let packet = encode(&Packet::Connect { id, kind: STREAM_TCP, port, host: host.to_string() });
            inner.ws.as_ref().ok_or("The relay is not connected")?.send_with_u8_array(&packet).map_err(|e| format!("{e:?}"))?;
            Ok(WispStream { mux: self.0.clone(), id, closed: false })
        }
    }

    fn set_timeout(cb: &JsValue, ms: i32) {
        let global = js_sys::global();
        if let Ok(f) = js_sys::Reflect::get(&global, &"setTimeout".into()) {
            if let Some(f) = f.dyn_ref::<js_sys::Function>() {
                let _ = f.call2(&global, cb, &JsValue::from(ms));
            }
        }
    }

    pub struct WispStream {
        mux: Rc<RefCell<Inner>>,
        id: u32,
        closed: bool,
    }

    impl WispStream {
        /// Read the next chunk. `Ok(None)` is a clean end of stream.
        pub async fn read(&mut self) -> Result<Option<Vec<u8>>, String> {
            let id = self.id;
            poll_fn(|cx| {
                let mut inner = self.mux.borrow_mut();
                let Some(slot) = inner.slots.get_mut(&id) else { return Poll::Ready(Ok(None)) };
                if let Some(chunk) = slot.rx.pop_front() {
                    return Poll::Ready(Ok(Some(chunk)));
                }
                match slot.closed {
                    Some(0x01 | 0x02) => Poll::Ready(Ok(None)),
                    Some(reason) => Poll::Ready(Err(format!("Connection failed: {}", close_reason(reason)))),
                    None => {
                        slot.read_waker = Some(cx.waker().clone());
                        Poll::Pending
                    }
                }
            })
            .await
        }

        pub async fn write(&mut self, data: &[u8]) -> Result<(), String> {
            for chunk in data.chunks(MAX_PAYLOAD) {
                let id = self.id;
                poll_fn(|cx| {
                    let mut inner = self.mux.borrow_mut();
                    let inner = &mut *inner;
                    let Some(slot) = inner.slots.get_mut(&id) else { return Poll::Ready(Err("Stream closed".to_string())) };
                    if let Some(reason) = slot.closed {
                        return Poll::Ready(Err(format!("Connection failed: {}", close_reason(reason))));
                    }
                    if slot.credit == 0 {
                        slot.write_waker = Some(cx.waker().clone());
                        return Poll::Pending;
                    }
                    slot.credit -= 1;
                    let packet = encode(&Packet::Data { id, payload: chunk.to_vec() });
                    let sent = inner.ws.as_ref().map(|ws| ws.send_with_u8_array(&packet));
                    Poll::Ready(match sent {
                        Some(Ok(())) => Ok(()),
                        _ => Err("The relay connection closed".to_string()),
                    })
                })
                .await?;
            }
            Ok(())
        }

        pub fn close(&mut self) {
            if self.closed {
                return;
            }
            self.closed = true;
            let mut inner = self.mux.borrow_mut();
            let open = inner.slots.remove(&self.id).is_some_and(|s| s.closed.is_none());
            if open {
                if let Some(ws) = &inner.ws {
                    let _ = ws.send_with_u8_array(&encode(&Packet::Close { id: self.id, reason: CLOSE_VOLUNTARY }));
                }
            }
        }
    }

    impl Drop for WispStream {
        fn drop(&mut self) {
            self.close();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_packet_type() {
        let packets = [
            Packet::Connect { id: 7, kind: STREAM_TCP, port: 443, host: "example.com".into() },
            Packet::Data { id: 7, payload: b"hello".to_vec() },
            Packet::Continue { id: 0, buffer: 128 },
            Packet::Close { id: 7, reason: CLOSE_VOLUNTARY },
        ];
        for p in packets {
            assert_eq!(decode(&encode(&p)).unwrap(), p);
        }
        assert_eq!(encode(&Packet::Connect { id: 1, kind: 1, port: 443, host: "a".into() }), [1, 1, 0, 0, 0, 1, 0xbb, 0x01, b'a']);
        assert!(decode(&[9, 0, 0, 0, 0]).is_err());
        assert!(decode(&[1, 0]).is_err());
    }
}
