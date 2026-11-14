//! The relay: accepts WebSocket connections from allowed page origins and forwards Wisp TCP
//! streams to public servers.

use crate::codec::{self, reason, Packet};
use crate::limits::{self, Clients, Usage};
use crate::policy;
use crate::Config;
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tokio_tungstenite::tungstenite::http::StatusCode;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::Message;

pub const MAX_STREAMS: usize = 256;
/// Packets the relay buffers per stream before the client must wait for CONTINUE.
pub const BUFFER_PACKETS: u32 = 128;
/// After this many packets are written to a TCP socket, the client's credit is replenished.
pub const CONTINUE_EVERY: u32 = 64;
pub const MAX_READ: usize = 32 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Serve until the listener fails. Each WebSocket connection gets its own task.
pub async fn serve(listener: TcpListener, config: Arc<Config>) -> std::io::Result<()> {
    let slots = Arc::new(Semaphore::new(config.max_connections.max(1)));
    let clients = Arc::new(Clients::default());
    loop {
        let (socket, peer) = listener.accept().await?;
        let _ = socket.set_nodelay(true);
        let Ok(permit) = slots.clone().try_acquire_owned() else {
            tokio::spawn(async move {
                let mut socket = socket;
                let _ = socket.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await;
            });
            continue;
        };
        let config = config.clone();
        let clients = clients.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(socket, peer, config.clone(), clients).await {
                if config.verbose {
                    eprintln!("[relay] {peer}: {e}");
                }
            }
            drop(permit);
        });
    }
}

fn reject(status: StatusCode, text: &str) -> ErrorResponse {
    let mut resp = ErrorResponse::new(Some(text.to_string()));
    *resp.status_mut() = status;
    resp
}

/// Releases a client's connection slot when the connection ends.
struct ClientSlot {
    clients: Arc<Clients>,
    ip: std::net::IpAddr,
    usage: Arc<Usage>,
}

impl Drop for ClientSlot {
    fn drop(&mut self) {
        self.clients.disconnect(self.ip, &self.usage);
    }
}

/// Counts one open stream against its client while the stream task lives.
struct StreamSlot(Arc<Usage>);

impl Drop for StreamSlot {
    fn drop(&mut self) {
        self.0.streams.fetch_sub(1, Ordering::SeqCst);
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

async fn handle_connection(socket: TcpStream, peer: SocketAddr, config: Arc<Config>, clients: Arc<Clients>) -> Result<(), String> {
    let origins = config.allow_origins.clone();
    let slot: Arc<std::sync::Mutex<Option<ClientSlot>>> = Arc::default();
    let slot_cb = slot.clone();
    let config_cb = config.clone();
    let callback = move |req: &Request, resp: Response| -> Result<Response, ErrorResponse> {
        let origin = req.headers().get("origin").and_then(|v| v.to_str().ok()).unwrap_or("");
        if !policy::origin_allowed(origin, &origins) {
            return Err(reject(StatusCode::FORBIDDEN, "This page's origin may not use this relay.\n"));
        }
        let headers: Vec<(&str, &str)> = req.headers().iter().filter_map(|(k, v)| Some((k.as_str(), v.to_str().ok()?))).collect();
        let ip = limits::client_ip(&headers, peer.ip(), config_cb.trust_forwarded);
        match clients.connect(ip, config_cb.max_per_client) {
            Some(usage) => {
                *slot_cb.lock().unwrap() = Some(ClientSlot { clients: clients.clone(), ip, usage });
                Ok(resp)
            }
            None => Err(reject(StatusCode::TOO_MANY_REQUESTS, "Too many relay connections from this address.\n")),
        }
    };
    let mut ws_config = WebSocketConfig::default();
    ws_config.max_message_size = Some(1 << 20);
    ws_config.max_frame_size = Some(1 << 20);
    let ws = tokio::time::timeout(HANDSHAKE_TIMEOUT, tokio_tungstenite::accept_hdr_async_with_config(socket, callback, Some(ws_config)))
        .await
        .map_err(|_| "handshake timed out".to_string())?
        .map_err(|e| format!("handshake failed: {e}"))?;
    let client = slot.lock().unwrap().take().ok_or("no client slot")?;
    let usage = client.usage.clone();
    if config.verbose {
        eprintln!("[relay] {peer}: connected");
    }
    let (mut sink, mut source) = ws.split();
    let last_activity = Arc::new(AtomicU64::new(now_secs()));
    let mut idle_check = tokio::time::interval(Duration::from_secs(30));
    idle_check.tick().await;

    // Everything sent to the client goes through one writer task.
    let (out_tx, mut out_rx) = mpsc::channel::<Vec<u8>>(256);
    let writer = tokio::spawn(async move {
        while let Some(bytes) = out_rx.recv().await {
            if sink.send(Message::Binary(bytes.into())).await.is_err() {
                break;
            }
        }
        let _ = sink.close().await;
    });
    out_tx
        .send(codec::encode(&Packet::Continue { id: 0, buffer: BUFFER_PACKETS }))
        .await
        .map_err(|_| "client went away")?;

    // Stream tasks report their end here so the stream table stays accurate.
    let (closed_tx, mut closed_rx) = mpsc::unbounded_channel::<(u32, u8, u64)>();
    let mut streams: HashMap<u32, StreamEntry> = HashMap::new();
    let mut generation: u64 = 0;

    loop {
        tokio::select! {
            msg = source.next() => {
                let Some(msg) = msg else { break };
                let msg = match msg { Ok(m) => m, Err(_) => break };
                last_activity.store(now_secs(), Ordering::Relaxed);
                let bytes = match msg {
                    Message::Binary(b) => b.to_vec(),
                    Message::Close(_) => break,
                    _ => continue, // text, ping and pong carry no Wisp packets
                };
                let Ok(packet) = codec::decode(&bytes) else { continue };
                match packet {
                    Packet::Connect { id, kind, port, host } => {
                        if let Some(old) = streams.remove(&id) {
                            old.abort();
                        }
                        let fail = |r: u8| codec::encode(&Packet::Close { id, reason: r });
                        if id == 0 || kind != codec::STREAM_TCP {
                            let _ = out_tx.send(fail(reason::INVALID_STREAM)).await;
                            continue;
                        }
                        let per_client = config.max_streams_per_client;
                        if streams.len() >= MAX_STREAMS
                            || (per_client > 0 && usage.streams.load(Ordering::SeqCst) >= per_client)
                            || usage.over_budget(config.client_bytes_per_hour)
                        {
                            let _ = out_tx.send(fail(reason::THROTTLED)).await;
                            continue;
                        }
                        if !policy::valid_host(&host) {
                            let _ = out_tx.send(fail(reason::UNREACHABLE_HOST)).await;
                            continue;
                        }
                        if !config.port_allowed(port) {
                            if config.verbose {
                                eprintln!("[relay] {peer}: blocked port {port}");
                            }
                            let _ = out_tx.send(fail(reason::BLOCKED)).await;
                            continue;
                        }
                        generation += 1;
                        usage.streams.fetch_add(1, Ordering::SeqCst);
                        let ctx = StreamContext { usage: usage.clone(), activity: last_activity.clone() };
                        let entry = open_stream(id, generation, host, port, config.clone(), out_tx.clone(), closed_tx.clone(), peer, ctx);
                        streams.insert(id, entry);
                    }
                    Packet::Data { id, payload } => {
                        let Some(entry) = streams.get(&id) else { continue };
                        entry.queued.fetch_add(1, Ordering::SeqCst);
                        if entry.to_tcp.try_send(payload).is_err() {
                            // The client ignored flow control (or the stream is ending).
                            if let Some(entry) = streams.remove(&id) {
                                entry.abort();
                            }
                            let _ = out_tx.send(codec::encode(&Packet::Close { id, reason: reason::NETWORK_ERROR })).await;
                        }
                    }
                    Packet::Close { id, .. } => {
                        if let Some(entry) = streams.remove(&id) {
                            entry.close();
                        }
                    }
                    Packet::Continue { .. } => {}
                }
            }
            _ = idle_check.tick() => {
                if now_secs().saturating_sub(last_activity.load(Ordering::Relaxed)) >= config.idle_timeout.as_secs() {
                    break;
                }
            }
            Some((id, why, generation)) = closed_rx.recv() => {
                // Only report the end of the stream that is still registered under this id.
                if streams.get(&id).is_some_and(|e| e.generation == generation) {
                    streams.remove(&id);
                    let _ = out_tx.send(codec::encode(&Packet::Close { id, reason: why })).await;
                }
            }
        }
    }
    for (_, entry) in streams.drain() {
        entry.abort();
    }
    drop(out_tx);
    let _ = writer.await;
    drop(client);
    if config.verbose {
        eprintln!("[relay] {peer}: disconnected");
    }
    Ok(())
}

/// What a stream reports back: bytes for its client's budget, and activity for the idle timer.
struct StreamContext {
    usage: Arc<Usage>,
    activity: Arc<AtomicU64>,
}

struct StreamEntry {
    generation: u64,
    to_tcp: mpsc::Sender<Vec<u8>>,
    queued: Arc<AtomicUsize>,
    task: JoinHandle<()>,
}

impl StreamEntry {
    /// The client closed the stream: stop both directions now.
    fn close(self) {
        self.task.abort();
    }
    fn abort(self) {
        self.task.abort();
    }
}

#[allow(clippy::too_many_arguments)]
fn open_stream(
    id: u32,
    generation: u64,
    host: String,
    port: u16,
    config: Arc<Config>,
    out_tx: mpsc::Sender<Vec<u8>>,
    closed_tx: mpsc::UnboundedSender<(u32, u8, u64)>,
    peer: SocketAddr,
    ctx: StreamContext,
) -> StreamEntry {
    // Room for the client's whole credit plus slack for packets in flight when CONTINUE is sent.
    let (to_tcp, mut from_client) = mpsc::channel::<Vec<u8>>(BUFFER_PACKETS as usize * 2);
    let queued = Arc::new(AtomicUsize::new(0));
    let queued_task = queued.clone();
    let StreamContext { usage, activity } = ctx;
    let task = tokio::spawn(async move {
        let _slot = StreamSlot(usage.clone());
        let usage_r = usage.clone();
        let activity_r = activity.clone();
        let finish = |why: u8| {
            let _ = closed_tx.send((id, why, generation));
        };
        let tcp = match connect(&host, port, &config).await {
            Ok(tcp) => tcp,
            Err(why) => {
                if config.verbose {
                    eprintln!("[relay] {peer}: stream {id} to {host}:{port} failed ({why:#x})");
                }
                return finish(why);
            }
        };
        if config.verbose {
            eprintln!("[relay] {peer}: stream {id} open to {host}:{port}");
        }
        let _ = tcp.set_nodelay(true);
        let (mut rd, mut wr) = tcp.into_split();
        let out = out_tx.clone();
        let reader = async move {
            let mut buf = vec![0u8; MAX_READ];
            loop {
                match rd.read(&mut buf).await {
                    Ok(0) => return reason::VOLUNTARY,
                    Ok(n) => {
                        usage_r.add_bytes(n as u64);
                        activity_r.store(now_secs(), Ordering::Relaxed);
                        let packet = codec::encode(&Packet::Data { id, payload: buf[..n].to_vec() });
                        if out.send(packet).await.is_err() {
                            return reason::NETWORK_ERROR;
                        }
                    }
                    Err(_) => return reason::NETWORK_ERROR,
                }
            }
        };
        let writer = async move {
            let mut written: u32 = 0;
            while let Some(payload) = from_client.recv().await {
                usage.add_bytes(payload.len() as u64);
                if wr.write_all(&payload).await.is_err() {
                    return Some(reason::NETWORK_ERROR);
                }
                let remaining = queued_task.fetch_sub(1, Ordering::SeqCst).saturating_sub(1);
                written += 1;
                if written.is_multiple_of(CONTINUE_EVERY) {
                    let buffer = BUFFER_PACKETS.saturating_sub(remaining as u32).max(1);
                    if out_tx.send(codec::encode(&Packet::Continue { id, buffer })).await.is_err() {
                        return Some(reason::NETWORK_ERROR);
                    }
                }
            }
            let _ = wr.shutdown().await;
            None
        };
        tokio::pin!(reader);
        tokio::pin!(writer);
        // The stream ends when the server stops sending, or when writing to it fails.
        let why = tokio::select! {
            why = &mut reader => why,
            Some(why) = &mut writer => why,
        };
        finish(why);
    });
    StreamEntry { generation, to_tcp, queued, task }
}

/// Resolve and connect, enforcing the address policy on every resolved address.
async fn connect(host: &str, port: u16, config: &Config) -> Result<TcpStream, u8> {
    let bare = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')).unwrap_or(host);
    let addrs: Vec<SocketAddr> = match tokio::time::timeout(CONNECT_TIMEOUT, tokio::net::lookup_host((bare, port))).await {
        Err(_) => return Err(reason::CONNECT_TIMEOUT),
        Ok(Err(_)) => return Err(reason::UNREACHABLE_HOST),
        Ok(Ok(addrs)) => addrs.collect(),
    };
    if addrs.is_empty() {
        return Err(reason::UNREACHABLE_HOST);
    }
    if !config.allow_private && addrs.iter().any(|a| policy::is_forbidden(a.ip())) {
        return Err(reason::BLOCKED);
    }
    // Try each address in order (e.g. IPv6 then IPv4) until one connects.
    let mut last = reason::UNREACHABLE_HOST;
    for addr in addrs {
        match tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(addr)).await {
            Err(_) => last = reason::CONNECT_TIMEOUT,
            Ok(Err(e)) if e.kind() == std::io::ErrorKind::ConnectionRefused => last = reason::REFUSED,
            Ok(Err(_)) => last = reason::UNREACHABLE_HOST,
            Ok(Ok(tcp)) => return Ok(tcp),
        }
    }
    Err(last)
}
