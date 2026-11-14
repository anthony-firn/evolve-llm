//! End-to-end tests: a real relay on an ephemeral port, a WebSocket client speaking Wisp, and
//! local TCP servers.

use evolve_relay::codec::{self, reason, Packet, STREAM_TCP};
use evolve_relay::{serve, Config};
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

async fn start_relay(config: Config) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(serve(listener, Arc::new(config)));
    addr
}

/// Echoes everything back; records whether the client closed its side.
async fn start_echo() -> (u16, tokio::sync::oneshot::Receiver<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (eof_tx, eof_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = vec![0u8; 65536];
        loop {
            match sock.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if sock.write_all(&buf[..n]).await.is_err() {
                        break;
                    }
                }
            }
        }
        let _ = eof_tx.send(());
    });
    (port, eof_rx)
}

async fn connect_ws(relay: SocketAddr, origin: Option<&str>) -> Result<Ws, tokio_tungstenite::tungstenite::Error> {
    let mut req = format!("ws://{relay}/").into_client_request().unwrap();
    if let Some(origin) = origin {
        req.headers_mut().insert("Origin", HeaderValue::from_str(origin).unwrap());
    }
    tokio_tungstenite::connect_async(req).await.map(|(ws, _)| ws)
}

async fn recv(ws: &mut Ws) -> Packet {
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(10), ws.next()).await.expect("relay replied in time").expect("open").expect("ok");
        if let Message::Binary(b) = msg {
            return codec::decode(&b).expect("valid packet");
        }
    }
}

async fn send(ws: &mut Ws, p: Packet) {
    ws.send(Message::Binary(codec::encode(&p).into())).await.unwrap();
}

fn private_config(extra_port: u16) -> Config {
    Config { allow_private: true, allow_ports: vec![extra_port], ..Config::default() }
}

#[tokio::test]
async fn forwards_a_tcp_stream_with_flow_control() {
    let (echo_port, eof) = start_echo().await;
    let relay = start_relay(private_config(echo_port)).await;
    let mut ws = connect_ws(relay, Some("http://localhost:5173")).await.expect("allowed origin connects");
    assert_eq!(recv(&mut ws).await, Packet::Continue { id: 0, buffer: 128 });

    send(&mut ws, Packet::Connect { id: 1, kind: STREAM_TCP, port: echo_port, host: "127.0.0.1".into() }).await;
    send(&mut ws, Packet::Data { id: 1, payload: b"hello relay".to_vec() }).await;
    let mut got = Vec::new();
    while got.len() < 11 {
        match recv(&mut ws).await {
            Packet::Data { id: 1, payload } => got.extend(payload),
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(got, b"hello relay");

    // 70 more packets: the relay must replenish credit after 64 forwarded packets.
    for i in 0..70u8 {
        send(&mut ws, Packet::Data { id: 1, payload: vec![i] }).await;
    }
    let mut echoed = 0;
    let mut continued = false;
    while echoed < 70 || !continued {
        match recv(&mut ws).await {
            Packet::Data { id: 1, payload } => echoed += payload.len(),
            Packet::Continue { id: 1, buffer } => {
                assert!((1..=128).contains(&buffer));
                continued = true;
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    send(&mut ws, Packet::Close { id: 1, reason: reason::VOLUNTARY }).await;
    tokio::time::timeout(Duration::from_secs(5), eof).await.expect("echo server saw the socket close").unwrap();
}

#[tokio::test]
async fn reports_server_eof_as_voluntary_close() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        sock.write_all(b"bye").await.unwrap();
        sock.shutdown().await.unwrap();
    });
    let relay = start_relay(private_config(port)).await;
    let mut ws = connect_ws(relay, Some("http://127.0.0.1:8080")).await.unwrap();
    recv(&mut ws).await;
    send(&mut ws, Packet::Connect { id: 5, kind: STREAM_TCP, port, host: "localhost".into() }).await;
    let mut got = Vec::new();
    loop {
        match recv(&mut ws).await {
            Packet::Data { id: 5, payload } => got.extend(payload),
            Packet::Close { id: 5, reason } => {
                assert_eq!(reason, reason::VOLUNTARY);
                break;
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(got, b"bye");
}

#[tokio::test]
async fn rejects_disallowed_and_missing_origins() {
    let relay = start_relay(Config::default()).await;
    for origin in [Some("https://evil.test"), Some("null"), None] {
        match connect_ws(relay, origin).await {
            Err(tokio_tungstenite::tungstenite::Error::Http(resp)) => assert_eq!(resp.status(), 403, "{origin:?}"),
            Err(e) => panic!("{origin:?}: expected HTTP 403, got {e}"),
            Ok(_) => panic!("{origin:?}: connection should have been refused"),
        }
    }
    assert!(connect_ws(relay, Some("https://www.evolvetech.dev")).await.is_ok());
}

#[tokio::test]
async fn refuses_private_addresses_bad_ports_and_stream_types() {
    let (echo_port, _eof) = start_echo().await;
    // Default config: private destinations and non-web ports are blocked.
    let relay = start_relay(Config { allow_ports: vec![echo_port], ..Config::default() }).await;
    let mut ws = connect_ws(relay, Some("http://localhost:3000")).await.unwrap();
    recv(&mut ws).await;

    send(&mut ws, Packet::Connect { id: 1, kind: STREAM_TCP, port: echo_port, host: "127.0.0.1".into() }).await;
    assert_eq!(recv(&mut ws).await, Packet::Close { id: 1, reason: reason::BLOCKED });

    send(&mut ws, Packet::Connect { id: 2, kind: STREAM_TCP, port: echo_port, host: "localhost".into() }).await;
    assert_eq!(recv(&mut ws).await, Packet::Close { id: 2, reason: reason::BLOCKED });

    send(&mut ws, Packet::Connect { id: 3, kind: STREAM_TCP, port: 443, host: "::ffff:10.0.0.1".into() }).await;
    assert_eq!(recv(&mut ws).await, Packet::Close { id: 3, reason: reason::BLOCKED });

    send(&mut ws, Packet::Connect { id: 4, kind: STREAM_TCP, port: 22, host: "example.com".into() }).await;
    assert_eq!(recv(&mut ws).await, Packet::Close { id: 4, reason: reason::BLOCKED });

    send(&mut ws, Packet::Connect { id: 5, kind: 0x02, port: 443, host: "example.com".into() }).await;
    assert_eq!(recv(&mut ws).await, Packet::Close { id: 5, reason: reason::INVALID_STREAM });

    send(&mut ws, Packet::Connect { id: 6, kind: STREAM_TCP, port: 443, host: "bad host".into() }).await;
    assert_eq!(recv(&mut ws).await, Packet::Close { id: 6, reason: reason::UNREACHABLE_HOST });

    // With --allow-private, a closed local port is reported as refused.
    let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let closed_port = closed.local_addr().unwrap().port();
    drop(closed);
    let relay = start_relay(private_config(closed_port)).await;
    let mut ws = connect_ws(relay, Some("http://localhost")).await.unwrap();
    recv(&mut ws).await;
    send(&mut ws, Packet::Connect { id: 7, kind: STREAM_TCP, port: closed_port, host: "127.0.0.1".into() }).await;
    assert_eq!(recv(&mut ws).await, Packet::Close { id: 7, reason: reason::REFUSED });
    assert_eq!(reason::REFUSED, 0x44);

    // A name that cannot resolve is an unreachable host (0x42).
    send(&mut ws, Packet::Connect { id: 8, kind: STREAM_TCP, port: closed_port, host: "nonexistent.invalid".into() }).await;
    assert_eq!(recv(&mut ws).await, Packet::Close { id: 8, reason: 0x42 });
}

/// Live check against the public internet: `cargo test -- --ignored`.
#[tokio::test]
#[ignore]
async fn fetches_a_public_page() {
    let relay = start_relay(Config::default()).await;
    let mut ws = connect_ws(relay, Some("https://www.evolvetech.dev")).await.unwrap();
    recv(&mut ws).await;
    send(&mut ws, Packet::Connect { id: 1, kind: STREAM_TCP, port: 80, host: "example.com".into() }).await;
    send(&mut ws, Packet::Data { id: 1, payload: b"GET / HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n".to_vec() }).await;
    let mut got = Vec::new();
    loop {
        match recv(&mut ws).await {
            Packet::Data { id: 1, payload } => got.extend(payload),
            Packet::Close { id: 1, reason } => {
                assert_eq!(reason, reason::VOLUNTARY);
                break;
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    let text = String::from_utf8_lossy(&got);
    assert!(text.starts_with("HTTP/1.1 200"), "{}", &text[..text.len().min(200)]);
    assert!(text.contains("Example Domain"));
}

async fn connect_as(relay: SocketAddr, client: &str) -> Result<Ws, tokio_tungstenite::tungstenite::Error> {
    let mut req = format!("ws://{relay}/").into_client_request().unwrap();
    req.headers_mut().insert("Origin", HeaderValue::from_static("https://www.evolvetech.dev"));
    req.headers_mut().insert("X-Forwarded-For", HeaderValue::from_str(client).unwrap());
    tokio_tungstenite::connect_async(req).await.map(|(ws, _)| ws)
}

#[tokio::test]
async fn public_mode_limits_each_client_address() {
    let mut config = Config::default().public();
    config.max_per_client = 1;
    config.max_streams_per_client = 1;
    let (echo_port, _eof) = start_echo().await;
    config.allow_private = true;
    config.allow_ports = vec![echo_port];
    let relay = start_relay(config).await;
    let mut first = connect_as(relay, "198.51.100.1").await.expect("first connection");
    assert!(matches!(recv(&mut first).await, Packet::Continue { id: 0, .. }));
    match connect_as(relay, "198.51.100.1, 10.0.0.1").await {
        Err(tokio_tungstenite::tungstenite::Error::Http(resp)) => assert_eq!(resp.status(), 429),
        other => panic!("expected 429, got {:?}", other.map(|_| ())),
    }
    let mut other = connect_as(relay, "198.51.100.2").await.expect("another client is unaffected");
    assert!(matches!(recv(&mut other).await, Packet::Continue { id: 0, .. }));
    // One stream per client: the second is throttled.
    send(&mut first, Packet::Connect { id: 1, kind: STREAM_TCP, port: echo_port, host: "127.0.0.1".into() }).await;
    send(&mut first, Packet::Connect { id: 2, kind: STREAM_TCP, port: echo_port, host: "127.0.0.1".into() }).await;
    loop {
        match recv(&mut first).await {
            Packet::Close { id: 2, reason: r } => {
                assert_eq!(r, reason::THROTTLED);
                break;
            }
            Packet::Close { id: 1, .. } => panic!("first stream should stay open"),
            _ => {}
        }
    }
    // Closing the first connection frees the address's slot.
    drop(first);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(connect_as(relay, "198.51.100.1").await.is_ok());
}
