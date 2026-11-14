//! Announcing on Nostr: a fake Nostr relay on an ephemeral port receives the signed event, and
//! the self-check talks to a real evolve-relay.

use evolve_relay::announce::{self, Announcement};
use evolve_relay::{serve, Config};
use futures_util::{SinkExt, StreamExt};
use k256::schnorr::SigningKey;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

#[derive(Clone, Copy)]
enum Reply {
    Accept,
    Reject,
    Silent,
}

/// A minimal Nostr relay: forwards every EVENT it receives and answers with `reply`.
async fn fake_nostr(reply: Reply) -> (String, mpsc::UnboundedReceiver<Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/", listener.local_addr().unwrap());
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Ok((sock, _)) = listener.accept().await {
            let tx = tx.clone();
            tokio::spawn(async move {
                let mut ws = tokio_tungstenite::accept_async(sock).await.unwrap();
                while let Some(Ok(msg)) = ws.next().await {
                    let Message::Text(text) = msg else { continue };
                    let frame: Value = serde_json::from_str(&text).unwrap();
                    assert_eq!(frame[0], "EVENT", "clients publish with EVENT");
                    let event = frame[1].clone();
                    let id = event["id"].clone();
                    tx.send(event).unwrap();
                    let answer = match reply {
                        Reply::Accept => json!(["OK", id, true, ""]),
                        Reply::Reject => json!(["OK", id, false, "blocked: not on the allow list"]),
                        Reply::Silent => continue,
                    };
                    ws.send(Message::text(json!(["NOTICE", "hello"]).to_string())).await.unwrap();
                    ws.send(Message::text(answer.to_string())).await.unwrap();
                }
            });
        }
    });
    (url, rx)
}

async fn start_relay(config: Config) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(serve(listener, Arc::new(config)));
    addr
}

fn key(n: u8) -> SigningKey {
    let mut bytes = [0u8; 32];
    bytes[31] = n;
    SigningKey::from_bytes(&bytes).unwrap()
}

fn tag<'a>(event: &'a Value, name: &str) -> Option<&'a str> {
    event["tags"].as_array()?.iter().find(|t| t[0] == name)?[1].as_str()
}

#[tokio::test]
async fn publish_sends_a_signed_event_and_reads_ok() {
    let (url, mut events) = fake_nostr(Reply::Accept).await;
    let ann = Announcement::from_config("wss://relay.example.com/", &Config::default().public());
    let event = announce::announcement_event(&key(7), &ann, announce::unix_now());
    assert_eq!(announce::publish(&url, &event, Duration::from_secs(5)).await, Ok(String::new()));
    let received = events.recv().await.unwrap();
    assert_eq!(received, event);
    announce::verify_event(&received).unwrap();
    assert_eq!(received["kind"], 38787);
}

#[tokio::test]
async fn publish_reports_rejection_and_silence() {
    let (url, _events) = fake_nostr(Reply::Reject).await;
    let event = announce::announcement_event(&key(7), &Announcement { public_url: "wss://r.example/".into(), ..Default::default() }, 1);
    let err = announce::publish(&url, &event, Duration::from_secs(5)).await.unwrap_err();
    assert!(err.contains("blocked: not on the allow list"), "{err}");

    let (url, _events) = fake_nostr(Reply::Silent).await;
    let err = announce::publish(&url, &event, Duration::from_millis(500)).await.unwrap_err();
    assert!(err.contains("no reply"), "{err}");

    let err = announce::publish("ws://127.0.0.1:1/", &event, Duration::from_secs(5)).await.unwrap_err();
    assert!(err.contains("cannot connect"), "{err}");
}

#[tokio::test]
async fn self_check_requires_a_reachable_wisp_relay_and_an_allowed_origin() {
    let relay = start_relay(Config::default()).await;
    let url = format!("ws://{relay}/");
    announce::self_check(&url, "http://localhost:3000", Duration::from_secs(5)).await.unwrap();
    assert!(announce::self_check(&url, "https://attacker.example", Duration::from_secs(5)).await.is_err());
    // A WebSocket server that is not a Wisp relay
    let (nostr, _events) = fake_nostr(Reply::Accept).await;
    assert!(announce::self_check(&nostr, "http://localhost:3000", Duration::from_millis(500)).await.is_err());
}

#[tokio::test]
async fn running_relay_announces_itself_with_its_limits() {
    let config = Config::default().public();
    let relay = start_relay(config.clone()).await;
    let (nostr, mut events) = fake_nostr(Reply::Accept).await;
    let mut ann = Announcement::from_config(&format!("ws://{relay}/"), &config);
    ann.name = Some("Test relay".into());
    ann.region = Some("local".into());
    let signer = key(9);
    let pubkey = announce::hex(&signer.verifying_key().to_bytes());
    tokio::spawn(announce::run(signer, ann, vec![nostr], "http://localhost:3000".into()));
    let event = tokio::time::timeout(Duration::from_secs(10), events.recv()).await.expect("announced in time").unwrap();
    announce::verify_event(&event).unwrap();
    assert_eq!(event["pubkey"], pubkey);
    assert_eq!(tag(&event, "d"), Some(format!("ws://{relay}/").as_str()));
    assert_eq!(tag(&event, "u"), tag(&event, "d"));
    assert_eq!(tag(&event, "proto"), Some("wisp/1"));
    assert_eq!(tag(&event, "limits"), Some("8/128/2147483648"));
    assert_eq!(tag(&event, "ports"), Some("80,443"));
    assert_eq!(tag(&event, "name"), Some("Test relay"));
    assert_eq!(tag(&event, "t"), Some("evolve-relay"));
    assert_eq!(tag(&event, "contact"), None);
    let created = event["created_at"].as_u64().unwrap();
    assert_eq!(tag(&event, "expiration").unwrap().parse::<u64>().unwrap(), created + 7200);
}

/// Read-only: opens TLS WebSockets to the default public Nostr relays and asks for
/// announcements (no publishing). At least one must answer. `cargo test -- --ignored`.
#[tokio::test]
#[ignore]
async fn queries_public_nostr_relays_over_tls() {
    let mut answered = vec![];
    for relay in announce::DEFAULT_NOSTR_RELAYS {
        let query = async {
            let mut ws = announce::connect(*relay).await?;
            let req = json!(["REQ", "evolve-test", {"kinds": [announce::KIND], "#t": [announce::TOPIC], "limit": 5}]);
            ws.send(Message::text(req.to_string())).await.map_err(|e| e.to_string())?;
            while let Some(msg) = ws.next().await {
                if let Message::Text(t) = msg.map_err(|e| e.to_string())? {
                    let frame: Value = serde_json::from_str(&t).map_err(|e| e.to_string())?;
                    if frame[0] == "EOSE" {
                        return Ok(());
                    }
                    if frame[0] == "EVENT" {
                        announce::verify_event(&frame[2])?;
                    }
                }
            }
            Err("closed".to_string())
        };
        let result = tokio::time::timeout(Duration::from_secs(15), query).await.unwrap_or_else(|_| Err("timeout".into()));
        eprintln!("{relay}: {result:?}");
        if result.is_ok() {
            answered.push(*relay);
        }
    }
    assert!(!answered.is_empty());
}
