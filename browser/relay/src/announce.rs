//! Announcing a public relay on Nostr, so browsers can find independently operated relays.
//!
//! A relay started with `--public --announce KEY` publishes a signed, parameterized replaceable
//! Nostr event (kind [`KIND`], NIP-01/NIP-33) describing itself to a few public Nostr relays, on
//! startup and every [`REPUBLISH`]. Each announcement expires after [`LIFETIME`] (NIP-40), so a
//! relay that stops simply disappears from the directory. Browsers verify the signature, drop
//! expired announcements and health-check the relay before offering it (`web/relay-directory.mjs`).
//!
//! An announcement is a claim by its operator, not proof of uptime or honesty. The engine
//! negotiates TLS end to end, so a relay still only sees encrypted bytes for HTTPS sites.

use futures_util::{SinkExt, StreamExt};
use k256::schnorr::{Signature, SigningKey, VerifyingKey};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio_tungstenite::tungstenite::Message;

/// Nostr event kind for relay announcements (parameterized replaceable range, NIP-01).
pub const KIND: u64 = 38787;
/// Value of the `t` tag that every announcement carries, so clients can filter on it.
pub const TOPIC: &str = "evolve-relay";
/// How long an announcement stays valid (NIP-40 `expiration`).
pub const LIFETIME: Duration = Duration::from_secs(2 * 3600);
/// How often a running relay publishes a fresh announcement.
pub const REPUBLISH: Duration = Duration::from_secs(30 * 60);
/// The public Nostr relays Fairground uses by default.
pub const DEFAULT_NOSTR_RELAYS: &[&str] = &["wss://relay.damus.io", "wss://nos.lol", "wss://relay.primal.net", "wss://offchain.pub"];

/// What a relay says about itself.
#[derive(Debug, Clone, Default)]
pub struct Announcement {
    /// The `wss://` address browsers connect to.
    pub public_url: String,
    pub name: Option<String>,
    pub region: Option<String>,
    pub contact: Option<String>,
    /// Per client address: connections / open streams / bytes per hour (0 = no limit).
    pub limits: (usize, usize, u64),
    /// Destination ports the relay allows.
    pub ports: Vec<u16>,
}

impl Announcement {
    pub fn from_config(public_url: &str, config: &crate::Config) -> Self {
        let mut ports: Vec<u16> = crate::policy::DEFAULT_PORTS.to_vec();
        ports.extend(config.allow_ports.iter().copied().filter(|p| !crate::policy::DEFAULT_PORTS.contains(p)));
        Announcement {
            public_url: public_url.to_string(),
            limits: (config.max_per_client, config.max_streams_per_client, config.client_bytes_per_hour),
            ports,
            ..Default::default()
        }
    }

    pub fn tags(&self, expires_at: u64) -> Vec<Vec<String>> {
        let s = |v: &str| v.to_string();
        let mut tags = vec![
            vec![s("d"), self.public_url.clone()],
            vec![s("u"), self.public_url.clone()],
            vec![s("proto"), s("wisp/1")],
            vec![s("limits"), format!("{}/{}/{}", self.limits.0, self.limits.1, self.limits.2)],
            vec![s("ports"), self.ports.iter().map(u16::to_string).collect::<Vec<_>>().join(",")],
        ];
        for (key, value) in [("name", &self.name), ("region", &self.region), ("contact", &self.contact)] {
            if let Some(v) = value.as_deref().filter(|v| !v.trim().is_empty()) {
                tags.push(vec![s(key), v.trim().to_string()]);
            }
        }
        tags.push(vec![s("expiration"), expires_at.to_string()]);
        tags.push(vec![s("t"), s(TOPIC)]);
        tags
    }
}

/// Checks that `url` is a `wss://` address with a host and no credentials, query or fragment.
pub fn check_public_url(url: &str) -> Result<(), String> {
    let rest = url.strip_prefix("wss://").ok_or_else(|| format!("--public-url must start with wss://, got {url:?}"))?;
    let host = rest.split(['/', ':']).next().unwrap_or("");
    if host.is_empty() || rest.contains(['@', '?', '#', ' ']) {
        return Err(format!("--public-url must look like wss://relay.example.com/, got {url:?}"));
    }
    Ok(())
}

/// NIP-01 event id: SHA-256 of the compact JSON `[0, pubkey, created_at, kind, tags, content]`.
/// serde_json escapes exactly as NIP-01 requires (`\n \" \\ \r \t \b \f`, other control characters
/// as `\u00XX`, everything else verbatim).
pub fn event_id(pubkey_hex: &str, created_at: u64, kind: u64, tags: &[Vec<String>], content: &str) -> [u8; 32] {
    let serialized = serde_json::to_string(&json!([0, pubkey_hex, created_at, kind, tags, content])).expect("serializable");
    Sha256::digest(serialized.as_bytes()).into()
}

/// Builds and signs an announcement event. `aux_rand` is BIP-340 auxiliary randomness.
pub fn sign_event(key: &SigningKey, tags: Vec<Vec<String>>, content: &str, created_at: u64, aux_rand: &[u8; 32]) -> Value {
    let pubkey = hex(&key.verifying_key().to_bytes());
    let id = event_id(&pubkey, created_at, KIND, &tags, content);
    let sig = key.sign_raw(&id, aux_rand).expect("BIP-340 signing");
    json!({
        "id": hex(&id),
        "pubkey": pubkey,
        "created_at": created_at,
        "kind": KIND,
        "tags": tags,
        "content": content,
        "sig": hex(&sig.to_bytes()),
    })
}

/// Checks an event's id and signature (used by tests and as a self-check before publishing).
pub fn verify_event(event: &Value) -> Result<(), String> {
    let field = |name: &str| event.get(name).ok_or_else(|| format!("event has no {name}"));
    let pubkey = field("pubkey")?.as_str().ok_or("pubkey is not a string")?;
    let created_at = field("created_at")?.as_u64().ok_or("created_at is not an integer")?;
    let kind = field("kind")?.as_u64().ok_or("kind is not an integer")?;
    let tags: Vec<Vec<String>> = serde_json::from_value(field("tags")?.clone()).map_err(|e| format!("bad tags: {e}"))?;
    let content = field("content")?.as_str().ok_or("content is not a string")?;
    let id = event_id(pubkey, created_at, kind, &tags, content);
    if field("id")?.as_str() != Some(hex(&id).as_str()) {
        return Err("event id does not match its contents".into());
    }
    let key = VerifyingKey::from_bytes(&unhex(pubkey)?).map_err(|_| "invalid public key")?;
    let sig = Signature::try_from(unhex(field("sig")?.as_str().ok_or("sig is not a string")?)?.as_slice()).map_err(|_| "invalid signature encoding")?;
    key.verify_raw(&id, &sig).map_err(|_| "signature does not verify".to_string())
}

pub fn announcement_event(key: &SigningKey, ann: &Announcement, now: u64) -> Value {
    let tags = ann.tags(now + LIFETIME.as_secs());
    sign_event(key, tags, "Wisp v1 relay for Evolve Browser", now, &random32())
}

// ---------------------------------------------------------------------------------------------
// Keys

/// Creates a new secret key file (64 hex characters, mode 0600). Refuses to overwrite a file.
/// Returns the public key.
pub fn keygen(path: &Path) -> Result<VerifyingKey, String> {
    let key = loop {
        if let Ok(k) = SigningKey::from_bytes(&random32()) {
            break k;
        }
    };
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options.open(path).map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    std::io::Write::write_all(&mut file, format!("{}\n", hex(&key.to_bytes())).as_bytes()).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(*key.verifying_key())
}

/// Reads a key file written by [`keygen`] (a 32-byte secret as hex). Warns when other users can
/// read it.
pub fn load_key(path: &Path) -> Result<SigningKey, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read key file {}: {e}", path.display()))?;
    #[cfg(unix)]
    if let Ok(meta) = std::fs::metadata(path) {
        if std::os::unix::fs::PermissionsExt::mode(&meta.permissions()) & 0o077 != 0 {
            eprintln!("evolve-relay: warning: {} is readable by other users; run chmod 600 on it", path.display());
        }
    }
    let bytes = unhex(text.trim()).map_err(|_| format!("{} must hold a 64-character hex secret key", path.display()))?;
    if bytes.len() != 32 {
        return Err(format!("{} must hold a 64-character hex secret key", path.display()));
    }
    SigningKey::from_bytes(&bytes).map_err(|_| format!("{} does not hold a valid secp256k1 secret key", path.display()))
}

/// NIP-19 `npub` (bech32) for a public key.
pub fn npub(key: &VerifyingKey) -> String {
    bech32("npub", &key.to_bytes())
}

// ---------------------------------------------------------------------------------------------
// Publishing

fn tls_config() -> Arc<rustls::ClientConfig> {
    let roots = rustls::RootCertStore { roots: webpki_roots::TLS_SERVER_ROOTS.to_vec() };
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()
        .expect("TLS versions")
        .with_root_certificates(roots)
        .with_no_client_auth();
    Arc::new(config)
}

pub type Socket = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Opens a WebSocket (`ws://` or `wss://` with the Mozilla root store) to a Nostr relay or relay.
pub async fn connect(request: impl tokio_tungstenite::tungstenite::client::IntoClientRequest + Unpin) -> Result<Socket, String> {
    let connector = tokio_tungstenite::Connector::Rustls(tls_config());
    let (ws, _) = tokio_tungstenite::connect_async_tls_with_config(request, None, false, Some(connector)).await.map_err(|e| format!("cannot connect: {e}"))?;
    Ok(ws)
}

/// Sends `["EVENT", event]` to one Nostr relay and waits for its `OK`. Returns the relay's message
/// on acceptance, or why it failed.
pub async fn publish(relay_url: &str, event: &Value, timeout: Duration) -> Result<String, String> {
    let id = event["id"].as_str().unwrap_or_default().to_string();
    let attempt = async {
        let mut ws = connect(relay_url).await?;
        ws.send(Message::text(json!(["EVENT", event]).to_string())).await.map_err(|e| format!("cannot send: {e}"))?;
        while let Some(message) = ws.next().await {
            let text = match message.map_err(|e| format!("connection failed: {e}"))? {
                Message::Text(t) => t.to_string(),
                Message::Close(_) => break,
                _ => continue,
            };
            let Ok(Value::Array(reply)) = serde_json::from_str::<Value>(&text) else { continue };
            match reply.first().and_then(Value::as_str) {
                Some("OK") if reply.get(1).and_then(Value::as_str) == Some(id.as_str()) => {
                    let message = reply.get(3).and_then(Value::as_str).unwrap_or("").to_string();
                    let _ = ws.close(None).await;
                    return if reply.get(2).and_then(Value::as_bool) == Some(true) { Ok(message) } else { Err(format!("rejected: {message}")) };
                }
                Some("NOTICE") => {
                    let notice = reply.get(1).and_then(Value::as_str).unwrap_or("");
                    if !notice.is_empty() {
                        eprintln!("evolve-relay: notice from {relay_url}: {notice}");
                    }
                }
                _ => {}
            }
        }
        Err("closed before acknowledging the event".to_string())
    };
    match tokio::time::timeout(timeout, attempt).await {
        Ok(result) => result,
        Err(_) => Err(format!("no reply within {} seconds", timeout.as_secs())),
    }
}

/// Opens the relay's own public address as a browser would and waits for the Wisp `CONTINUE`
/// packet, so a relay that is not reachable is not announced.
pub async fn self_check(public_url: &str, origin: &str, timeout: Duration) -> Result<(), String> {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    let attempt = async {
        let mut request = public_url.into_client_request().map_err(|e| format!("bad address: {e}"))?;
        request.headers_mut().insert("Origin", origin.parse().map_err(|_| "bad origin")?);
        let mut ws = connect(request).await?;
        while let Some(message) = ws.next().await {
            if let Message::Binary(b) = message.map_err(|e| e.to_string())? {
                let _ = ws.close(None).await;
                return if b.len() == 9 && b[0] == 0x03 && b[1..5] == [0, 0, 0, 0] { Ok(()) } else { Err("the server did not speak Wisp v1".to_string()) };
            }
        }
        Err("closed without a Wisp CONTINUE packet".to_string())
    };
    tokio::time::timeout(timeout, attempt).await.unwrap_or_else(|_| Err(format!("no reply within {} seconds", timeout.as_secs())))
}

/// Publishes an announcement now and every [`REPUBLISH`] until the process stops. Skips a round
/// (and retries in five minutes) while the relay's public address does not answer.
pub async fn run(key: SigningKey, ann: Announcement, nostr_relays: Vec<String>, origin: String) {
    loop {
        let next = match self_check(&ann.public_url, &origin, Duration::from_secs(15)).await {
            Err(e) => {
                eprintln!("evolve-relay: not announcing yet: {} does not answer ({e}); retrying in 5 minutes", ann.public_url);
                Duration::from_secs(5 * 60)
            }
            Ok(()) => {
                let event = announcement_event(&key, &ann, unix_now());
                let results = futures_util::future::join_all(nostr_relays.iter().map(|r| publish(r, &event, Duration::from_secs(15)))).await;
                let accepted = results.iter().filter(|r| r.is_ok()).count();
                for (relay, result) in nostr_relays.iter().zip(&results) {
                    if let Err(e) = result {
                        eprintln!("evolve-relay: announcement not accepted by {relay}: {e}");
                    }
                }
                eprintln!("evolve-relay: announced {} on {accepted} of {} Nostr relays (expires in {} h)", ann.public_url, nostr_relays.len(), LIFETIME.as_secs() / 3600);
                REPUBLISH
            }
        };
        tokio::time::sleep(next).await;
    }
}

// ---------------------------------------------------------------------------------------------
// Helpers

pub fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn random32() -> [u8; 32] {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).expect("operating system randomness");
    bytes
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("not hex".into());
    }
    Ok((0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect())
}

/// BIP-173 bech32 (not bech32m), as NIP-19 uses.
fn bech32(hrp: &str, data: &[u8]) -> String {
    const CHARSET: &[u8] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    fn polymod(values: &[u8]) -> u32 {
        const GEN: [u32; 5] = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];
        let mut chk: u32 = 1;
        for v in values {
            let top = chk >> 25;
            chk = (chk & 0x1ffffff) << 5 ^ *v as u32;
            for (i, g) in GEN.iter().enumerate() {
                if (top >> i) & 1 == 1 {
                    chk ^= g;
                }
            }
        }
        chk
    }
    // 8-bit bytes to 5-bit groups, padded
    let mut five = Vec::new();
    let (mut acc, mut bits) = (0u32, 0u32);
    for b in data {
        acc = (acc << 8) | *b as u32;
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            five.push(((acc >> bits) & 31) as u8);
        }
    }
    if bits > 0 {
        five.push(((acc << (5 - bits)) & 31) as u8);
    }
    let mut values: Vec<u8> = hrp.bytes().map(|c| c >> 5).collect();
    values.push(0);
    values.extend(hrp.bytes().map(|c| c & 31));
    values.extend(&five);
    values.extend([0u8; 6]);
    let pm = polymod(&values) ^ 1;
    let checksum = (0..6).map(|i| ((pm >> (5 * (5 - i))) & 31) as u8);
    let mut out = format!("{hrp}1");
    out.extend(five.into_iter().chain(checksum).map(|v| CHARSET[v as usize] as char));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The official BIP-340 vectors (bitcoin/bips bip-0340/test-vectors.csv), shared with the
    /// JavaScript tests.
    const VECTORS: &str = include_str!("../../../tests/fixtures/bip340-vectors.csv");

    #[test]
    fn bip340_official_vectors() {
        let mut checked = 0;
        for line in VECTORS.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.splitn(8, ',').collect();
            let (secret, public, aux, msg, sig, expected) = (f[1], f[2], f[3], f[4], f[5], f[6] == "TRUE");
            let msg = unhex(msg).unwrap();
            if !secret.is_empty() {
                let key = SigningKey::from_bytes(&unhex(secret).unwrap()).unwrap();
                assert_eq!(hex(&key.verifying_key().to_bytes()), public.to_lowercase(), "vector {}", f[0]);
                let aux: [u8; 32] = unhex(aux).unwrap().try_into().unwrap();
                assert_eq!(hex(&key.sign_raw(&msg, &aux).unwrap().to_bytes()), sig.to_lowercase(), "vector {}", f[0]);
            }
            let verified = VerifyingKey::from_bytes(&unhex(public).unwrap())
                .ok()
                .zip(Signature::try_from(unhex(sig).unwrap().as_slice()).ok())
                .is_some_and(|(k, s)| k.verify_raw(&msg, &s).is_ok());
            assert_eq!(verified, expected, "vector {}: {}", f[0], f.get(7).unwrap_or(&""));
            checked += 1;
        }
        assert_eq!(checked, 19);
    }

    #[test]
    fn event_id_follows_nip01_serialization() {
        // Escaping and field order as NIP-01 specifies, checked against a hand-written string.
        let tags = vec![vec!["d".to_string(), "wss://r.example/".to_string()]];
        let content = "line\nquote\" back\\ tab\t é /";
        let expected = "[0,\"ab\",7,38787,[[\"d\",\"wss://r.example/\"]],\"line\\nquote\\\" back\\\\ tab\\t é /\"]";
        assert_eq!(event_id("ab", 7, KIND, &tags, content), <[u8; 32]>::from(Sha256::digest(expected.as_bytes())));
    }

    #[test]
    fn signed_announcement_verifies_and_matches_the_shared_fixture() {
        // Deterministic: secret key 3, aux_rand zero. tests/relay-directory.test.mjs checks the
        // same event with the browser's verifier.
        let key = SigningKey::from_bytes(&unhex(&format!("{:064x}", 3)).unwrap()).unwrap();
        let ann = Announcement {
            public_url: "wss://relay.example.com/".into(),
            name: Some("Example relay".into()),
            region: Some("us-west".into()),
            contact: Some("ops@example.com".into()),
            limits: (8, 128, 2 << 30),
            ports: vec![80, 443],
        };
        let event = sign_event(&key, ann.tags(1_700_007_200), "Wisp v1 relay for Evolve Browser", 1_700_000_000, &[0; 32]);
        verify_event(&event).unwrap();
        let fixture: Value = serde_json::from_str(include_str!("../../../tests/fixtures/relay-announcement.json")).unwrap();
        assert_eq!(event, fixture, "regenerate tests/fixtures/relay-announcement.json if the format changed:\n{event}");
        let tag = |k: &str| event["tags"].as_array().unwrap().iter().find(|t| t[0] == k).map(|t| t[1].as_str().unwrap().to_string());
        assert_eq!(tag("limits").as_deref(), Some("8/128/2147483648"));
        assert_eq!(tag("expiration").as_deref(), Some("1700007200"));
        assert_eq!(tag("t").as_deref(), Some(TOPIC));

        let mut tampered = event.clone();
        tampered["content"] = json!("changed");
        assert!(verify_event(&tampered).is_err());
        let mut forged = event.clone();
        forged["pubkey"] = json!(hex(&SigningKey::from_bytes(&unhex(&format!("{:064x}", 5)).unwrap()).unwrap().verifying_key().to_bytes()));
        assert!(verify_event(&forged).is_err());
    }

    #[test]
    fn npub_matches_nip19_example() {
        let key = VerifyingKey::from_bytes(&unhex("3bf0c63fcb93463407af97a5e5ee64fa883d107ef9e558472c4eb9aaaefa459d").unwrap()).unwrap();
        assert_eq!(npub(&key), "npub180cvv07tjdrrgpa0j7j7tmnyl2yr6yr7l8j4s3evf6u64th6gkwsyjh6w6");
    }

    #[test]
    fn public_url_rules() {
        assert!(check_public_url("wss://relay.example.com/").is_ok());
        assert!(check_public_url("wss://relay.example.com:8443/wisp/").is_ok());
        assert!(check_public_url("ws://relay.example.com/").is_err());
        assert!(check_public_url("https://relay.example.com/").is_err());
        assert!(check_public_url("wss:///").is_err());
        assert!(check_public_url("wss://user@relay.example.com/").is_err());
        assert!(check_public_url("wss://relay.example.com/?x=1").is_err());
    }

    #[test]
    fn keygen_writes_a_private_key_file_and_refuses_to_overwrite() {
        let dir = std::env::temp_dir().join(format!("evolve-relay-keygen-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("key");
        let _ = std::fs::remove_file(&path);
        let public = keygen(&path).unwrap();
        #[cfg(unix)]
        assert_eq!(std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(&path).unwrap().permissions()) & 0o777, 0o600);
        assert_eq!(*load_key(&path).unwrap().verifying_key(), public);
        assert!(keygen(&path).is_err());
        std::fs::write(&path, "not a key").unwrap();
        assert!(load_key(&path).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
