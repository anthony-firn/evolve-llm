//! The Tor route: TCP streams through the Tor network, so the browser needs no relay of its own.
//!
//! The engine reaches Tor's Snowflake bridge over a WebSocket (WebSocket → Turbo → KCP → SMUX →
//! TLS → Tor channel, implemented by the vendored `webtor` crate on Arti's `tor-proto`), builds
//! three-hop circuits from the public consensus, and opens one Tor stream per TCP connection.
//! The exit relay resolves host names, so there are no DNS lookups outside Tor. HTTP and TLS run
//! in the engine on top of the stream exactly as they do over a Wisp relay: exits see only
//! encrypted bytes for https:// sites.
//!
//! Circuits are isolated by first-party site (like Tor Browser): every connection made for one
//! site's pages shares that site's circuit, and different sites never share one. A circuit takes
//! new streams for ten minutes (Tor's MaxCircuitDirtiness), and one spare circuit is kept ready
//! so a newly visited site does not wait for a build. This bookkeeping lives here, on the
//! engine's single thread, rather than in webtor's lock-based circuit manager.

use crate::host::{self, OutEvent};
use crate::net::client::with_timeout;
use futures::future::{FutureExt, LocalBoxFuture, Shared};
use futures::{AsyncReadExt, AsyncWriteExt};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use webtor::{TorClient, TorClientOptions};

/// Snowflake bridges reachable over WebSocket, tried in order.
const BRIDGES: [&str; 2] = [webtor::config::SNOWFLAKE_URL_PRIMARY, webtor::config::SNOWFLAKE_URL_SECONDARY];
/// How long a site's circuit takes new streams (Tor's MaxCircuitDirtiness).
const CIRCUIT_DIRTY_MS: f64 = 10.0 * 60_000.0;
/// How long an exit may take to open a stream before another circuit is tried.
const STREAM_OPEN_MS: i32 = 20_000;
/// Circuits tried per connection.
const ATTEMPTS: usize = 4;

type Tunnel = Arc<tor_proto::ClientTunnel>;
type Connecting = Shared<LocalBoxFuture<'static, Result<Rc<TorClient>, String>>>;
type Building = Shared<LocalBoxFuture<'static, Result<Tunnel, String>>>;

enum SiteCircuit {
    Ready { tunnel: Tunnel, since: f64 },
    Building(Building),
}

#[derive(Default)]
struct State {
    client: Option<Rc<TorClient>>,
    connecting: Option<Connecting>,
    sites: HashMap<String, SiteCircuit>,
    spare: Option<Tunnel>,
    spare_building: bool,
    /// Incremented on every reset, so work started for an old client cannot install results.
    generation: u64,
}

/// A lazily started Tor client shared by every connection of the browser.
#[derive(Clone, Default)]
pub struct TorNet(Rc<RefCell<State>>);

pub struct TorStream {
    inner: webtor::DataStream,
}

impl TorStream {
    /// Read the next chunk. `Ok(None)` is a clean end of stream.
    pub async fn read(&mut self) -> Result<Option<Vec<u8>>, String> {
        let mut buf = vec![0u8; 16 * 1024];
        match self.inner.read(&mut buf).await {
            Ok(0) => Ok(None),
            Ok(n) => {
                buf.truncate(n);
                Ok(Some(buf))
            }
            Err(e) => Err(format!("The Tor stream failed: {e}")),
        }
    }

    pub async fn write(&mut self, data: &[u8]) -> Result<(), String> {
        self.inner.write_all(data).await.map_err(|e| format!("The Tor stream failed: {e}"))?;
        self.inner.flush().await.map_err(|e| format!("The Tor stream failed: {e}"))
    }
}

fn status(state: &str, detail: impl Into<String>) {
    host::emit(OutEvent::Network { route: "tor".into(), state: state.into(), detail: detail.into() });
}

fn now() -> f64 {
    js_sys::Date::now()
}

fn usable(tunnel: &Tunnel) -> bool {
    !tunnel.is_closed()
}

impl TorNet {
    /// Where the cached consensus comes from, in order (see webtor's directory module).
    pub fn set_consensus_sources(urls: Vec<String>) {
        if !urls.is_empty() {
            webtor::directory::set_consensus_sources(urls);
        }
    }

    pub fn is_ready(&self) -> bool {
        self.0.borrow().client.is_some()
    }

    /// Forget the client and every circuit (after the bridge connection dropped).
    pub fn reset(&self) {
        let mut s = self.0.borrow_mut();
        if let Some(client) = s.client.take() {
            client.abort();
        }
        s.connecting = None;
        s.sites.clear();
        s.spare = None;
        s.spare_building = false;
        s.generation += 1;
    }

    async fn client(&self) -> Result<Rc<TorClient>, String> {
        let existing = self.0.borrow().client.clone();
        if let Some(client) = existing {
            if client.has_channel().await {
                return Ok(client);
            }
            status("connecting", "The connection to the Tor bridge dropped. Reconnecting…");
            self.reset();
        }
        let (pending, generation) = {
            let mut s = self.0.borrow_mut();
            let generation = s.generation;
            let pending = s.connecting.get_or_insert_with(|| start().boxed_local().shared()).clone();
            (pending, generation)
        };
        let result = pending.await;
        let installed = {
            let mut s = self.0.borrow_mut();
            if s.generation == generation && s.connecting.is_some() {
                s.connecting = None;
                if let Ok(client) = &result {
                    s.client = Some(client.clone());
                }
                true
            } else {
                false
            }
        };
        if let (true, Ok(client)) = (installed, &result) {
            self.spawn_spare(client);
        }
        result
    }

    /// Connect to Tor now, so the first page does not wait for the bridge and the first circuit.
    pub async fn warm(&self) -> Result<(), String> {
        self.client().await.map(|_| ())
    }

    /// Keep one ready circuit that no site uses yet.
    fn spawn_spare(&self, client: &Rc<TorClient>) {
        let generation = {
            let mut s = self.0.borrow_mut();
            if s.spare_building || s.spare.as_ref().is_some_and(usable) {
                return;
            }
            s.spare_building = true;
            s.generation
        };
        let (net, client) = (self.clone(), client.clone());
        wasm_bindgen_futures::spawn_local(async move {
            let built = client.build_circuit().await;
            let mut s = net.0.borrow_mut();
            if s.generation == generation {
                s.spare_building = false;
                if let Ok(tunnel) = built {
                    s.spare = Some(tunnel);
                }
            }
        });
    }

    /// The circuit for `site`: its current one, the spare, or a new build shared by every
    /// request for that site that arrives meanwhile.
    async fn site_circuit(&self, client: &Rc<TorClient>, site: &str) -> Result<Tunnel, String> {
        let (pending, generation) = {
            let mut s = self.0.borrow_mut();
            let generation = s.generation;
            match s.sites.get(site) {
                Some(SiteCircuit::Ready { tunnel, since }) if usable(tunnel) && now() - since < CIRCUIT_DIRTY_MS => {
                    return Ok(tunnel.clone());
                }
                Some(SiteCircuit::Building(pending)) => (pending.clone(), generation),
                _ => {
                    if let Some(tunnel) = s.spare.take().filter(usable) {
                        s.sites.insert(site.to_string(), SiteCircuit::Ready { tunnel: tunnel.clone(), since: now() });
                        drop(s);
                        self.spawn_spare(client);
                        return Ok(tunnel);
                    }
                    let c = client.clone();
                    let pending = async move { c.build_circuit().await.map_err(|e| e.to_string()) }.boxed_local().shared();
                    s.sites.insert(site.to_string(), SiteCircuit::Building(pending.clone()));
                    (pending, generation)
                }
            }
        };
        let result = pending.await;
        {
            let mut s = self.0.borrow_mut();
            if s.generation == generation && matches!(s.sites.get(site), Some(SiteCircuit::Building(_))) {
                match &result {
                    Ok(tunnel) => {
                        s.sites.insert(site.to_string(), SiteCircuit::Ready { tunnel: tunnel.clone(), since: now() });
                    }
                    Err(_) => {
                        s.sites.remove(site);
                    }
                }
            }
        }
        result
    }

    /// Stop using `tunnel` for `site` (it failed to open a stream).
    fn retire(&self, site: &str, tunnel: &Tunnel) {
        let mut s = self.0.borrow_mut();
        if let Some(SiteCircuit::Ready { tunnel: current, .. }) = s.sites.get(site) {
            if Arc::ptr_eq(current, tunnel) {
                s.sites.remove(site);
            }
        }
    }

    /// Open a stream to `host:port` on the circuit for `site` (the first party it is fetched
    /// for). A circuit that cannot be built or whose exit refuses or ignores the stream is
    /// replaced, up to four circuits in all.
    pub async fn connect(&self, host_name: &str, port: u16, site: &str) -> Result<TorStream, String> {
        let client = self.client().await?;
        let mut last = String::new();
        for _ in 0..ATTEMPTS {
            let tunnel = match self.site_circuit(&client, site).await {
                Ok(tunnel) => tunnel,
                Err(e) => {
                    last = e;
                    if !client.has_channel().await {
                        self.reset();
                        break;
                    }
                    continue;
                }
            };
            match with_timeout(tunnel.begin_stream(host_name, port, None), STREAM_OPEN_MS).await {
                Some(Ok(inner)) => return Ok(TorStream { inner }),
                Some(Err(e)) => {
                    last = e.to_string();
                    self.retire(site, &tunnel);
                }
                None => {
                    last = format!("the exit did not answer within {} seconds", STREAM_OPEN_MS / 1000);
                    self.retire(site, &tunnel);
                }
            }
        }
        Err(format!("Tor could not reach {host_name}:{port}: {last}"))
    }
}

async fn start() -> Result<Rc<TorClient>, String> {
    let started = now();
    let mut last = String::new();
    for (i, bridge) in BRIDGES.iter().enumerate() {
        status(
            "connecting",
            if i == 0 { "Connecting to the Tor network through the Snowflake bridge…".to_string() } else { "Trying Snowflake's second bridge…".to_string() },
        );
        let mut options = TorClientOptions::snowflake_with_url(bridge.to_string());
        // The channel only; the engine builds and owns its circuits (see TorNet).
        options.create_circuit_early = false;
        options.connection_timeout = 30_000;
        options.circuit_timeout = 60_000;
        options.circuit_update_interval = None;
        let client = match TorClient::new(options).await {
            Ok(client) => client,
            Err(e) => {
                last = e.to_string();
                continue;
            }
        };
        if let Err(e) = client.bootstrap_channel().await {
            client.abort();
            last = e.to_string();
            continue;
        }
        if webtor::directory::cached_consensus_is_stale() {
            status("connecting", "Refreshing the Tor relay list through the bridge…");
            if let Err(e) = client.bootstrap().await {
                last = format!("the relay list could not be refreshed: {e}");
                client.abort();
                continue;
            }
        }
        let secs = (now() - started) / 1000.0;
        status("ready", format!("Connected to Tor in {secs:.1} s"));
        return Ok(Rc::new(client));
    }
    let message = format!("Could not connect to the Tor network: {last}");
    status("failed", message.clone());
    Err(message)
}
