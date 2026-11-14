//! evolve-relay: a Wisp v1 relay for Evolve Browser.
//!
//! The browser engine runs inside a web page, which cannot open TCP sockets. It opens one
//! WebSocket to this relay and multiplexes TCP streams over it with the Wisp protocol. TLS is
//! negotiated end to end inside the engine, so the relay forwards ciphertext only: it never sees
//! page contents, cookies or passwords for HTTPS sites.

pub mod announce;
pub mod codec;
pub mod limits;
pub mod policy;
pub mod server;

pub use server::serve;

#[derive(Debug, Clone)]
pub struct Config {
    /// Page origins allowed to open the relay WebSocket (see `policy::origin_allowed`).
    pub allow_origins: Vec<String>,
    /// Allow loopback, private and other non-public destination addresses.
    pub allow_private: bool,
    /// Destination ports allowed in addition to 80 and 443.
    pub allow_ports: Vec<u16>,
    /// Log connections and destinations to stderr.
    pub verbose: bool,
    /// Identify clients by the CDN's forwarded address (`CDN-ClientIP`, `X-Real-IP`, or the first
    /// `X-Forwarded-For` entry). Only enable behind a proxy that sets these headers.
    pub trust_forwarded: bool,
    /// WebSocket connections served at once.
    pub max_connections: usize,
    /// WebSocket connections per client address (0 = no limit).
    pub max_per_client: usize,
    /// Open streams per client address across its connections (0 = no limit).
    pub max_streams_per_client: usize,
    /// Bytes relayed per client address per hour, both directions (0 = no limit). New streams are
    /// refused once a client is over its budget.
    pub client_bytes_per_hour: u64,
    /// Close a WebSocket that carries no packets for this long.
    pub idle_timeout: std::time::Duration,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            allow_origins: policy::DEFAULT_ORIGINS.iter().map(|s| s.to_string()).collect(),
            allow_private: false,
            allow_ports: vec![],
            verbose: false,
            trust_forwarded: false,
            max_connections: 64,
            max_per_client: 0,
            max_streams_per_client: 0,
            client_bytes_per_hour: 0,
            idle_timeout: std::time::Duration::from_secs(3600),
        }
    }
}

impl Config {
    /// Settings for a relay that anyone on the internet can reach (behind a CDN).
    pub fn public(mut self) -> Self {
        self.trust_forwarded = true;
        self.max_connections = 1024;
        self.max_per_client = 8;
        self.max_streams_per_client = 128;
        self.client_bytes_per_hour = 2 << 30;
        self.idle_timeout = std::time::Duration::from_secs(15 * 60);
        self.allow_origins.push("https://evolvetech.dev".into());
        self
    }
}

impl Config {
    pub fn port_allowed(&self, port: u16) -> bool {
        policy::DEFAULT_PORTS.contains(&port) || self.allow_ports.contains(&port)
    }
}
