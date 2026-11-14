//! Per-client limits for a public relay: connections, streams and an hourly byte budget.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Usage {
    pub connections: AtomicUsize,
    pub streams: AtomicUsize,
    pub bytes: AtomicU64,
    window: Mutex<Option<Instant>>,
}

impl Usage {
    /// Add relayed bytes to the current hour's total.
    pub fn add_bytes(&self, n: u64) {
        self.roll();
        self.bytes.fetch_add(n, Ordering::Relaxed);
    }

    fn roll(&self) {
        let mut w = self.window.lock().unwrap();
        let now = Instant::now();
        match *w {
            Some(start) if now.duration_since(start) < Duration::from_secs(3600) => {}
            _ => {
                *w = Some(now);
                self.bytes.store(0, Ordering::Relaxed);
            }
        }
    }

    pub fn over_budget(&self, budget: u64) -> bool {
        self.roll();
        budget > 0 && self.bytes.load(Ordering::Relaxed) >= budget
    }
}

#[derive(Default)]
pub struct Clients {
    map: Mutex<HashMap<IpAddr, Arc<Usage>>>,
}

impl Clients {
    /// Register a connection for `ip`. `None` when it already has `max` connections.
    pub fn connect(&self, ip: IpAddr, max: usize) -> Option<Arc<Usage>> {
        let mut map = self.map.lock().unwrap();
        let usage = map.entry(ip).or_default().clone();
        let n = usage.connections.fetch_add(1, Ordering::SeqCst);
        if max > 0 && n >= max {
            usage.connections.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(usage)
    }

    pub fn disconnect(&self, ip: IpAddr, usage: &Usage) {
        usage.connections.fetch_sub(1, Ordering::SeqCst);
        let mut map = self.map.lock().unwrap();
        // Forget idle clients whose budget window has nothing left to enforce.
        if usage.connections.load(Ordering::SeqCst) == 0 && usage.streams.load(Ordering::SeqCst) == 0 && usage.bytes.load(Ordering::Relaxed) == 0 {
            map.remove(&ip);
        }
    }

    pub fn len(&self) -> usize {
        self.map.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The client's address: the CDN's forwarded address when trusted, else the TCP peer.
pub fn client_ip(headers: &[(&str, &str)], peer: IpAddr, trust_forwarded: bool) -> IpAddr {
    if !trust_forwarded {
        return peer;
    }
    let get = |name: &str| headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| *v);
    for value in [get("cdn-clientip"), get("x-real-ip"), get("x-forwarded-for").and_then(|v| v.split(',').next())].into_iter().flatten() {
        if let Ok(ip) = value.trim().parse() {
            return ip;
        }
    }
    peer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_connections_and_bytes_per_client() {
        let clients = Clients::default();
        let ip: IpAddr = "203.0.113.5".parse().unwrap();
        let a = clients.connect(ip, 2).unwrap();
        let _b = clients.connect(ip, 2).unwrap();
        assert!(clients.connect(ip, 2).is_none());
        assert!(clients.connect("203.0.113.6".parse().unwrap(), 2).is_some());
        clients.disconnect(ip, &a);
        assert!(clients.connect(ip, 2).is_some());
        a.add_bytes(100);
        assert!(!a.over_budget(101));
        a.add_bytes(1);
        assert!(a.over_budget(101));
        assert!(!a.over_budget(0));
    }

    #[test]
    fn forwarded_addresses_only_when_trusted() {
        let peer: IpAddr = "10.1.2.3".parse().unwrap();
        let h = [("X-Forwarded-For", "198.51.100.7, 10.0.0.1")];
        assert_eq!(client_ip(&h, peer, false), peer);
        assert_eq!(client_ip(&h, peer, true), "198.51.100.7".parse::<IpAddr>().unwrap());
        let h = [("CDN-ClientIP", "2001:db8::1"), ("X-Forwarded-For", "198.51.100.7")];
        assert_eq!(client_ip(&h, peer, true), "2001:db8::1".parse::<IpAddr>().unwrap());
        assert_eq!(client_ip(&[("X-Forwarded-For", "junk")], peer, true), peer);
    }
}
