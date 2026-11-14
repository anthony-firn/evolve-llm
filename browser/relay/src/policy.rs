//! What the relay is willing to connect to, and which pages may use it.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// Addresses that must never be reachable through the relay unless `--allow-private`:
/// anything that is not a public unicast destination.
pub fn is_forbidden(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => forbidden_v4(v4),
        IpAddr::V6(v6) => forbidden_v6(v6),
    }
}

fn in_v4(ip: Ipv4Addr, net: [u8; 4], prefix: u32) -> bool {
    let mask = if prefix == 0 { 0 } else { u32::MAX << (32 - prefix) };
    (u32::from(ip) & mask) == (u32::from(Ipv4Addr::from(net)) & mask)
}

fn forbidden_v4(ip: Ipv4Addr) -> bool {
    const NETS: &[([u8; 4], u32)] = &[
        ([0, 0, 0, 0], 8),        // "this network", includes unspecified
        ([10, 0, 0, 0], 8),       // private
        ([100, 64, 0, 0], 10),    // carrier-grade NAT
        ([127, 0, 0, 0], 8),      // loopback
        ([169, 254, 0, 0], 16),   // link-local
        ([172, 16, 0, 0], 12),    // private
        ([192, 0, 0, 0], 24),     // IETF protocol assignments
        ([192, 0, 2, 0], 24),     // documentation (TEST-NET-1)
        ([192, 88, 99, 0], 24),   // deprecated 6to4 relay anycast
        ([192, 168, 0, 0], 16),   // private
        ([198, 18, 0, 0], 15),    // benchmarking
        ([198, 51, 100, 0], 24),  // documentation (TEST-NET-2)
        ([203, 0, 113, 0], 24),   // documentation (TEST-NET-3)
        ([224, 0, 0, 0], 4),      // multicast
        ([240, 0, 0, 0], 4),      // reserved, includes broadcast
    ];
    NETS.iter().any(|(net, prefix)| in_v4(ip, *net, *prefix))
}

fn forbidden_v6(ip: Ipv6Addr) -> bool {
    let s = ip.segments();
    if ip.is_unspecified() || ip.is_loopback() {
        return true;
    }
    // IPv4-mapped ::ffff:a.b.c.d and the deprecated IPv4-compatible ::a.b.c.d
    if s[..5] == [0, 0, 0, 0, 0] && (s[5] == 0xffff || s[5] == 0) {
        return forbidden_v4(Ipv4Addr::new((s[6] >> 8) as u8, s[6] as u8, (s[7] >> 8) as u8, s[7] as u8));
    }
    // NAT64 64:ff9b::/96 embeds an IPv4 address in the last 32 bits
    if s[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
        return forbidden_v4(Ipv4Addr::new((s[6] >> 8) as u8, s[6] as u8, (s[7] >> 8) as u8, s[7] as u8));
    }
    // 6to4 2002::/16 embeds an IPv4 address in bits 16..48
    if s[0] == 0x2002 {
        return forbidden_v4(Ipv4Addr::new((s[1] >> 8) as u8, s[1] as u8, (s[2] >> 8) as u8, s[2] as u8));
    }
    (s[0] & 0xfe00) == 0xfc00          // unique local fc00::/7
        || (s[0] & 0xffc0) == 0xfe80   // link-local fe80::/10
        || (s[0] & 0xffc0) == 0xfec0   // deprecated site-local fec0::/10
        || (s[0] & 0xff00) == 0xff00   // multicast ff00::/8
        || (s[0] == 0x2001 && s[1] == 0x0db8) // documentation 2001:db8::/32
        || (s[0] == 0x2001 && s[1] < 0x200 && s[1] != 0) // IETF protocol assignments 2001::/23 except Teredo below
        || (s[0] == 0x2001 && s[1] == 0) // Teredo 2001::/32 (tunnels to arbitrary IPv4 hosts)
        || s[..4] == [0x100, 0, 0, 0] // discard-only 100::/64
}

pub const DEFAULT_PORTS: &[u16] = &[80, 443];

pub const DEFAULT_ORIGINS: &[&str] = &["https://www.evolvetech.dev", "http://localhost:*", "http://127.0.0.1:*"];

/// Whether a page `origin` (the WebSocket `Origin` header) matches an allowlist entry.
/// Entries are exact origins (`https://example.com`), or `scheme://host:*`, which matches the
/// host with any port or none.
pub fn origin_allowed(origin: &str, allowed: &[String]) -> bool {
    let origin = origin.trim();
    if origin.is_empty() || origin == "null" {
        return false;
    }
    allowed.iter().any(|entry| {
        let entry = entry.trim().trim_end_matches('/');
        if let Some(base) = entry.strip_suffix(":*") {
            if origin.eq_ignore_ascii_case(base) {
                return true;
            }
            match origin.get(..base.len()) {
                Some(prefix) if prefix.eq_ignore_ascii_case(base) => {
                    let rest = &origin[base.len()..];
                    rest.len() > 1 && rest.starts_with(':') && rest[1..].bytes().all(|b| b.is_ascii_digit())
                }
                _ => false,
            }
        } else {
            origin.eq_ignore_ascii_case(entry)
        }
    })
}

/// A host name the relay will try to resolve: a DNS name or an IP literal, nothing else.
pub fn valid_host(host: &str) -> bool {
    let host = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')).unwrap_or(host);
    if host.parse::<IpAddr>().is_ok() {
        return true;
    }
    !host.is_empty()
        && host.len() <= 253
        && host.split('.').all(|label| {
            !label.is_empty() && label.len() <= 63 && label.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn blocks_non_public_ipv4() {
        for s in [
            "0.0.0.0", "10.1.2.3", "100.64.0.1", "100.127.255.254", "127.0.0.1", "127.255.0.1", "169.254.169.254",
            "172.16.0.1", "172.31.255.255", "192.0.0.8", "192.0.2.1", "192.168.1.1", "198.18.0.1", "198.19.255.1",
            "198.51.100.7", "203.0.113.9", "224.0.0.1", "239.255.255.250", "240.0.0.1", "255.255.255.255",
        ] {
            assert!(is_forbidden(ip(s)), "{s} should be blocked");
        }
    }

    #[test]
    fn allows_public_ipv4() {
        for s in ["1.1.1.1", "8.8.8.8", "93.184.215.14", "100.63.255.255", "100.128.0.1", "172.15.255.255", "172.32.0.1", "192.169.0.1", "198.20.0.1", "223.255.255.255"] {
            assert!(!is_forbidden(ip(s)), "{s} should be allowed");
        }
    }

    #[test]
    fn blocks_non_public_ipv6_and_embedded_ipv4() {
        for s in [
            "::", "::1", "fc00::1", "fd12:3456::1", "fe80::1", "febf::1", "fec0::1", "ff02::1", "2001:db8::1", "2001::1",
            "::ffff:127.0.0.1", "::ffff:10.0.0.1", "::ffff:192.168.0.1", "::ffff:169.254.169.254", "::127.0.0.1",
            "64:ff9b::7f00:1", "64:ff9b::a00:1", "2002:c0a8:101::1", "2002:7f00:1::", "100::1",
        ] {
            assert!(is_forbidden(ip(s)), "{s} should be blocked");
        }
    }

    #[test]
    fn allows_public_ipv6() {
        for s in ["2606:4700:4700::1111", "2a00:1450:4001:80b::200e", "::ffff:8.8.8.8", "64:ff9b::808:808", "2002:808:808::1"] {
            assert!(!is_forbidden(ip(s)), "{s} should be allowed");
        }
    }

    #[test]
    fn matches_origins() {
        let allowed: Vec<String> = DEFAULT_ORIGINS.iter().map(|s| s.to_string()).collect();
        assert!(origin_allowed("https://www.evolvetech.dev", &allowed));
        assert!(origin_allowed("http://localhost:8080", &allowed));
        assert!(origin_allowed("http://localhost", &allowed));
        assert!(origin_allowed("http://127.0.0.1:5173", &allowed));
        assert!(!origin_allowed("http://localhost.evil.test", &allowed));
        assert!(!origin_allowed("http://localhost:80x", &allowed));
        assert!(!origin_allowed("http://localhost:", &allowed));
        assert!(!origin_allowed("https://localhost:8080", &allowed));
        assert!(!origin_allowed("https://evolvetech.dev", &allowed));
        assert!(!origin_allowed("https://www.evolvetech.dev.evil.test", &allowed));
        assert!(!origin_allowed("null", &allowed));
        assert!(!origin_allowed("", &allowed));
        let extra = vec!["https://example.com/".to_string()];
        assert!(origin_allowed("https://example.com", &extra));
        assert!(!origin_allowed("https://example.com:8443", &extra));
    }

    #[test]
    fn validates_hosts() {
        assert!(valid_host("example.com"));
        assert!(valid_host("a-b_c.example.co.uk"));
        assert!(valid_host("93.184.215.14"));
        assert!(valid_host("::1"));
        assert!(valid_host("[2606:4700::1111]"));
        assert!(!valid_host(""));
        assert!(!valid_host("exa mple.com"));
        assert!(!valid_host("example..com"));
        assert!(!valid_host("example.com:443"));
        assert!(!valid_host(&"a".repeat(64)));
    }
}
