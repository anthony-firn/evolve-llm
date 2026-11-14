use evolve_relay::{serve, Config};
use std::sync::Arc;

const HELP: &str = "evolve-relay: Wisp v1 relay for Evolve Browser

USAGE:
    evolve-relay [OPTIONS]

OPTIONS:
    --listen ADDR         Address to listen on (default 127.0.0.1:8787)
    --allow-origin URL    Also allow WebSocket connections from this page origin. Repeatable.
                          Use scheme://host:* to allow any port. Defaults:
                          https://www.evolvetech.dev, http://localhost:*, http://127.0.0.1:*
    --allow-private       Allow loopback, private, link-local and other non-public destinations
    --allow-port N        Allow destination port N in addition to 80 and 443. Repeatable.
    --public              Settings for a relay reachable from the internet behind a CDN: client
                          addresses from CDN-ClientIP / X-Forwarded-For, 1024 connections,
                          8 connections, 128 streams and 2 GiB per client address per hour,
                          15-minute idle timeout, and https://evolvetech.dev allowed
    --verbose             Log connections and destinations to stderr
    -h, --help            Show this help

The browser engine negotiates TLS itself, so this relay only forwards encrypted bytes.
";

fn parse(args: impl IntoIterator<Item = String>) -> Result<(String, Config), String> {
    let mut listen = "127.0.0.1:8787".to_string();
    let mut config = Config::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |name: &str| inline.clone().or_else(|| args.next()).ok_or_else(|| format!("{name} needs a value"));
        match flag.as_str() {
            "-h" | "--help" => return Err(String::new()),
            "--listen" => listen = value("--listen")?,
            "--allow-origin" => {
                let origin = value("--allow-origin")?;
                if !(origin.starts_with("http://") || origin.starts_with("https://")) {
                    return Err(format!("--allow-origin expects an origin like https://example.com, got {origin:?}"));
                }
                config.allow_origins.push(origin.trim_end_matches('/').to_string());
            }
            "--allow-private" => config.allow_private = true,
            "--allow-port" => {
                let port = value("--allow-port")?;
                config.allow_ports.push(port.parse().map_err(|_| format!("--allow-port expects a port number, got {port:?}"))?);
            }
            "--verbose" | "-v" => config.verbose = true,
            "--public" => config = config.public(),
            other => return Err(format!("unknown option {other:?}")),
        }
    }
    Ok((listen, config))
}

#[tokio::main]
async fn main() {
    let (listen, config) = match parse(std::env::args().skip(1)) {
        Ok(v) => v,
        Err(e) if e.is_empty() => {
            print!("{HELP}");
            return;
        }
        Err(e) => {
            eprintln!("evolve-relay: {e}\n\n{HELP}");
            std::process::exit(2);
        }
    };
    let listener = match tokio::net::TcpListener::bind(&listen).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("evolve-relay: cannot listen on {listen}: {e}");
            std::process::exit(1);
        }
    };
    let addr = listener.local_addr().map(|a| a.to_string()).unwrap_or(listen);
    eprintln!("evolve-relay listening on ws://{addr}/");
    eprintln!("  allowed page origins: {}", config.allow_origins.join(", "));
    let mut ports = vec!["80".to_string(), "443".to_string()];
    ports.extend(config.allow_ports.iter().map(|p| p.to_string()));
    eprintln!("  destination ports: {}", ports.join(", "));
    if config.allow_private {
        eprintln!("  warning: --allow-private lets pages reach this machine and its local network");
    }
    if config.trust_forwarded {
        eprintln!("  public mode: {} connections, per client {} connections / {} streams / {} MiB per hour", config.max_connections, config.max_per_client, config.max_streams_per_client, config.client_bytes_per_hour >> 20);
    }
    if let Err(e) = serve(listener, Arc::new(config)).await {
        eprintln!("evolve-relay: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn parses_flags() {
        let (listen, c) = parse(args("--listen 0.0.0.0:9000 --allow-origin https://example.com/ --allow-port 8443 --allow-private --verbose")).unwrap();
        assert_eq!(listen, "0.0.0.0:9000");
        assert!(c.allow_origins.contains(&"https://example.com".to_string()));
        assert!(c.allow_origins.contains(&"https://www.evolvetech.dev".to_string()));
        assert!(c.allow_private && c.verbose);
        assert!(c.port_allowed(8443) && c.port_allowed(443) && !c.port_allowed(22));
        let (listen, _) = parse(args("--listen=127.0.0.1:1")).unwrap();
        assert_eq!(listen, "127.0.0.1:1");
        assert_eq!(parse(args("--help")).unwrap_err(), "");
        assert!(parse(args("--bogus")).is_err());
        assert!(parse(args("--allow-port")).is_err());
        assert!(parse(args("--allow-port x")).is_err());
        assert!(parse(args("--allow-origin example.com")).is_err());
    }
}
