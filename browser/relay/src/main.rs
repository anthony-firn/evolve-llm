use evolve_relay::{announce, serve, Config};
use std::path::PathBuf;
use std::sync::Arc;

const HELP: &str = "evolve-relay: Wisp v1 relay for Evolve Browser

USAGE:
    evolve-relay [OPTIONS]
    evolve-relay keygen PATH    Create a Nostr key file for --announce (prints the public key)

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

ANNOUNCING (join the community relay network; needs --public):
    --announce KEY_FILE   Publish a signed Nostr announcement of this relay on startup and every
                          30 minutes (each expires after 2 hours)
    --public-url URL      The wss:// address browsers use to reach this relay (required)
    --nostr-relay URL     Nostr relay to publish to. Repeatable. Defaults: wss://relay.damus.io,
                          wss://nos.lol, wss://relay.primal.net, wss://offchain.pub
    --name TEXT           Short name shown to people choosing a relay
    --region TEXT         Where the relay runs, e.g. us-west or eu-central
    --contact TEXT        How to reach the operator (email, npub or URL)

The browser engine negotiates TLS itself, so this relay only forwards encrypted bytes.
";

struct Options {
    listen: String,
    config: Config,
    /// Key file, announcement and Nostr relays when --announce is given.
    announce: Option<(PathBuf, announce::Announcement, Vec<String>)>,
}

#[allow(clippy::large_enum_variant)] // parsed once at startup
enum Command {
    Run(Options),
    Keygen(PathBuf),
}

fn parse(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut args = args.into_iter().peekable();
    if args.peek().map(String::as_str) == Some("keygen") {
        args.next();
        let path = args.next().ok_or("keygen needs a file path, e.g. evolve-relay keygen relay.key")?;
        if let Some(extra) = args.next() {
            return Err(format!("unexpected argument {extra:?} after keygen PATH"));
        }
        return Ok(Command::Keygen(path.into()));
    }
    let mut listen = "127.0.0.1:8787".to_string();
    let mut config = Config::default();
    let (mut key_file, mut public_url, mut nostr_relays) = (None::<String>, None::<String>, Vec::<String>::new());
    let mut ann = announce::Announcement::default();
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
            "--announce" => key_file = Some(value("--announce")?),
            "--public-url" => public_url = Some(value("--public-url")?),
            "--nostr-relay" => {
                let url = value("--nostr-relay")?;
                if !(url.starts_with("wss://") || url.starts_with("ws://")) {
                    return Err(format!("--nostr-relay expects a wss:// address, got {url:?}"));
                }
                nostr_relays.push(url);
            }
            "--name" => ann.name = Some(value("--name")?),
            "--region" => ann.region = Some(value("--region")?),
            "--contact" => ann.contact = Some(value("--contact")?),
            other => return Err(format!("unknown option {other:?}")),
        }
    }
    let announce = match key_file {
        None => {
            if public_url.is_some() || !nostr_relays.is_empty() || ann.name.is_some() || ann.region.is_some() || ann.contact.is_some() {
                return Err("--public-url, --nostr-relay, --name, --region and --contact only apply with --announce".into());
            }
            None
        }
        Some(key_file) => {
            if !config.trust_forwarded {
                return Err("--announce needs --public: only a relay with per-client limits may be offered to strangers".into());
            }
            if config.allow_private {
                return Err("--announce cannot be combined with --allow-private".into());
            }
            let url = public_url.ok_or("--announce needs --public-url wss://your-relay.example/")?;
            announce::check_public_url(&url)?;
            let mut full = announce::Announcement::from_config(&url, &config);
            (full.name, full.region, full.contact) = (ann.name, ann.region, ann.contact);
            if nostr_relays.is_empty() {
                nostr_relays = announce::DEFAULT_NOSTR_RELAYS.iter().map(|s| s.to_string()).collect();
            }
            Some((PathBuf::from(key_file), full, nostr_relays))
        }
    };
    Ok(Command::Run(Options { listen, config, announce }))
}

#[tokio::main]
async fn main() {
    let Options { listen, config, announce: announcement } = match parse(std::env::args().skip(1)) {
        Ok(Command::Run(options)) => options,
        Ok(Command::Keygen(path)) => match announce::keygen(&path) {
            Ok(public) => {
                eprintln!("evolve-relay: wrote a new secret key to {} (mode 0600); keep it private", path.display());
                println!("{}", announce::npub(&public));
                println!("{}", announce::hex(&public.to_bytes()));
                return;
            }
            Err(e) => {
                eprintln!("evolve-relay: {e}");
                std::process::exit(1);
            }
        },
        Err(e) if e.is_empty() => {
            print!("{HELP}");
            return;
        }
        Err(e) => {
            eprintln!("evolve-relay: {e}\n\n{HELP}");
            std::process::exit(2);
        }
    };
    // Load the key before listening, so a bad key file fails fast.
    let announcement = match announcement.map(|(path, ann, relays)| announce::load_key(&path).map(|key| (key, ann, relays))).transpose() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("evolve-relay: {e}");
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
    if let Some((key, ann, relays)) = announcement {
        eprintln!("  announcing {} as {} on {}", ann.public_url, announce::npub(key.verifying_key()), relays.join(", "));
        // The self-check connects like a browser page would, so it needs an allowed origin.
        let origin = config.allow_origins.iter().find(|o| o.starts_with("https://") && !o.contains('*')).cloned().unwrap_or_else(|| "https://www.evolvetech.dev".into());
        tokio::spawn(announce::run(key, ann, relays, origin));
    }
    if let Err(e) = serve(listener, Arc::new(config)).await {
        eprintln!("evolve-relay: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{parse, Command, Options};

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    fn run(s: &str) -> Result<Options, String> {
        match parse(args(s))? {
            Command::Run(o) => Ok(o),
            Command::Keygen(_) => Err("keygen".into()),
        }
    }

    #[test]
    fn parses_flags() {
        let Options { listen, config: c, announce } = run("--listen 0.0.0.0:9000 --allow-origin https://example.com/ --allow-port 8443 --allow-private --verbose").unwrap();
        assert_eq!(listen, "0.0.0.0:9000");
        assert!(announce.is_none());
        assert!(c.allow_origins.contains(&"https://example.com".to_string()));
        assert!(c.allow_origins.contains(&"https://www.evolvetech.dev".to_string()));
        assert!(c.allow_private && c.verbose);
        assert!(c.port_allowed(8443) && c.port_allowed(443) && !c.port_allowed(22));
        assert_eq!(run("--listen=127.0.0.1:1").unwrap().listen, "127.0.0.1:1");
        assert_eq!(parse(args("--help")).err().unwrap(), "");
        assert!(parse(args("--bogus")).is_err());
        assert!(parse(args("--allow-port")).is_err());
        assert!(parse(args("--allow-port x")).is_err());
        assert!(parse(args("--allow-origin example.com")).is_err());
    }

    #[test]
    fn parses_announce_flags() {
        let o = run("--public --announce relay.key --public-url wss://relay.example.com/ --allow-port 8443 --name Example --region us-west --contact ops@example.com").unwrap();
        let (path, ann, relays) = o.announce.unwrap();
        assert_eq!(path.to_str(), Some("relay.key"));
        assert_eq!(ann.public_url, "wss://relay.example.com/");
        assert_eq!((ann.name.as_deref(), ann.region.as_deref(), ann.contact.as_deref()), (Some("Example"), Some("us-west"), Some("ops@example.com")));
        assert_eq!(ann.limits, (8, 128, 2 << 30));
        assert_eq!(ann.ports, vec![80, 443, 8443]);
        assert_eq!(relays.len(), 4);
        let o = run("--public --announce k --public-url wss://r.example/ --nostr-relay wss://nostr.example").unwrap();
        assert_eq!(o.announce.unwrap().2, vec!["wss://nostr.example".to_string()]);
        // A relay without per-client limits must never be announced.
        assert!(run("--announce k --public-url wss://r.example/").err().unwrap().contains("--public"));
        assert!(run("--public --announce k --public-url wss://r.example/ --allow-private").is_err());
        assert!(run("--public --announce k").err().unwrap().contains("--public-url"));
        assert!(run("--public --announce k --public-url ws://r.example/").is_err());
        assert!(run("--public --name x").is_err());
        assert!(run("--public --announce k --public-url wss://r.example/ --nostr-relay https://x").is_err());
        assert!(matches!(parse(args("keygen relay.key")), Ok(Command::Keygen(p)) if p.to_str() == Some("relay.key")));
        assert!(parse(args("keygen")).is_err());
        assert!(parse(args("keygen a b")).is_err());
    }
}
