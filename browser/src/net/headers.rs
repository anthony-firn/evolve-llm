//! Request header fields and HTTP/2 request framing for each browser identity.
//!
//! Firefox identities follow the field set and order Firefox ESR sends (see
//! docs/browser-protocol-reference.md and tests/fixtures/firefox-esr-153.7-protocol.json).
//! Other identities keep the engine's original fields and order.

use super::{policy, FetchRequest, Mode};
use crate::identity::Identity;

/// Builds the request's header fields, excluding Host (HTTP/1.1 adds it; HTTP/2 uses :authority).
pub fn request_headers(
    req: &FetchRequest,
    identity: Identity,
    user_agent: &str,
    accept_language: &str,
    cookie: Option<String>,
    cors_origin: Option<&str>,
) -> Vec<(String, String)> {
    if identity.is_firefox() {
        firefox(req, user_agent, accept_language, cookie, cors_origin)
    } else {
        evolve(req, user_agent, accept_language, cookie, cors_origin)
    }
}

/// HTTP/2 framing for a request: Firefox's pseudo-header order, HEADERS priority weight
/// and 12 MiB stream receive window, or h2's defaults for other identities.
pub fn h2_framing(identity: Identity, dest: &str) -> h2::ext::RequestFraming {
    if !identity.is_firefox() {
        return h2::ext::RequestFraming::default();
    }
    h2::ext::RequestFraming {
        path_before_authority: true,
        priority: Some((false, 0, firefox_priority(dest).1)),
        stream_window_increment: Some(FIREFOX_STREAM_WINDOW - FIREFOX_INITIAL_WINDOW),
    }
}

/// SETTINGS_INITIAL_WINDOW_SIZE Firefox advertises
pub const FIREFOX_INITIAL_WINDOW: u32 = 131072;
/// Firefox raises each stream's window from the advertised 131072 to 12 MiB.
pub const FIREFOX_STREAM_WINDOW: u32 = 12 * 1024 * 1024;
/// Firefox raises the connection window from 65535 to 12 MiB (increment 12517377).
pub const FIREFOX_CONNECTION_WINDOW: u32 = 12 * 1024 * 1024;

/// The RFC 9218 `priority` field and HTTP/2 weight Firefox uses for a destination.
/// Weights are 22 minus Firefox's internal priority: highest 42, high 32, normal 22, low 12.
fn firefox_priority(dest: &str) -> (Option<&'static str>, u16) {
    match dest {
        "document" => (Some("u=0, i"), 42),
        "style" => (Some("u=2"), 32),
        "font" => (None, 32),
        "script" => (None, 22),
        "image" => (Some("u=5, i"), 12),
        // fetch(), XMLHttpRequest and frames
        _ => (Some("u=4"), 22),
    }
}

fn accept(dest: &str, firefox: bool) -> &'static str {
    match (dest, firefox) {
        ("document" | "iframe" | "frame", true) => "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        ("document" | "iframe" | "frame", false) => "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        ("style", _) => "text/css,*/*;q=0.1",
        ("image", true) => "image/avif,image/webp,image/png,image/svg+xml,image/*;q=0.8,*/*;q=0.5",
        ("image", false) => "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        _ => "*/*",
    }
}

const ACCEPT_ENCODING: &str = "gzip, deflate, br, zstd";

fn referer(req: &FetchRequest) -> Option<String> {
    // strict-origin-when-cross-origin: the full URL within an origin, else the origin only
    let initiator = req.initiator.as_ref()?;
    let downgrade = initiator.scheme() == "https" && req.url.scheme() == "http";
    if downgrade || !matches!(initiator.scheme(), "http" | "https") {
        return None;
    }
    Some(if policy::same_origin(initiator, &req.url) {
        let mut r = initiator.clone();
        r.set_fragment(None);
        let _ = r.set_username("");
        let _ = r.set_password(None);
        r.to_string()
    } else {
        format!("{}/", initiator.origin().ascii_serialization())
    })
}

fn origin(req: &FetchRequest, cors_origin: Option<&str>) -> Option<String> {
    cors_origin.map(str::to_string).or_else(|| {
        (!matches!(req.method.as_str(), "GET" | "HEAD"))
            .then(|| req.initiator.as_ref().map(|u| u.origin().ascii_serialization()).unwrap_or_else(|| "null".into()))
    })
}

fn fetch_site(req: &FetchRequest) -> &'static str {
    match &req.initiator {
        None => "none",
        Some(i) if policy::same_origin(i, &req.url) => "same-origin",
        Some(i) if policy::same_site(i, &req.url) => "same-site",
        Some(_) => "cross-site",
    }
}

fn fetch_mode(req: &FetchRequest) -> &'static str {
    match req.mode {
        Mode::Navigate => "navigate",
        Mode::NoCors => "no-cors",
        Mode::Cors => "cors",
    }
}

/// Fields a page script may set (forbidden names are dropped; preflight fields only for CORS).
fn script_fields(req: &FetchRequest) -> impl Iterator<Item = &(String, String)> {
    req.headers.iter().filter(|(k, _)| !policy::forbidden_request_header(k) || req.mode != Mode::Cors && k.starts_with("Access-Control-Request"))
}

fn evolve(req: &FetchRequest, user_agent: &str, accept_language: &str, cookie: Option<String>, cors_origin: Option<&str>) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = Vec::new();
    let has = |name: &str| req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(name));
    headers.push(("User-Agent".into(), user_agent.into()));
    if !has("accept") {
        headers.push(("Accept".into(), accept(req.dest, false).into()));
    }
    if !has("accept-language") {
        headers.push(("Accept-Language".into(), accept_language.into()));
    }
    headers.push(("Accept-Encoding".into(), ACCEPT_ENCODING.into()));
    headers.extend(script_fields(req).cloned());
    if let Some(cookie) = cookie {
        headers.push(("Cookie".into(), cookie));
    }
    if let Some(referer) = referer(req) {
        headers.push(("Referer".into(), referer));
    }
    if let Some(origin) = origin(req, cors_origin) {
        headers.push(("Origin".into(), origin));
    }
    headers.push(("Sec-Fetch-Site".into(), fetch_site(req).into()));
    headers.push(("Sec-Fetch-Mode".into(), fetch_mode(req).into()));
    headers.push(("Sec-Fetch-Dest".into(), req.dest.into()));
    if req.mode == Mode::Navigate {
        if req.user_activated {
            headers.push(("Sec-Fetch-User".into(), "?1".into()));
        }
        headers.push(("Upgrade-Insecure-Requests".into(), "1".into()));
    }
    headers
}

fn firefox(req: &FetchRequest, user_agent: &str, accept_language: &str, cookie: Option<String>, cors_origin: Option<&str>) -> Vec<(String, String)> {
    let script_value = |name: &str| req.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.clone());
    let mut headers: Vec<(String, String)> = Vec::new();
    headers.push(("User-Agent".into(), user_agent.into()));
    headers.push(("Accept".into(), script_value("accept").unwrap_or_else(|| accept(req.dest, true).into())));
    headers.push(("Accept-Language".into(), script_value("accept-language").unwrap_or_else(|| accept_language.into())));
    headers.push(("Accept-Encoding".into(), ACCEPT_ENCODING.into()));
    if let Some(referer) = referer(req) {
        headers.push(("Referer".into(), referer));
    }
    headers.extend(script_fields(req).filter(|(k, _)| !k.eq_ignore_ascii_case("accept") && !k.eq_ignore_ascii_case("accept-language")).cloned());
    if !req.body.is_empty() || matches!(req.method.as_str(), "POST" | "PUT" | "PATCH") {
        headers.push(("Content-Length".into(), req.body.len().to_string()));
    }
    if let Some(origin) = origin(req, cors_origin) {
        headers.push(("Origin".into(), origin));
    }
    if let Some(cookie) = cookie {
        headers.push(("Cookie".into(), cookie));
    }
    if req.mode == Mode::Navigate {
        headers.push(("Upgrade-Insecure-Requests".into(), "1".into()));
    }
    headers.push(("Sec-Fetch-Dest".into(), req.dest.into()));
    headers.push(("Sec-Fetch-Mode".into(), fetch_mode(req).into()));
    headers.push(("Sec-Fetch-Site".into(), fetch_site(req).into()));
    if req.mode == Mode::Navigate && req.dest == "document" && req.user_activated {
        headers.push(("Sec-Fetch-User".into(), "?1".into()));
    }
    if let Some(priority) = firefox_priority(req.dest).0 {
        headers.push(("Priority".into(), priority.into()));
    }
    headers.push(("TE".into(), "trailers".into()));
    headers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::{Credentials, Redirect};
    use url::Url;

    const UA: &str = "Mozilla/5.0 (X11; Linux x86_64; rv:153.0) Gecko/20100101 Firefox/153.0";

    fn request(method: &str, url: &str, initiator: Option<&str>, mode: Mode, dest: &'static str) -> FetchRequest {
        FetchRequest {
            url: Url::parse(url).unwrap(),
            method: method.into(),
            headers: vec![],
            body: vec![],
            initiator: initiator.map(|u| Url::parse(u).unwrap()),
            mode,
            credentials: Credentials::Include,
            redirect: Redirect::Follow,
            dest,
            user_activated: true,
        }
    }

    fn names(headers: &[(String, String)]) -> Vec<String> {
        headers.iter().map(|(k, _)| k.to_ascii_lowercase()).collect()
    }

    /// Each recorded Firefox request shape, reproduced from the same inputs, gives the same
    /// field order (one cookie field here; HTTP/2 splits it into crumbs later).
    #[test]
    fn firefox_field_order_matches_reference() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/firefox-esr-153.7-protocol.json")).unwrap();
        let page = "https://accounts.google.com/v3/signin/identifier";
        let mut checked = 0;
        for r in fixture["requests"].as_array().unwrap() {
            let host = r["host"].as_str().unwrap();
            let path = r["path"].as_str().unwrap();
            let method = r["method"].as_str().unwrap();
            let public = &r["public_headers"];
            let dest: &'static str = match public["sec-fetch-dest"].as_str().unwrap() {
                "document" => "document",
                "iframe" => "iframe",
                "image" => "image",
                "empty" => "empty",
                other => panic!("unexpected destination {other}"),
            };
            let mode = match public["sec-fetch-mode"].as_str().unwrap() {
                "navigate" => Mode::Navigate,
                "cors" => Mode::Cors,
                _ => Mode::NoCors,
            };
            let initiator = match public["sec-fetch-site"].as_str().unwrap() {
                "none" => None,
                "cross-site" => Some("https://accounts.google.com/v3/signin/identifier"),
                "same-site" => Some("https://accounts.google.com/v3/signin/identifier"),
                _ => Some(page),
            };
            let order: Vec<String> = r["header_order"].as_array().unwrap().iter()
                .map(|v| v.as_str().unwrap().to_string()).filter(|n| !n.starts_with(':')).collect();
            // Fields our engine does not produce yet are left out of the comparison.
            let order: Vec<String> = order.into_iter().filter(|n| n != "sec-fetch-storage-access").fold(Vec::new(), |mut v, n| {
                if n != "cookie" || v.last() != Some(&n) { v.push(n) }
                v
            });
            let mut req = request(method, &format!("https://{host}{path}"), initiator, mode, dest);
            req.user_activated = order.iter().any(|n| n == "sec-fetch-user");
            if method == "POST" {
                req.body = b"f.req=x".to_vec();
                for name in &order {
                    if name.starts_with("x-") || name == "content-type" {
                        req.headers.push((name.clone(), "1".into()));
                    }
                }
            }
            let cookie = order.iter().any(|n| n == "cookie").then(|| "NID=1".to_string());
            let cors = (mode == Mode::Cors).then_some("https://accounts.google.com");
            let headers = request_headers(&req, Identity::FirefoxEsr, UA, "en-US,en;q=0.9", cookie, cors);
            assert_eq!(names(&headers), order, "{method} {path}");
            for (name, value) in &headers {
                let lower = name.to_ascii_lowercase();
                if let Some(expected) = public[&lower].as_str() {
                    assert_eq!(value, expected, "{lower} for {method} {path}");
                }
            }
            checked += 1;
        }
        assert!(checked >= 15, "checked {checked} requests");
    }

    #[test]
    fn firefox_framing_uses_reference_weights() {
        let framing = h2_framing(Identity::FirefoxEsr, "document");
        assert!(framing.path_before_authority);
        assert_eq!(framing.priority, Some((false, 0, 42)));
        assert_eq!(framing.stream_window_increment, Some(12451840));
        assert_eq!(h2_framing(Identity::FirefoxEsr, "image").priority, Some((false, 0, 12)));
        assert_eq!(h2_framing(Identity::FirefoxEsr, "empty").priority, Some((false, 0, 22)));
        assert_eq!(h2_framing(Identity::Legacy, "document"), h2::ext::RequestFraming::default());
    }

    #[test]
    fn other_identities_keep_engine_order() {
        let req = request("GET", "https://example.com/", None, Mode::Navigate, "document");
        let headers = request_headers(&req, Identity::Legacy, "UA", "en", None, None);
        assert_eq!(names(&headers), ["user-agent", "accept", "accept-language", "accept-encoding", "sec-fetch-site", "sec-fetch-mode", "sec-fetch-dest", "sec-fetch-user", "upgrade-insecure-requests"]);
    }
}
