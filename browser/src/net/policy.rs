//! Site and origin rules: registrable domains, SameSite cookie selection and CORS checks.
//! Pure so the security rules are unit tested natively.

use std::sync::OnceLock;
use url::Url;

static PSL: OnceLock<publicsuffix::List> = OnceLock::new();

pub fn suffix_list() -> &'static publicsuffix::List {
    PSL.get_or_init(|| {
        include_str!("../../data/public_suffix_list.dat")
            .parse()
            .expect("bundled public suffix list parses")
    })
}

/// The registrable domain ("site") of a host, e.g. `news.ycombinator.com` → `ycombinator.com`.
pub fn site(url: &Url) -> String {
    let Some(host) = url.host_str() else { return url.origin().ascii_serialization() };
    if url.host().is_some_and(|h| !matches!(h, url::Host::Domain(_))) {
        return host.to_string();
    }
    use publicsuffix::Psl;
    suffix_list()
        .domain(host.as_bytes())
        .and_then(|d| std::str::from_utf8(d.as_bytes()).ok().map(str::to_string))
        .unwrap_or_else(|| host.to_string())
        .to_ascii_lowercase()
}

/// Schemeful same-site, as Chrome applies it to cookies.
pub fn same_site(a: &Url, b: &Url) -> bool {
    a.scheme() == b.scheme() && site(a) == site(b)
}

pub fn same_origin(a: &Url, b: &Url) -> bool {
    a.origin() == b.origin()
}

/// What kind of request is asking for cookies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Context {
    /// A top-level navigation (typed URL, link, form submission)
    Navigation { safe_method: bool },
    /// Images, stylesheets, scripts, fetch() and XHR
    Subresource,
}

/// SameSite selection (Chrome's rules, with Lax as the default for unmarked cookies).
pub fn cookie_allowed(same_site: Option<cookie::SameSite>, initiator: Option<&Url>, target: &Url, ctx: Context) -> bool {
    let Some(initiator) = initiator else { return true }; // typed into the address bar
    if self::same_site(initiator, target) {
        return true;
    }
    match same_site.unwrap_or(cookie::SameSite::Lax) {
        cookie::SameSite::None => true,
        cookie::SameSite::Strict => false,
        cookie::SameSite::Lax => matches!(ctx, Context::Navigation { safe_method: true }),
    }
}

/// CORS check for a response to a script-initiated cross-origin request.
pub fn cors_allows(headers: &[(String, String)], origin: &str, with_credentials: bool) -> Result<(), String> {
    let get = |name: &str| headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.trim());
    match get("access-control-allow-origin") {
        Some("*") if !with_credentials => {}
        Some(v) if v == origin => {
            if with_credentials && get("access-control-allow-credentials") != Some("true") {
                return Err("CORS: the server did not allow credentials".into());
            }
        }
        Some(_) => return Err(format!("CORS: the response is not shared with {origin}")),
        None => return Err(format!("CORS: no Access-Control-Allow-Origin header for {origin}")),
    }
    Ok(())
}

const SAFELISTED_HEADERS: &[&str] = &["accept", "accept-language", "content-language", "content-type", "range"];

/// Whether a cross-origin request needs a preflight.
pub fn needs_preflight(method: &str, headers: &[(String, String)]) -> bool {
    if !matches!(method, "GET" | "HEAD" | "POST") {
        return true;
    }
    headers.iter().any(|(name, value)| {
        let name = name.to_ascii_lowercase();
        if !SAFELISTED_HEADERS.contains(&name.as_str()) {
            return true;
        }
        name == "content-type"
            && !matches!(
                value.split(';').next().unwrap_or("").trim().to_ascii_lowercase().as_str(),
                "application/x-www-form-urlencoded" | "multipart/form-data" | "text/plain"
            )
    })
}

pub fn preflight_allows(headers: &[(String, String)], method: &str, request_headers: &[(String, String)], with_credentials: bool) -> Result<(), String> {
    let get = |name: &str| headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str()).unwrap_or("");
    let list = |name: &str| get(name).split(',').map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty()).collect::<Vec<_>>();
    let methods = list("access-control-allow-methods");
    let wildcard = !with_credentials;
    if !matches!(method, "GET" | "HEAD" | "POST") && !methods.contains(&method.to_ascii_lowercase()) && !(wildcard && methods.iter().any(|m| m == "*")) {
        return Err(format!("CORS: {method} is not allowed by the preflight response"));
    }
    let allowed = list("access-control-allow-headers");
    for (name, value) in request_headers {
        let lower = name.to_ascii_lowercase();
        let safelisted = SAFELISTED_HEADERS.contains(&lower.as_str()) && !needs_preflight("GET", &[(name.clone(), value.clone())]);
        if !safelisted && !allowed.contains(&lower) && !(wildcard && allowed.iter().any(|h| h == "*")) {
            return Err(format!("CORS: header {name} is not allowed by the preflight response"));
        }
    }
    Ok(())
}

/// Headers page scripts may not set (the Fetch standard's forbidden request headers).
pub fn forbidden_request_header(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    matches!(
        n.as_str(),
        "accept-charset" | "accept-encoding" | "access-control-request-headers" | "access-control-request-method"
            | "connection" | "content-length" | "cookie" | "cookie2" | "date" | "dnt" | "expect" | "host" | "keep-alive"
            | "origin" | "referer" | "set-cookie" | "te" | "trailer" | "transfer-encoding" | "upgrade" | "via"
    ) || n.starts_with("proxy-")
        || n.starts_with("sec-")
}

/// Response headers a script may read on a CORS response (plus any exposed by the server).
pub fn exposed_headers(headers: &[(String, String)], cross_origin: bool) -> Vec<(String, String)> {
    let filtered = headers.iter().filter(|(k, _)| !k.eq_ignore_ascii_case("set-cookie") && !k.eq_ignore_ascii_case("set-cookie2"));
    if !cross_origin {
        return filtered.cloned().collect();
    }
    let expose: Vec<String> = headers
        .iter()
        .filter(|(k, _)| k.eq_ignore_ascii_case("access-control-expose-headers"))
        .flat_map(|(_, v)| v.split(',').map(|s| s.trim().to_ascii_lowercase()))
        .collect();
    const SAFE: &[&str] = &["cache-control", "content-language", "content-length", "content-type", "expires", "last-modified", "pragma"];
    filtered
        .filter(|(k, _)| {
            let k = k.to_ascii_lowercase();
            SAFE.contains(&k.as_str()) || expose.contains(&k) || expose.iter().any(|e| e == "*")
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cookie::SameSite;

    fn u(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    #[test]
    fn registrable_domains_follow_the_suffix_list() {
        assert_eq!(site(&u("https://news.ycombinator.com/x")), "ycombinator.com");
        assert_eq!(site(&u("https://a.b.example.co.uk/")), "example.co.uk");
        assert_eq!(site(&u("https://user.github.io/")), "user.github.io");
        assert!(same_site(&u("https://a.example.com"), &u("https://b.example.com")));
        assert!(!same_site(&u("https://a.example.com"), &u("http://a.example.com")));
        assert!(!same_site(&u("https://a.github.io"), &u("https://b.github.io")));
    }

    #[test]
    fn samesite_rules_protect_cross_site_requests() {
        let bank = u("https://bank.example/account");
        let evil = u("https://evil.test/");
        let nav_get = Context::Navigation { safe_method: true };
        let nav_post = Context::Navigation { safe_method: false };
        assert!(cookie_allowed(None, None, &bank, Context::Subresource));
        assert!(cookie_allowed(Some(SameSite::Strict), Some(&bank), &bank, Context::Subresource));
        assert!(!cookie_allowed(None, Some(&evil), &bank, Context::Subresource));
        assert!(cookie_allowed(None, Some(&evil), &bank, nav_get));
        assert!(!cookie_allowed(None, Some(&evil), &bank, nav_post));
        assert!(!cookie_allowed(Some(SameSite::Strict), Some(&evil), &bank, nav_get));
        assert!(cookie_allowed(Some(SameSite::None), Some(&evil), &bank, Context::Subresource));
    }

    #[test]
    fn cors_requires_an_explicit_grant() {
        let h = |v: &[(&str, &str)]| v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect::<Vec<_>>();
        let o = "https://app.example";
        assert!(cors_allows(&h(&[]), o, false).is_err());
        assert!(cors_allows(&h(&[("Access-Control-Allow-Origin", "*")]), o, false).is_ok());
        assert!(cors_allows(&h(&[("Access-Control-Allow-Origin", "*")]), o, true).is_err());
        assert!(cors_allows(&h(&[("Access-Control-Allow-Origin", o)]), o, true).is_err());
        assert!(cors_allows(&h(&[("Access-Control-Allow-Origin", o), ("Access-Control-Allow-Credentials", "true")]), o, true).is_ok());
        assert!(cors_allows(&h(&[("Access-Control-Allow-Origin", "https://other")]), o, false).is_err());
        assert!(!needs_preflight("POST", &h(&[("Content-Type", "application/x-www-form-urlencoded")])));
        assert!(needs_preflight("POST", &h(&[("Content-Type", "application/json")])));
        assert!(needs_preflight("PUT", &[]));
        assert!(needs_preflight("GET", &h(&[("Authorization", "x")])));
        let pre = h(&[("Access-Control-Allow-Methods", "PUT, DELETE"), ("Access-Control-Allow-Headers", "content-type, authorization")]);
        assert!(preflight_allows(&pre, "PUT", &h(&[("Content-Type", "application/json")]), true).is_ok());
        assert!(preflight_allows(&pre, "PATCH", &[], true).is_err());
        assert!(preflight_allows(&pre, "PUT", &h(&[("X-Token", "1")]), true).is_err());
        assert!(forbidden_request_header("Cookie") && forbidden_request_header("sec-fetch-site") && !forbidden_request_header("X-Requested-With"));
        let exposed = exposed_headers(&h(&[("Content-Type", "a"), ("X-Secret", "b"), ("Set-Cookie", "c")]), true);
        assert_eq!(exposed, h(&[("Content-Type", "a")]));
    }
}
