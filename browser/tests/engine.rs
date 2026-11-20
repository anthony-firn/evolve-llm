//! End-to-end engine tests against an in-memory site: parsing, CSS, layout, painting,
//! JavaScript, events, forms, timers, fetch, cookies and modules.

use evolve_browser::testing::Harness;

fn h() -> Harness {
    Harness::new()
}

#[test]
fn identities_are_selected_before_navigation_and_survive_later_navigations() {
    use evolve_browser::identity::Identity;
    for profile in ["legacy", "firefox-esr", "firefox", "chrome", "evolve"] {
        let mut t = h();
        t.engine.set_identity(profile).unwrap();
        t.site.add("https://a.test/", "text/html", "<script>window.initialUA = navigator.userAgent</script>");
        t.site.add("https://b.test/", "text/html", "<p>another origin</p>");
        t.open("https://a.test/");
        let expected = serde_json::to_string(Identity::parse(profile).unwrap().user_agent()).unwrap();
        assert_eq!(t.eval("initialUA"), expected);
        assert!(t.engine.set_identity("evolve").is_err());
        if profile.starts_with("firefox") {
            assert_eq!(t.eval("navigator.vendor"), "\"\"");
            assert_eq!(t.eval("'userAgentData' in navigator || 'deviceMemory' in navigator || 'connection' in navigator"), "false");
            assert_eq!(t.eval("navigator.appVersion"), "\"5.0 (X11)\"");
        }
        t.open("https://b.test/");
        assert_eq!(t.eval("navigator.userAgent"), expected);
    }
    assert!(Identity::parse("invalid\r\nInjected: yes").is_err());
}

#[test]
fn renders_a_styled_page() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", "<!doctype html><title>Hello</title><style>body{background:#204080;margin:0} h1{color:white}</style><h1>Hi there</h1>");
    t.open("https://a.test/");
    let (w, hgt) = t.engine.frame_size();
    assert_eq!((w, hgt), (800, 600));
    let frame = t.engine.take_frame().expect("a frame").to_vec();
    // Background color fills the viewport
    let px = &frame[(500 * 800 + 400) * 4..][..4];
    assert_eq!(&px[..3], &[0x20, 0x40, 0x80], "background pixel {px:?}");
    // Some white text pixels near the top
    let white = frame.chunks(4).take(800 * 60).filter(|p| p[0] > 200 && p[1] > 200 && p[2] > 200).count();
    assert!(white > 50, "expected heading text, found {white} white pixels");
    assert!(t.events.iter().any(|e| matches!(e, evolve_browser::host::OutEvent::Title { title } if title == "Hello")));
}

#[test]
fn noscript_fallbacks_are_inert_in_documents_and_html_fragments() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", r#"<!doctype html><head>
      <noscript><style>body { opacity: 0; }</style><link rel=stylesheet href=/fallback.css></noscript>
      </head><body><p>Visible sign-in page</p><div id=host></div>
      <noscript><script>window.fallbackRan = true</script><img src=/fallback.png></noscript>
      <script>window.scriptRan = true</script></body>"#);
    t.open("https://a.test/");
    assert_eq!(t.eval("getComputedStyle(document.body).opacity"), "\"1\"");
    assert_eq!(t.eval("scriptRan && !window.fallbackRan"), "true");
    assert_eq!(t.eval("document.querySelectorAll('noscript style, noscript img, noscript script').length"), "0");
    assert!(!t.site.requests.borrow().iter().any(|r| r.url.path().starts_with("/fallback")));
    t.eval("document.getElementById('host').innerHTML = '<noscript><style>body { opacity: 0; }</style></noscript>'");
    assert_eq!(t.eval("getComputedStyle(document.body).opacity"), "\"1\"");
    assert_eq!(t.eval("document.querySelectorAll('noscript style').length"), "0");
}

#[test]
fn runs_scripts_in_order_and_mutates_the_dom() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", r#"<p id=out>start</p>
      <script>window.log = ['inline1'];</script>
      <script src="/a.js"></script>
      <script defer src="/d.js"></script>
      <script>log.push('inline2'); document.getElementById('out').textContent = 'changed';
        document.addEventListener('DOMContentLoaded', () => log.push('dcl'));
        window.addEventListener('load', () => log.push('load'));</script>"#);
    t.site.add("https://a.test/a.js", "text/javascript", "log.push('external')");
    t.site.add("https://a.test/d.js", "text/javascript", "log.push('deferred')");
    t.open("https://a.test/");
    assert_eq!(t.eval("log.join(',')"), "\"inline1,external,inline2,deferred,dcl,load\"");
    assert_eq!(t.eval("document.querySelector('#out').textContent"), "\"changed\"");
    assert_eq!(t.eval("document.readyState"), "\"complete\"");
}

#[test]
fn inserted_inline_scripts_execute_before_insertion_returns_once_per_element() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", r#"<script id=outer>
      window.order = ['before'];
      const s = document.createElement('script');
      s.id = 'inner';
      s.text = "order.push('inline:' + document.currentScript.id); window.readyNow = true; let insertedLexical = 42";
      document.head.appendChild(s);
      order.push('after:' + readyNow + ':' + document.currentScript.id);
      s.remove(); document.head.appendChild(s);
      const detached = document.createElement('div');
      const nested = document.createElement('script'); nested.textContent = "order.push('nested')";
      detached.appendChild(nested);
      order.push('detached'); document.body.appendChild(detached);
      const fails = document.createElement('script'); fails.text = "throw Error('expected inline error')";
      document.head.insertBefore(fails, s); order.push('continued');
    </script>"#);
    t.open("https://a.test/");
    assert_eq!(t.eval("order.join(',')"), "\"before,inline:inner,after:true:outer,detached,nested,continued\"");
    assert_eq!(t.eval("document.currentScript"), "null");
    assert_eq!(t.eval("insertedLexical"), "42");
}

#[test]
fn navigation_timing_is_numeric_and_tracks_document_events() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", r#"<script>
      window.nav = performance.getEntriesByType('navigation')[0];
      window.earlyLoad = nav.loadEventEnd;
      document.addEventListener('DOMContentLoaded', () => {
        window.duringDcl = nav.domContentLoadedEventStart;
        window.endDuringDcl = nav.domContentLoadedEventEnd;
      });
      window.addEventListener('load', () => window.duringLoad = nav.loadEventStart);
    </script>"#);
    t.open("https://a.test/");
    assert_eq!(t.eval("nav instanceof PerformanceNavigationTiming && nav instanceof PerformanceResourceTiming && nav instanceof PerformanceEntry"), "true");
    assert_eq!(t.eval("earlyLoad === 0 && endDuringDcl === 0"), "true");
    assert_eq!(t.eval("['fetchStart','domainLookupStart','domainLookupEnd','connectStart','connectEnd','requestStart','responseStart','responseEnd','domInteractive','domComplete','loadEventStart','loadEventEnd'].every(k => Number.isFinite(Math.round(nav[k])))"), "true");
    assert_eq!(t.eval("nav.domInteractive <= duringDcl && duringDcl <= nav.domContentLoadedEventEnd && nav.domContentLoadedEventEnd <= nav.domComplete && nav.domComplete <= duringLoad && duringLoad <= nav.loadEventEnd"), "true");
    assert_eq!(t.eval("nav.duration === nav.loadEventEnd && performance.getEntries()[0] === nav && performance.getEntriesByName(location.href, 'navigation')[0] === nav && performance.getEntriesByName(location.href, 'mark').length === 0"), "true");
    assert_eq!(t.eval("JSON.parse(JSON.stringify(nav)).entryType"), "\"navigation\"");
}

#[test]
fn dom_apis_behave_like_a_browser() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", "<div id=root class='a b'><span>one</span><span>two</span></div>");
    t.open("https://a.test/");
    let checks = [
        ("document.querySelectorAll('span').length", "2"),
        ("document.getElementById('root').classList.contains('b')", "true"),
        ("(() => { const d = document.createElement('div'); d.id = 'new'; d.innerHTML = '<b>bold</b>'; document.body.appendChild(d); return document.querySelector('#new b').textContent })()", "\"bold\""),
        ("document.getElementById('root').children[1].previousElementSibling.textContent", "\"one\""),
        ("(() => { const r = document.getElementById('root'); r.dataset.fooBar = 'x'; return r.getAttribute('data-foo-bar') })()", "\"x\""),
        ("(() => { const r = document.getElementById('root'); r.style.color = 'red'; return r.style.color })()", "\"red\""),
        ("getComputedStyle(document.getElementById('root')).display", "\"block\""),
        ("document.getElementById('root').getBoundingClientRect().width > 100", "true"),
        ("document.body.contains(document.querySelector('span'))", "true"),
        ("(() => { const f = document.createDocumentFragment(); f.append('a', document.createElement('i')); document.body.append(f); return document.body.lastChild.nodeName })()", "\"I\""),
        ("new URL('../x?y=1#z', 'https://a.test/p/q').href", "\"https://a.test/x?y=1#z\""),
        ("new URLSearchParams('a=1&b=two%20words').get('b')", "\"two words\""),
        ("btoa('hello') + atob('aGk=')", "\"aGVsbG8=hi\""),
        ("typeof crypto.randomUUID()", "\"string\""),
        ("(() => { localStorage.setItem('k', 'v'); return localStorage.getItem('k') })()", "\"v\""),
        ("location.hostname", "\"a.test\""),
        ("navigator.userAgent.includes('Chrome')", "true"),
        ("matchMedia('(min-width: 600px)').matches", "true"),
        ("document.querySelector('span') instanceof HTMLSpanElement && document.body instanceof HTMLElement", "true"),
        ("new Intl.NumberFormat('en-US').format(1234567.891)", "\"1,234,567.891\""),
        ("new TextDecoder().decode(new TextEncoder().encode('héllo'))", "\"héllo\""),
    ];
    for (code, want) in checks {
        assert_eq!(t.eval(code), want, "{code}");
    }
}

#[test]
fn constructed_stylesheets_update_rendered_layout_and_live_rule_lists() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", "<style>#box { width: 10px; height: 10px; }</style><div id=box></div>");
    t.open("https://a.test/");
    t.eval("window.sheet = new CSSStyleSheet(); window.rules = sheet.cssRules; sheet.replaceSync('#box { width: 123px; }'); document.adoptedStyleSheets = [sheet]");
    assert_eq!(t.eval("document.getElementById('box').getBoundingClientRect().width"), "123");
    assert_eq!(t.eval("rules.length === 1 && rules[0] instanceof CSSRule && rules[0].parentStyleSheet === sheet"), "true");
    assert_eq!(t.eval("document.styleSheets.length"), "1");
    t.eval("sheet.insertRule('#box { height: 45px; }', 1)");
    assert_eq!(t.eval("document.getElementById('box').getBoundingClientRect().height"), "45");
    assert_eq!(t.eval("rules.length"), "2");
    t.eval("sheet.deleteRule(1); sheet.replace('#box { width: 88px; }').then(s => window.replaced = s === sheet)");
    assert_eq!(t.eval("replaced"), "true");
    assert_eq!(t.eval("document.getElementById('box').getBoundingClientRect().width"), "88");
    t.eval("sheet.disabled = true");
    assert_eq!(t.eval("document.getElementById('box').getBoundingClientRect().width"), "10");
    t.eval("sheet.disabled = false");
    assert_eq!(t.eval("document.getElementById('box').getBoundingClientRect().width"), "88");
    t.eval("document.adoptedStyleSheets.pop()");
    assert_eq!(t.eval("document.getElementById('box').getBoundingClientRect().width"), "10");
}

#[test]
fn constructed_stylesheets_handle_css_boundaries_and_atomic_adoption() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", "<div id=host></div>");
    t.open("https://a.test/");
    t.eval(r#"window.sheet = new CSSStyleSheet(); sheet.replaceSync('/* heading */ @import "skip.css"; @media screen { p { color: red; } } p::before { content: "};{"; }'); window.rules = sheet.cssRules"#);
    assert_eq!(t.eval("rules.length"), "2");
    assert_eq!(t.eval("(() => { try { sheet.insertRule('p{} q{}'); } catch (e) { return e.name; } })()"), "\"SyntaxError\"");
    assert_eq!(t.eval("(() => { try { sheet.insertRule('@import \"x.css\";'); } catch (e) { return e.name; } })()"), "\"SyntaxError\"");
    assert_eq!(t.eval("(() => { try { sheet.deleteRule(2); } catch (e) { return e.name; } })()"), "\"IndexSizeError\"");
    t.eval("document.adoptedStyleSheets.push(sheet); window.shadow = document.getElementById('host').attachShadow({mode:'open'}); shadow.adoptedStyleSheets = [sheet]");
    assert_eq!(t.eval("document.adoptedStyleSheets[0] === sheet && shadow.adoptedStyleSheets[0] === sheet"), "true");
    assert_eq!(t.eval("(() => { try { document.adoptedStyleSheets = [sheet, {}]; } catch (e) { return e.name; } })()"), "\"NotAllowedError\"");
    assert_eq!(t.eval("document.adoptedStyleSheets.length"), "1");
    assert_eq!(t.eval("(() => { try { document.adoptedStyleSheets.push({}); } catch (e) { return e.name; } })()"), "\"NotAllowedError\"");
    assert_eq!(t.eval("document.adoptedStyleSheets.length"), "1");
}

#[test]
fn dynamic_async_false_scripts_wait_for_dependencies_but_async_scripts_do_not() {
    use evolve_browser::{host::{self, Loader}, net::{FetchRequest, FetchResponse}, testing::SiteLoader};
    use std::{cell::RefCell, rc::Rc};
    type Completion = Box<dyn FnOnce(Result<FetchResponse, String>)>;
    struct Delayed {
        site: SiteLoader,
        pending: RefCell<Option<(FetchRequest, Completion)>>,
    }
    impl Loader for Delayed {
        fn fetch(&self, req: FetchRequest, done: Completion) {
            if req.url.path() == "/slow.js" { *self.pending.borrow_mut() = Some((req, done)); }
            else { self.site.fetch(req, done); }
        }
        fn script_cookies(&self, url: &url::Url) -> String { self.site.script_cookies(url) }
        fn set_script_cookie(&self, url: &url::Url, value: &str) { self.site.set_script_cookie(url, value); }
    }
    let mut t = h();
    t.site.add("https://a.test/", "text/html", r#"<script>
      window.order = []; window.defaults = document.createElement('script').async;
      for (const [src, ordered] of [['/slow.js', true], ['/dependent.js', true], ['/async.js', false]]) {
        const s = document.createElement('script'); s.src = src;
        if (ordered) s.async = false;
        document.head.appendChild(s);
      }
    </script>"#);
    t.site.add("https://a.test/slow.js", "text/javascript", "window.Dependency = function() {}; order.push('first')");
    t.site.add("https://a.test/dependent.js", "text/javascript", "new Dependency(); order.push('second')");
    t.site.add("https://a.test/async.js", "text/javascript", "order.push('async')");
    let loader = Rc::new(Delayed { site: SiteLoader(t.site.clone()), pending: RefCell::new(None) });
    host::LOADER.with(|l| *l.borrow_mut() = Some(loader.clone()));
    t.open("https://a.test/");
    assert_eq!(t.eval("defaults"), "true");
    assert_eq!(t.eval("document.querySelector('script').async"), "false");
    assert_eq!(t.eval("order.join(',')"), "\"async\"");
    let (req, done) = loader.pending.borrow_mut().take().unwrap();
    loader.site.fetch(req, done);
    t.settle();
    assert_eq!(t.eval("order.join(',')"), "\"async,first,second\"");
    assert!(!t.console().iter().any(|s| s.contains("ReferenceError")), "{:?}", t.console());
}

#[test]
fn clicks_reach_listeners_and_inline_handlers() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", r#"<style>button{width:200px;height:60px;display:block}</style>
      <button id=b onclick="this.dataset.inline = 'yes'">Press</button><p id=o></p>
      <script>let n = 0; document.getElementById('b').addEventListener('click', e => { n++; document.getElementById('o').textContent = 'clicked ' + n + ' ' + e.isTrusted; });</script>"#);
    t.open("https://a.test/");
    let r: Vec<f64> = serde_json::from_str(&t.eval("JSON.stringify((() => { const r = document.getElementById('b').getBoundingClientRect(); return [r.x, r.y, r.width, r.height] })())").trim_matches('"').replace("\\\"", "\"")).unwrap_or_default();
    let (x, y) = if r.len() == 4 { ((r[0] + r[2] / 2.0) as f32, (r[1] + r[3] / 2.0) as f32) } else { (100.0, 30.0) };
    t.engine.pointer("move", x, y, 0, 0, 0);
    t.engine.pointer("down", x, y, 0, 1, 0);
    t.engine.pointer("up", x, y, 0, 0, 0);
    t.settle();
    assert_eq!(t.eval("document.getElementById('o').textContent"), "\"clicked 1 true\"");
    assert_eq!(t.eval("document.getElementById('b').dataset.inline"), "\"yes\"");
}

#[test]
fn timers_promises_and_animation_frames_run() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", r#"<script>
      window.seen = [];
      setTimeout(() => seen.push('timeout'), 20);
      Promise.resolve().then(() => seen.push('micro'));
      requestAnimationFrame(() => seen.push('raf'));
      const id = setInterval(() => { seen.push('tick'); if (seen.filter(s => s === 'tick').length === 2) clearInterval(id); }, 5);
      (async () => { await new Promise(r => setTimeout(r, 30)); seen.push('async'); })();
    </script>"#);
    t.open("https://a.test/");
    let seen = t.eval("seen.slice().sort().join(',')");
    assert_eq!(seen, "\"async,micro,raf,tick,tick,timeout\"");
}

#[test]
fn fetch_and_xhr_with_cors_rules() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", "<p id=o></p>");
    t.site.add("https://a.test/data.json", "application/json", r#"{"name":"evolve"}"#);
    t.site.add("https://other.test/secret", "application/json", r#"{"secret":1}"#);
    t.site.add_with("https://open.test/public", 200, "application/json", r#"{"open":true}"#, &[("Access-Control-Allow-Origin", "*")]);
    t.open("https://a.test/");
    t.eval("fetch('/data.json').then(r => r.json()).then(j => { document.getElementById('o').textContent = j.name })");
    assert_eq!(t.eval("document.getElementById('o').textContent"), "\"evolve\"");
    t.eval("window.x = ''; const q = new XMLHttpRequest(); q.open('GET', '/data.json'); q.onload = () => { x = q.status + ':' + JSON.parse(q.responseText).name }; q.send()");
    assert_eq!(t.eval("x"), "\"200:evolve\"");
    // Cross-origin without CORS headers is blocked; with ACAO it's readable.
    t.eval("window.c1 = ''; fetch('https://other.test/secret').then(() => c1 = 'read', e => c1 = 'blocked')");
    // (The in-memory loader doesn't enforce CORS; the rustls client does. Here we only check the
    // request was made in CORS mode without credentials.)
    let req = t.site.requests.borrow().iter().find(|r| r.url.as_str() == "https://other.test/secret").cloned().unwrap();
    assert_eq!(req.mode, evolve_browser::net::Mode::Cors);
    assert_eq!(req.credentials, evolve_browser::net::Credentials::SameOrigin);
}

#[test]
fn cookies_from_responses_and_script() {
    let mut t = h();
    t.site.add_with("https://a.test/", 200, "text/html", "<p>hi</p>", &[("Set-Cookie", "session=abc; HttpOnly; Path=/"), ("Set-Cookie", "theme=dark; Path=/")]);
    t.open("https://a.test/");
    assert_eq!(t.eval("document.cookie"), "\"theme=dark\"");
    t.eval("document.cookie = 'lang=en; path=/'");
    let c = t.eval("document.cookie");
    assert!(c.contains("theme=dark") && c.contains("lang=en") && !c.contains("session"), "{c}");
}

#[test]
fn forms_submit_with_script_interception() {
    let mut t = h();
    t.site.add("https://a.test/login", "text/html", r#"<style>input,button{display:block;width:200px;height:30px}</style>
      <form id=f method=post action="/session"><input name=user id=u><input type=password name=pass id=p><button id=go>Log in</button></form>
      <script>window.submits = 0; document.getElementById('f').addEventListener('submit', e => { submits++; if (!window.allow) e.preventDefault(); });</script>"#);
    t.site.add("https://a.test/session", "text/html", "<title>Welcome</title><p>Logged in</p>");
    t.open("https://a.test/login");
    t.eval("document.getElementById('u').value = 'anthony'; document.getElementById('p').value = 'hunter2'");
    t.eval("document.getElementById('go').click()");
    assert_eq!(t.eval("submits"), "1");
    assert!(!t.site.requests.borrow().iter().any(|r| r.url.path() == "/session"), "prevented submit must not navigate");
    t.eval("window.allow = true; document.getElementById('go').click()");
    t.settle();
    let post = t.site.requests.borrow().iter().find(|r| r.url.path() == "/session").cloned().expect("POST sent");
    assert_eq!(post.method, "POST");
    assert_eq!(String::from_utf8_lossy(&post.body), "user=anthony&pass=hunter2");
    assert_eq!(t.eval("document.title"), "\"Welcome\"");
}

#[test]
fn dynamic_scripts_and_modules_load() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", r#"<script type=importmap>{"imports":{"lib":"/lib.js"}}</script>
      <script type=module>import { greet } from 'lib'; import two from './two.js'; window.mod = greet() + two;</script>
      <script>const s = document.createElement('script'); s.src = '/dyn.js'; s.onload = () => window.dynLoaded = true; document.head.appendChild(s);</script>"#);
    t.site.add("https://a.test/lib.js", "text/javascript", "export function greet() { return 'hi' }");
    t.site.add("https://a.test/two.js", "text/javascript", "export default 2");
    t.site.add("https://a.test/dyn.js", "text/javascript", "window.dyn = 'ran'");
    t.open("https://a.test/");
    assert_eq!(t.eval("window.mod"), "\"hi2\"", "console: {:?}", t.console());
    assert_eq!(t.eval("window.dyn + ',' + window.dynLoaded"), "\"ran,true\"");
}

#[test]
fn history_and_fragments() {
    let mut t = h();
    t.site.add("https://a.test/one", "text/html", "<a id=l href='/two'>two</a>");
    t.site.add("https://a.test/two", "text/html", "<title>Two</title><div style='height:3000px'></div><p id=end>end</p>");
    t.open("https://a.test/one");
    t.eval("document.getElementById('l').click()");
    t.settle();
    assert_eq!(t.eval("location.pathname"), "\"/two\"");
    t.eval("history.pushState({n:1}, '', '/two?x=1')");
    assert_eq!(t.eval("location.search"), "\"?x=1\"");
    t.eval("window.pops = 0; addEventListener('popstate', () => pops++); history.back()");
    t.settle();
    assert_eq!(t.eval("location.search + ':' + pops"), "\":1\"");
    t.engine.back();
    t.settle();
    assert_eq!(t.eval("location.pathname"), "\"/one\"");
}

#[test]
fn console_errors_are_reported_not_fatal() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", "<script>throw new Error('boom')</script><script>window.after = 1; console.log('still', {a: 1})</script>");
    t.open("https://a.test/");
    assert_eq!(t.eval("after"), "1");
    let console = t.console();
    assert!(console.iter().any(|l| l.contains("boom")), "{console:?}");
    assert!(console.iter().any(|l| l.contains("still {a: 1}")), "{console:?}");
}

#[test]
fn agent_snapshot_and_actions_share_the_page() {
    let mut t = h();
    t.site.add("https://a.test/login", "text/html", r#"<title>Sign in</title><style>input,button{display:block;width:240px;height:30px;margin:6px}</style>
      <h1>Sign in</h1><form method=post action="/session"><label>Email <input name=email></label><label>Password <input type=password name=pw></label>
      <select name=plan><option value=free>Free</option><option value=pro>Pro</option></select><button>Continue</button></form>"#);
    t.site.add("https://a.test/session", "text/html", "<title>Done</title><p>ok</p>");
    t.open("https://a.test/login");
    let snap = t.engine.snapshot(4000);
    let text = snap["text"].as_str().unwrap().to_string();
    assert!(text.contains("Sign in"), "{text}");
    let find = |needle: &str| -> usize {
        let line = text.split('[').find(|seg| seg.to_lowercase().contains(needle)).unwrap_or_else(|| panic!("{needle} in {text}"));
        line.split(']').next().unwrap().parse().unwrap()
    };
    let email = find("email");
    let pw = find("password");
    let plan = find("combobox");
    assert!(!text.contains("hunter"), "{text}");
    let r = t.engine.agent_type(email, "me@example.com", false);
    assert_eq!(r["ok"], true, "{r}");
    t.settle();
    assert_eq!(t.eval("document.querySelector('[name=email]').value"), "\"me@example.com\"");
    let r = t.engine.agent_type(pw, "secret", false);
    assert_eq!(r["error"], "password", "{r}");
    // The person types the password themselves; the agent's old refs are now stale.
    t.eval("document.querySelector('[name=pw]').focus()");
    t.engine.ime_commit("hunter2");
    t.settle();
    let r = t.engine.agent_select(plan, "pro");
    assert_eq!(r["error"], "stale", "{r}");
    let snap = t.engine.snapshot(4000);
    let text = snap["text"].as_str().unwrap().to_string();
    assert!(!text.contains("hunter2"), "password value leaked: {text}");
    let find = |needle: &str| -> usize {
        let line = text.split('[').find(|seg| seg.to_lowercase().contains(needle)).unwrap();
        line.split(']').next().unwrap().parse().unwrap()
    };
    let r = t.engine.agent_select(find("combobox"), "pro");
    assert_eq!(r["ok"], true, "{r}");
    let r = t.engine.agent_click(find("continue"));
    assert_eq!(r["ok"], true, "{r}");
    t.settle();
    let post = t.site.requests.borrow().iter().find(|r| r.url.path() == "/session").cloned().expect("submitted");
    assert_eq!(String::from_utf8_lossy(&post.body), "email=me%40example.com&pw=hunter2&plan=pro");
    assert_eq!(t.eval("document.title"), "\"Done\"");
}

#[test]
fn replaced_children_stay_usable_by_scripts() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", "<div id=a><b>1</b><i>2</i></div><div id=b></div>");
    t.open("https://a.test/");
    // jQuery's buildFragment pattern: keep childNodes, clear the container, move them elsewhere.
    let r = t.eval("(() => { const a = document.getElementById('a'); const kids = [...a.childNodes]; a.textContent = ''; const b = document.getElementById('b'); kids.forEach(k => b.appendChild(k)); const tmp = document.createElement('div'); tmp.innerHTML = '<u>3</u>'; const u = tmp.firstChild; tmp.innerHTML = ''; b.appendChild(u); return b.textContent + '|' + a.childNodes.length })()");
    assert_eq!(r, "\"123|0\"");
}

#[test]
fn person_select_menu_and_choice() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", "<style>select{width:200px;height:30px}</style><select id=s onchange=\"document.title='picked '+this.value\"><option value=a>Alpha</option><optgroup label=G><option value=b>Beta</option></optgroup></select>");
    t.open("https://a.test/");
    t.events.clear();
    t.engine.pointer("move", 20.0, 20.0, 0, 0, 0);
    t.engine.pointer("down", 20.0, 20.0, 0, 1, 0);
    t.engine.pointer("up", 20.0, 20.0, 0, 0, 0);
    t.settle();
    let menu = t.events.iter().find_map(|e| match e { evolve_browser::host::OutEvent::Select { id, options, .. } => Some((*id, options.clone())), _ => None }).expect("select menu");
    assert_eq!(menu.1.len(), 2);
    assert_eq!(menu.1[1].group.as_deref(), Some("G"));
    assert!(menu.1[0].selected);
    t.engine.choose_option(menu.0, 1);
    t.settle();
    assert_eq!(t.eval("document.getElementById('s').value + ',' + document.title"), "\"b,picked b\"");
}

#[test]
fn typing_inserts_each_character_once() {
    let mut t = h();
    t.site.add("https://a.test/", "text/html", "<style>input{width:200px;height:30px}</style><input id=i><script>window.keys='';document.getElementById('i').addEventListener('keydown',e=>keys+=e.key)</script>");
    t.open("https://a.test/");
    t.engine.pointer("down", 20.0, 15.0, 0, 1, 0);
    t.engine.pointer("up", 20.0, 15.0, 0, 0, 0);
    for c in ["h", "i"] {
        t.engine.key("down", c, &format!("Key{}", c.to_uppercase()), 0, false);
        t.engine.ime_commit(c);
        t.engine.key("up", c, &format!("Key{}", c.to_uppercase()), 0, false);
    }
    t.engine.key("down", "Backspace", "Backspace", 0, false);
    t.engine.key("up", "Backspace", "Backspace", 0, false);
    t.settle();
    assert_eq!(t.eval("document.getElementById('i').value + ':' + keys"), "\"h:hiBackspace\"");
}

// A cross-origin child frame posting to its parent, as Google's sign-in does to probe
// third-party cookies: the parent must receive the message with the child's origin and with
// event.source identical to the iframe's contentWindow.
#[test]
fn cross_origin_iframe_posts_message_to_parent() {
    let mut t = h();
    t.site.add("https://accounts.google.test/", "text/html", r#"<!doctype html><body><script>
        window.__got = null;
        window.addEventListener('message', (e) => { window.__got = { data: e.data, origin: e.origin, sameSource: e.source === frame.contentWindow, isWindow: e.source === frame.contentWindow }; });
        var frame = document.createElement('iframe');
        document.body.appendChild(frame);
        frame.src = 'https://checkcookie.youtube.test/check';
    </script></body>"#);
    t.site.add("https://checkcookie.youtube.test/check", "text/html", r#"<!doctype html><script>
        if (window.parent && window.parent.postMessage) window.parent.postMessage('youtube:0', 'https://accounts.google.test');
    </script>"#);
    t.open("https://accounts.google.test/");
    assert_eq!(t.eval("window.__got && window.__got.data"), "\"youtube:0\"", "{:?}", t.console());
    assert_eq!(t.eval("window.__got.origin"), "\"https://checkcookie.youtube.test\"");
    assert_eq!(t.eval("window.__got.sameSource"), "true", "event.source must be the iframe's contentWindow");
    // The child's realm is its own global: its window is not the parent's.
    assert_eq!(t.eval("frame.contentWindow === window"), "false");
}

#[test]
fn cross_origin_frames_cannot_reach_parent_globals_or_host_capabilities() {
    let mut t = h();
    t.site.add("https://parent.test/", "text/html", r#"<body><script>
      window.privateValue = 'parent-secret'; window.result = null;
      addEventListener('message', e => result = e.data);
      var frame = document.createElement('iframe'); frame.src = 'https://child.test/f'; document.body.appendChild(frame);
    </script>"#);
    t.site.add("https://child.test/f", "text/html", r#"<script>
      const result = {hostHidden: typeof __h === 'undefined', denied: [], localFunction: false};
      for (const key of ['document', 'localStorage', 'privateValue', 'eval', '__h', 'constructor']) {
        try { void parent[key]; result.denied.push('allowed:' + key); }
        catch (e) { result.denied.push(e.name); }
      }
      result.localFunction = parent.postMessage.constructor('return typeof privateValue')() === 'undefined';
      result.nullPrototype = Object.getPrototypeOf(parent) === null;
      try { Object.getOwnPropertyDescriptor(parent, 'document'); result.descriptor = 'allowed'; }
      catch (e) { result.descriptor = e.name; }
      parent.postMessage(result, 'https://parent.test');
    </script>"#);
    t.open("https://parent.test/");
    assert_eq!(t.eval("result && result.hostHidden && result.localFunction && result.nullPrototype && result.descriptor === 'SecurityError' && result.denied.length === 6 && result.denied.every(x => x === 'SecurityError')"), "true", "{:?}", t.console());
    assert_eq!(t.eval("(() => { try { return frame.contentWindow.document; } catch (e) { return e.name; } })()"), "\"SecurityError\"");
}

#[test]
fn iframe_messages_are_queued_and_copied_in_the_receivers_realm() {
    let mut t = h();
    t.site.add("https://parent.test/", "text/html", r#"<body><script>
      window.reply = null; addEventListener('message', e => reply = e.data);
      var frame = document.createElement('iframe'); frame.src = 'https://child.test/f'; document.body.appendChild(frame);
    </script>"#);
    t.site.add("https://child.test/f", "text/html", r#"<script>
      addEventListener('message', e => {
        const local = Object.getPrototypeOf(e.data) === Object.prototype;
        e.source.postMessage({value:e.data.nested.value, local}, e.origin);
        e.data.nested.value = 99;
      });
    </script>"#);
    t.open("https://parent.test/");
    assert_eq!(t.eval("(() => { window.payload = {nested:{value:1}}; frame.contentWindow.postMessage(payload, 'https://child.test'); payload.nested.value = 2; return reply === null; })()"), "true");
    t.settle();
    assert_eq!(t.eval("reply && reply.value === 1 && reply.local && payload.nested.value === 2"), "true", "{:?}", t.console());
    assert_eq!(t.eval("(() => { try { frame.contentWindow.postMessage({fn(){}}, '*'); } catch (e) { return e.name; } })()"), "\"DataCloneError\"");
    assert_eq!(t.eval("(() => { try { frame.contentWindow.postMessage('x', 'not an origin'); } catch (e) { return e.name; } })()"), "\"SyntaxError\"");
}

#[test]
fn iframe_navigation_keeps_proxy_identity_and_revokes_old_realm_capabilities() {
    let mut t = h();
    t.site.add("https://parent.test/", "text/html", r#"<body><script>
      window.messages = [];
      addEventListener('message', e => messages.push(e.data + ':' + e.origin));
      var frame = document.createElement('iframe'); frame.src = '/first'; document.body.appendChild(frame);
    </script>"#);
    t.site.add("https://parent.test/first", "text/html", r#"<script>
      window.canary = 'first'; window.oldSend = () => parent.postMessage('old', '*');
    </script>"#);
    t.site.add("https://child.test/second", "text/html", r#"<script>parent.postMessage('new', 'https://parent.test')</script>"#);
    t.open("https://parent.test/");
    assert_eq!(t.eval("window.cached = frame.contentWindow; window.oldSend = cached.oldSend; cached.canary"), "\"first\"");
    t.eval("frame.src = 'https://child.test/second'");
    t.settle();
    assert_eq!(t.eval("cached === frame.contentWindow"), "true");
    assert_eq!(t.eval("(() => { try { return cached.canary; } catch(e) { return e.name; } })()"), "\"SecurityError\"");
    t.eval("oldSend()");
    assert_eq!(t.eval("messages.join(',')"), "\"new:https://child.test\"");
}

#[test]
fn iframe_sandbox_disables_scripts_and_keeps_script_enabled_frames_opaque() {
    let mut t = h();
    t.site.add("https://parent.test/", "text/html", r#"<body><script>
      window.privateValue = 'parent-secret'; window.messages = [];
      addEventListener('message', e => messages.push(e.data + ':' + e.origin));
    </script><iframe sandbox="allow-scripts" src="/opaque"></iframe><iframe sandbox src="/disabled"></iframe>"#);
    t.site.add("https://parent.test/opaque", "text/html", r#"<script>
      try { parent.privateValue; parent.postMessage('leaked', '*'); }
      catch (e) { parent.postMessage(e.name, '*'); }
    </script>"#);
    t.site.add("https://parent.test/disabled", "text/html", "<script>parent.postMessage('forbidden', '*')</script>");
    t.open("https://parent.test/");
    assert_eq!(t.eval("messages.join(',')"), "\"SecurityError:null\"", "{:?}", t.console());
}

// Both directions: the parent posts into the child frame, the child's listener receives it with
// the parent as source, and replies; the parent then sees the reply from the child.
#[test]
fn iframe_postmessage_is_bidirectional() {
    let mut t = h();
    t.site.add("https://host.test/", "text/html", r#"<!doctype html><body><script>
        window.__reply = null;
        window.addEventListener('message', (e) => { if (e.data && e.data.pong) window.__reply = { data: e.data.pong, origin: e.origin, fromChild: e.source === frame.contentWindow }; });
        var frame = document.createElement('iframe');
        document.body.appendChild(frame);
        frame.src = 'https://widget.test/w';
    </script></body>"#);
    t.site.add("https://widget.test/w", "text/html", r#"<!doctype html><script>
        window.addEventListener('message', (e) => {
            // Reply to the sender using its origin, proving parent->child delivery and source.
            if (e.source && e.source.postMessage) e.source.postMessage({ pong: e.data, parentOrigin: e.origin, parentIsTop: e.source === window.top }, e.origin);
        });
    </script>"#);
    t.open("https://host.test/");
    // Give the child time to attach and register its listener, then post into it.
    t.settle();
    t.eval("frame.contentWindow.postMessage('ping', 'https://widget.test')");
    assert_eq!(t.eval("window.__reply && window.__reply.data"), "\"ping\"", "{:?}", t.console());
    assert_eq!(t.eval("window.__reply.origin"), "\"https://widget.test\"");
    assert_eq!(t.eval("window.__reply.fromChild"), "true");
}

// A same-origin iframe exposes its window synchronously (as a real browser does, about:blank),
// before its content has loaded; a cross-origin one does not until it loads.
#[test]
fn same_origin_iframe_window_is_available_synchronously() {
    let mut t = h();
    t.site.add("https://app.test/", "text/html", "<!doctype html><body></body>");
    t.open("https://app.test/");
    // Same-origin frame: contentWindow is a usable window immediately after the element exists,
    // before any network load.
    assert_eq!(t.eval(r#"(() => { const f = document.createElement('iframe'); f.src = '/relay'; document.body.appendChild(f); const w = f.contentWindow; return w && typeof w.postMessage; })()"#), "\"function\"", "{:?}", t.console());
    // A src-less (about:blank) same-origin frame too.
    assert_eq!(t.eval(r#"(() => { const f = document.createElement('iframe'); document.body.appendChild(f); return f.contentWindow ? 'win' : 'null'; })()"#), "\"win\"");
    // Cross-origin frame: no window until it actually loads.
    assert_eq!(t.eval(r#"(() => { const f = document.createElement('iframe'); f.src = 'https://other.test/x'; document.body.appendChild(f); return f.contentWindow === null ? 'null' : 'win'; })()"#), "\"null\"");
}

// targetOrigin must match the receiver's origin (or be "*"): a mismatched origin is dropped.
#[test]
fn iframe_postmessage_respects_target_origin() {
    let mut t = h();
    t.site.add("https://parent.test/", "text/html", r#"<!doctype html><body><script>
        window.__count = 0;
        window.addEventListener('message', () => { window.__count++; });
        var frame = document.createElement('iframe');
        document.body.appendChild(frame);
        frame.src = 'https://child.test/f';
    </script></body>"#);
    t.site.add("https://child.test/f", "text/html", r#"<!doctype html><script>
        window.parent.postMessage('wrong', 'https://evil.test');
        window.parent.postMessage('right', 'https://parent.test');
        window.parent.postMessage('any', '*');
    </script>"#);
    t.open("https://parent.test/");
    // Only the matching-origin and "*" posts are delivered.
    assert_eq!(t.eval("window.__count"), "2", "{:?}", t.console());
}

#[test]
fn import_meta_and_blob_scripts() {
    let mut t = h();
    t.site.add("https://a.test/app/", "text/html", r#"<script type=module src="/assets/main.js"></script>
      <script>const b = new Blob(["window.fromBlob = 'yes'"], {type: 'text/javascript'}); const s = document.createElement('script'); s.src = URL.createObjectURL(b); document.head.appendChild(s);</script>"#);
    t.site.add("https://a.test/assets/main.js", "text/javascript", "window.meta = import.meta.url + '|' + import.meta.resolve('./chunk.js');");
    t.open("https://a.test/app/");
    assert_eq!(t.eval("window.meta"), "\"https://a.test/assets/main.js|https://a.test/assets/chunk.js\"", "{:?}", t.console());
    assert_eq!(t.eval("window.fromBlob"), "\"yes\"");
}
