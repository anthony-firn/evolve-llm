//! Page snapshots and element geometry for the agent. See API.md section 3.
//!
//! A snapshot is a compact text outline of what a person can see on the page, in document
//! order: headings, text, and numbered interactive elements (`[n] role "name"`). The engine
//! keeps `refs` so the agent can act on `[n]`. Values of password and payment fields are
//! never included.

use blitz_dom::{local_name, BaseDocument, ElementData, Node, NodeData, NodeId};

pub struct Snapshot {
    pub text: String,
    /// ref n = refs[n - 1]
    pub refs: Vec<NodeId>,
}

const MAX_ITEMS: usize = 6000;
const MAX_DEPTH: usize = 512;
const TEXT_CHARS: usize = 200;
const HEADING_CHARS: usize = 150;
const NAME_CHARS: usize = 100;
const VALUE_CHARS: usize = 80;
const HREF_CHARS: usize = 60;
const LINE_CHARS: usize = 220;

const SKIP_TAGS: &[&str] = &["script", "style", "template", "noscript", "head", "title", "meta", "link", "base", "datalist", "param", "source", "track"];
const ARIA_CONTROL_ROLES: &[&str] = &[
    "button", "link", "checkbox", "radio", "switch", "tab", "menuitem", "menuitemcheckbox", "menuitemradio", "option",
    "combobox", "textbox", "searchbox", "slider", "spinbutton", "treeitem", "gridcell",
];

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Heading(usize),
    Control,
    Text,
}

struct Item {
    kind: Kind,
    node: Option<NodeId>,
    text: String,
    y: f32,
    h: f32,
}

/// Build the outline of a fully resolved document, at most `max_chars` characters long.
pub fn snapshot(doc: &BaseDocument, max_chars: usize) -> Snapshot {
    let mut w = Walker::new(doc);
    w.walk(doc.root_node().id, 0);
    w.flush();
    render(w.items, viewport_height(doc), max_chars)
}

/// The element's border box in viewport CSS px (x, y, width, height), if it has one.
pub fn element_box(doc: &BaseDocument, node: NodeId) -> Option<(f32, f32, f32, f32)> {
    let r = doc.get_client_bounding_rect(node)?;
    (r.width > 0.0 || r.height > 0.0).then_some((r.x as f32, r.y as f32, r.width as f32, r.height as f32))
}

/// Whether a field holds a secret the agent must not type or read: passwords (including
/// "show password" text fields) and payment card details.
pub fn is_password(doc: &BaseDocument, node: NodeId) -> bool {
    doc.get_node(node).and_then(|n| n.element_data()).is_some_and(is_secret_field)
}

fn is_secret_field(el: &ElementData) -> bool {
    let tag = &*el.name.local;
    if tag != "input" && tag != "textarea" {
        return false;
    }
    let lower = |name: blitz_dom::LocalName| el.attr(name).map(|v: &str| v.to_ascii_lowercase()).unwrap_or_default();
    if lower(local_name!("type")) == "password" {
        return true;
    }
    let autocomplete = lower(local_name!("autocomplete"));
    if autocomplete.contains("password") || autocomplete.split_whitespace().any(|t| t.starts_with("cc-") && t != "cc-name") || autocomplete.contains("one-time-code") {
        return true;
    }
    let ident = format!("{} {}", lower(local_name!("name")), lower(local_name!("id")));
    ["passw", "pwd", "passcode", "cvv", "cvc", "card-number", "cardnumber", "card_number", "ccnum", "cc-num", "otp"]
        .iter()
        .any(|k| ident.contains(k))
}

fn viewport_height(doc: &BaseDocument) -> f32 {
    let v = doc.viewport();
    let scale = v.scale().max(0.01);
    (v.window_size.1 as f32 / scale).max(1.0)
}

struct Walker<'a> {
    doc: &'a BaseDocument,
    items: Vec<Item>,
    buf: String,
    buf_pos: Option<(f32, f32)>,
    buf_full: bool,
    /// Text is not added while > 0 (inside headings and labels, whose text is used elsewhere)
    suppress: usize,
    focused: Option<NodeId>,
    labels: Vec<(String, NodeId)>,
}

impl<'a> Walker<'a> {
    fn new(doc: &'a BaseDocument) -> Self {
        let labels = doc
            .query_selector_all("label[for]")
            .map(|ids| {
                ids.iter()
                    .filter_map(|&id| {
                        let target = doc.get_node(id)?.attr(local_name!("for"))?.to_string();
                        Some((target, id))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Walker { doc, items: vec![], buf: String::new(), buf_pos: None, buf_full: false, suppress: 0, focused: doc.get_focussed_node_id(), labels }
    }

    fn walk(&mut self, id: NodeId, depth: usize) {
        if self.items.len() >= MAX_ITEMS || depth > MAX_DEPTH {
            return;
        }
        let Some(node) = self.doc.get_node(id) else { return };
        match &node.data {
            NodeData::Document(_) => {
                for &c in &node.children {
                    self.walk(c, depth + 1);
                }
            }
            NodeData::Text(t) => self.push_text(&t.content, node.parent),
            NodeData::Element(el) => self.element(id, node, el, depth),
            NodeData::AnonymousBlock(_) | NodeData::Comment { .. } => {}
        }
    }

    fn displayed(&self, id: NodeId, node: &Node) -> Option<String> {
        node.primary_styles()?;
        let display = self.doc.resolved_style_value(id, "display");
        if display == "none" {
            return None;
        }
        let visibility = self.doc.resolved_style_value(id, "visibility");
        if visibility == "hidden" || visibility == "collapse" {
            return None;
        }
        Some(display)
    }

    fn element(&mut self, id: NodeId, node: &Node, el: &ElementData, depth: usize) {
        let tag = &*el.name.local;
        if SKIP_TAGS.contains(&tag) || el.attr(local_name!("aria-hidden")) == Some("true") {
            return;
        }
        if tag == "input" && el.attr(local_name!("type")).is_some_and(|t| t.eq_ignore_ascii_case("hidden")) {
            return;
        }
        let Some(display) = self.displayed(id, node) else { return };
        let is_block = !(display.starts_with("inline") || display == "contents" || display.is_empty() || display.starts_with("ruby"));
        if is_block {
            self.flush();
        }

        if let Some(level) = heading_level(el, tag) {
            self.flush();
            let text = clip(&collapse(&node.text_content()), HEADING_CHARS);
            if !text.is_empty() && self.has_area(id) {
                let (y, h) = self.pos(id);
                self.items.push(Item { kind: Kind::Heading(level), node: Some(id), text, y, h });
            }
            // Links and buttons inside headings still get refs; their text is already shown.
            self.suppress += 1;
            for &c in &node.children {
                self.walk(c, depth + 1);
            }
            self.suppress -= 1;
            self.flush();
            return;
        }

        if let Some(role) = self.control_role(id, el, tag) {
            // Checkboxes are often visually replaced by styled labels, and Blitz draws no box
            // for <select>; both are still usable controls.
            if self.has_area(id) || matches!(role, "checkbox" | "radio") || tag == "select" {
                self.flush();
                let text = self.describe(id, node, el, tag, role);
                let (y, h) = self.pos(id);
                self.items.push(Item { kind: Kind::Control, node: Some(id), text, y, h });
            }
            return;
        }

        if tag == "img" || (tag == "input" && el.attr(local_name!("type")) == Some("image")) {
            if let Some(alt) = el.attr(local_name!("alt")).map(collapse).filter(|a| !a.is_empty()) {
                if self.has_area(id) && self.suppress == 0 {
                    self.flush();
                    let (y, h) = self.pos(id);
                    self.items.push(Item { kind: Kind::Text, node: Some(id), text: format!("image \"{}\"", clip(&alt, NAME_CHARS)), y, h });
                }
            }
            return;
        }
        if tag == "iframe" {
            if self.has_area(id) {
                self.flush();
                let (y, h) = self.pos(id);
                let title = el.attr(local_name!("title")).map(collapse).unwrap_or_default();
                let text = if title.is_empty() { "(embedded frame)".to_string() } else { format!("(embedded frame \"{}\")", clip(&title, NAME_CHARS)) };
                self.items.push(Item { kind: Kind::Text, node: Some(id), text, y, h });
            }
            return;
        }
        if tag == "br" {
            self.buf.push(' ');
            return;
        }

        let is_label = tag == "label" && self.label_target(id, el).is_some();
        if is_label {
            self.suppress += 1;
        }
        for &c in &node.children {
            self.walk(c, depth + 1);
        }
        if is_label {
            self.suppress -= 1;
        }
        if is_block {
            self.flush();
        }
    }

    fn has_area(&self, id: NodeId) -> bool {
        self.doc.get_client_bounding_rect(id).is_some_and(|r| r.width > 0.5 && r.height > 0.5)
    }

    /// Vertical position of the element, or of its nearest ancestor that has a box.
    fn pos(&self, id: NodeId) -> (f32, f32) {
        let mut cur = Some(id);
        while let Some(n) = cur {
            if let Some(r) = self.doc.get_client_bounding_rect(n).filter(|r| r.width > 0.0 || r.height > 0.0) {
                return (r.y as f32, r.height as f32);
            }
            cur = self.doc.get_node(n).and_then(|n| n.parent);
        }
        (0.0, 0.0)
    }

    fn push_text(&mut self, text: &str, parent: Option<NodeId>) {
        if self.suppress > 0 {
            return;
        }
        let collapsed = collapse(text);
        if collapsed.is_empty() {
            if !self.buf.is_empty() && !self.buf.ends_with(' ') {
                self.buf.push(' ');
            }
            return;
        }
        let Some(parent) = parent else { return };
        let Some(rect) = self.doc.get_client_bounding_rect(parent) else { return };
        // Screen-reader-only text (1px clipped boxes) and collapsed containers are not visible.
        if rect.width <= 1.0 || rect.height <= 1.0 {
            return;
        }
        if self.buf_pos.is_none() {
            self.buf_pos = Some((rect.y as f32, rect.height as f32));
        }
        if self.buf_full {
            return;
        }
        if !self.buf.is_empty() && !self.buf.ends_with(' ') && starts_word(text) {
            self.buf.push(' ');
        }
        self.buf.push_str(&collapsed);
        if text.ends_with(char::is_whitespace) {
            self.buf.push(' ');
        }
        if self.buf.len() > TEXT_CHARS * 2 {
            self.buf_full = true;
        }
    }

    fn flush(&mut self) {
        let text = clip(self.buf.trim(), TEXT_CHARS);
        if !text.is_empty() {
            let (y, h) = self.buf_pos.unwrap_or((0.0, 0.0));
            self.items.push(Item { kind: Kind::Text, node: None, text, y, h });
        }
        self.buf.clear();
        self.buf_pos = None;
        self.buf_full = false;
    }

    fn control_role(&self, id: NodeId, el: &ElementData, tag: &str) -> Option<&'static str> {
        let native = match tag {
            "a" | "area" if el.attr(local_name!("href")).is_some() => Some("link"),
            "button" => Some("button"),
            "input" => Some(match el.attr(local_name!("type")).map(|t| t.to_ascii_lowercase()).as_deref() {
                Some("submit" | "reset" | "button" | "image") => "button",
                Some("checkbox") => "checkbox",
                Some("radio") => "radio",
                Some("range") => "slider",
                Some("file") => "file",
                Some("search") => "searchbox",
                Some("color") => "colorpicker",
                _ if is_secret_field(el) => "password",
                _ => "textbox",
            }),
            "textarea" => Some(if is_secret_field(el) { "password" } else { "textbox" }),
            "select" => Some("combobox"),
            "summary" => Some("summary"),
            _ => None,
        };
        if native.is_some() {
            return native;
        }
        let aria = el.attr(local_name!("role")).map(|r| r.trim().to_ascii_lowercase());
        let generic = if let Some(role) = aria.as_deref().and_then(|r| ARIA_CONTROL_ROLES.iter().find(|x| **x == r)) {
            Some(*role)
        } else if el.attr(local_name!("contenteditable")).is_some_and(|v| v.is_empty() || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("plaintext-only")) {
            Some("textbox")
        } else if el.attr(local_name!("onclick")).is_some()
            || el.attr(local_name!("tabindex")).and_then(|t| t.trim().parse::<i32>().ok()).is_some_and(|t| t >= 0)
        {
            Some("clickable")
        } else {
            None
        };
        // A generic container that holds real controls is walked instead, so those get refs.
        generic.filter(|_| !self.has_control_descendant(id))
    }

    fn has_control_descendant(&self, id: NodeId) -> bool {
        let mut stack: Vec<NodeId> = self.doc.get_node(id).map(|n| n.children.to_vec()).unwrap_or_default();
        let mut seen = 0;
        while let Some(c) = stack.pop() {
            seen += 1;
            if seen > 2000 {
                return false;
            }
            let Some(node) = self.doc.get_node(c) else { continue };
            if let Some(el) = node.element_data() {
                let tag = &*el.name.local;
                if matches!(tag, "button" | "input" | "select" | "textarea" | "summary")
                    || (tag == "a" && el.attr(local_name!("href")).is_some())
                    || el.attr(local_name!("role")).is_some_and(|r| ARIA_CONTROL_ROLES.contains(&r.trim().to_ascii_lowercase().as_str()))
                {
                    return true;
                }
                stack.extend(node.children.iter().copied());
            }
        }
        false
    }

    fn label_target(&self, label: NodeId, el: &ElementData) -> Option<NodeId> {
        if let Some(target) = el.attr(local_name!("for")) {
            return self.doc.get_element_by_id(target);
        }
        let mut stack: Vec<NodeId> = self.doc.get_node(label)?.children.to_vec();
        while let Some(c) = stack.pop() {
            let node = self.doc.get_node(c)?;
            if let Some(e) = node.element_data() {
                if matches!(&*e.name.local, "input" | "select" | "textarea" | "button") {
                    return Some(c);
                }
                stack.extend(node.children.iter().copied());
            }
        }
        None
    }

    fn describe(&self, id: NodeId, node: &Node, el: &ElementData, tag: &str, role: &str) -> String {
        let mut out = String::from(role);
        let name = clip(&self.name(id, node, el, tag, role), NAME_CHARS);
        if !name.is_empty() {
            out.push_str(&format!(" \"{}\"", name.replace('"', "'")));
        }
        let input_type = el.attr(local_name!("type")).map(|t| t.to_ascii_lowercase()).unwrap_or_default();
        match role {
            "password" => out.push_str(" (value hidden)"),
            "textbox" | "searchbox" | "slider" | "colorpicker" => {
                if matches!(input_type.as_str(), "email" | "number" | "tel" | "url" | "date" | "time" | "datetime-local" | "month" | "week") {
                    out.push_str(&format!(" type={input_type}"));
                }
                if tag == "input" || tag == "textarea" {
                    out.push_str(&format!(" value=\"{}\"", clip(&field_value(el), VALUE_CHARS).replace('"', "'")));
                } else if role == "textbox" {
                    let text = clip(&collapse(&node.text_content()), VALUE_CHARS);
                    out.push_str(&format!(" value=\"{}\"", text.replace('"', "'")));
                }
            }
            "checkbox" | "radio" | "switch" | "menuitemcheckbox" | "menuitemradio" => {
                let checked = if tag == "input" {
                    el.checkbox_input_checked().unwrap_or_else(|| el.attr(local_name!("checked")).is_some())
                } else {
                    el.attr(local_name!("aria-checked")) == Some("true")
                };
                out.push_str(if checked { " checked" } else { " unchecked" });
            }
            "combobox" if tag == "select" => {
                let options = self.options(id);
                let selected = options.iter().find(|o| o.1).or(options.first()).map(|o| o.0.clone()).unwrap_or_default();
                out.push_str(&format!(" value=\"{}\"", clip(&selected, VALUE_CHARS).replace('"', "'")));
                if !options.is_empty() {
                    let list: Vec<String> = options.iter().take(8).map(|o| format!("\"{}\"", clip(&o.0, 40).replace('"', "'"))).collect();
                    out.push_str(&format!(" options: {}", list.join(" | ")));
                    if options.len() > 8 {
                        out.push_str(&format!(" (+{} more)", options.len() - 8));
                    }
                }
            }
            "link" => {
                if let Some(href) = el.attr(local_name!("href")) {
                    if let Some(short) = self.short_href(href) {
                        out.push_str(&format!(" -> {short}"));
                    }
                }
            }
            _ => {}
        }
        if el.attr(local_name!("aria-expanded")) == Some("true") {
            out.push_str(" expanded");
        } else if el.attr(local_name!("aria-expanded")) == Some("false") {
            out.push_str(" collapsed");
        }
        if el.attr(local_name!("disabled")).is_some() || el.attr(local_name!("aria-disabled")) == Some("true") {
            out.push_str(" disabled");
        }
        if el.attr(local_name!("required")).is_some() {
            out.push_str(" required");
        }
        if self.focused == Some(id) {
            out.push_str(" focused");
        }
        out
    }

    fn name(&self, id: NodeId, node: &Node, el: &ElementData, tag: &str, role: &str) -> String {
        if let Some(ids) = el.attr(local_name!("aria-labelledby")) {
            let text: Vec<String> = ids
                .split_whitespace()
                .filter_map(|i| self.doc.get_element_by_id(i))
                .filter_map(|n| self.doc.get_node(n))
                .map(|n| collapse(&n.text_content()))
                .filter(|t| !t.is_empty())
                .collect();
            if !text.is_empty() {
                return text.join(" ");
            }
        }
        if let Some(label) = el.attr(local_name!("aria-label")).map(collapse).filter(|l| !l.is_empty()) {
            return label;
        }
        let attr = |name: blitz_dom::LocalName| el.attr(name).map(collapse).filter(|v: &String| !v.is_empty());
        let is_field = matches!(tag, "input" | "textarea" | "select") && role != "button";
        if is_field {
            if let Some(eid) = el.attr(local_name!("id")) {
                let text: Vec<String> = self
                    .labels
                    .iter()
                    .filter(|(target, _)| target == eid)
                    .filter_map(|(_, label)| self.doc.get_node(*label))
                    .map(|l| self.text_without(l, id))
                    .filter(|t| !t.is_empty())
                    .collect();
                if !text.is_empty() {
                    return text.join(" ");
                }
            }
            let mut cur = node.parent;
            while let Some(p) = cur {
                let Some(pn) = self.doc.get_node(p) else { break };
                if pn.element_data().is_some_and(|e| &*e.name.local == "label") {
                    let text = self.text_without(pn, id);
                    if !text.is_empty() {
                        return text;
                    }
                    break;
                }
                cur = pn.parent;
            }
            return attr(local_name!("placeholder"))
                .or_else(|| attr(local_name!("title")))
                .or_else(|| attr(local_name!("name")))
                .unwrap_or_default();
        }
        if role == "textbox" || role == "searchbox" {
            // Editable regions are named by their label, not by what has been typed in them.
            return attr(local_name!("aria-placeholder")).or_else(|| attr(local_name!("title"))).unwrap_or_default();
        }
        if tag == "input" {
            // input buttons
            return attr(local_name!("value")).or_else(|| attr(local_name!("alt"))).unwrap_or_else(|| {
                match el.attr(local_name!("type")).map(|t| t.to_ascii_lowercase()).as_deref() {
                    Some("reset") => "Reset".into(),
                    Some("submit") | Some("image") => "Submit".into(),
                    _ => String::new(),
                }
            });
        }
        let text = self.visible_text(node);
        if !text.is_empty() {
            return text;
        }
        // Icon-only links and buttons: use an image's alt, an SVG title or a descendant's label.
        let mut stack: Vec<NodeId> = node.children.to_vec();
        while let Some(c) = stack.pop() {
            let Some(n) = self.doc.get_node(c) else { continue };
            if let Some(e) = n.element_data() {
                let t = &*e.name.local;
                if let Some(v) = e.attr(local_name!("aria-label")).or(if t == "img" { e.attr(local_name!("alt")) } else { None }).map(collapse).filter(|v| !v.is_empty()) {
                    return v;
                }
                if t == "title" {
                    let v = collapse(&n.text_content());
                    if !v.is_empty() {
                        return v;
                    }
                }
                stack.extend(n.children.iter().copied());
            }
        }
        attr(local_name!("title")).unwrap_or_default()
    }

    /// Text content of a subtree, skipping hidden and script elements.
    fn visible_text(&self, node: &Node) -> String {
        let mut out = String::new();
        self.collect_text(node, None, &mut out, 0);
        collapse(&out)
    }

    fn text_without(&self, node: &Node, exclude: NodeId) -> String {
        let mut out = String::new();
        self.collect_text(node, Some(exclude), &mut out, 0);
        collapse(&out)
    }

    fn collect_text(&self, node: &Node, exclude: Option<NodeId>, out: &mut String, depth: usize) {
        if depth > 64 || out.len() > 400 {
            return;
        }
        for &c in &node.children {
            if Some(c) == exclude {
                continue;
            }
            let Some(n) = self.doc.get_node(c) else { continue };
            match &n.data {
                NodeData::Text(t) => {
                    out.push_str(&t.content);
                    out.push(' ');
                }
                NodeData::Element(e) => {
                    let tag = &*e.name.local;
                    if SKIP_TAGS.contains(&tag) || matches!(tag, "select" | "textarea") || n.primary_styles().is_none() || self.doc.resolved_style_value(c, "display") == "none" {
                        continue;
                    }
                    self.collect_text(n, exclude, out, depth + 1);
                }
                _ => {}
            }
        }
    }

    /// (text, selected) for each option of a select.
    fn options(&self, select: NodeId) -> Vec<(String, bool)> {
        let mut out = vec![];
        let mut stack: Vec<NodeId> = self.doc.get_node(select).map(|n| n.children.iter().rev().copied().collect()).unwrap_or_default();
        while let Some(c) = stack.pop() {
            let Some(n) = self.doc.get_node(c) else { continue };
            let Some(e) = n.element_data() else { continue };
            match &*e.name.local {
                "option" => out.push((collapse(&n.text_content()), e.attr(local_name!("selected")).is_some())),
                "optgroup" => stack.extend(n.children.iter().rev().copied()),
                _ => {}
            }
        }
        out
    }

    fn short_href(&self, href: &str) -> Option<String> {
        let trimmed = href.trim();
        if trimmed.is_empty() || trimmed == "#" || trimmed.to_ascii_lowercase().starts_with("javascript:") {
            return None;
        }
        let base = self.doc.url();
        let url = base.join(trimmed).ok()?;
        let text = if url.origin() == base.origin() {
            let mut s = url.path().to_string();
            if let Some(q) = url.query() {
                s.push('?');
                s.push_str(q);
            }
            if let Some(f) = url.fragment() {
                if url.path() == base.path() && url.query() == base.query() {
                    s = format!("#{f}");
                }
            }
            s
        } else if matches!(url.scheme(), "http" | "https") {
            let mut s = url.host_str().unwrap_or("").trim_start_matches("www.").to_string();
            if url.path() != "/" {
                s.push_str(url.path());
            }
            if let Some(q) = url.query() {
                s.push('?');
                s.push_str(q);
            }
            s
        } else {
            url.as_str().to_string()
        };
        Some(clip(&text, HREF_CHARS))
    }
}

fn field_value(el: &ElementData) -> String {
    if is_secret_field(el) {
        return String::new();
    }
    if let Some(input) = el.text_input_data() {
        return input.editor.raw_text().to_string();
    }
    el.attr(local_name!("value")).unwrap_or("").to_string()
}

fn heading_level(el: &ElementData, tag: &str) -> Option<usize> {
    if let [b'h', d @ b'1'..=b'6'] = tag.as_bytes() {
        return Some((d - b'0') as usize);
    }
    if el.attr(local_name!("role")).is_some_and(|r| r.trim().eq_ignore_ascii_case("heading")) {
        return Some(el.attr(local_name!("aria-level")).and_then(|l| l.trim().parse().ok()).unwrap_or(2usize).clamp(1, 6));
    }
    None
}

fn starts_word(text: &str) -> bool {
    text.starts_with(char::is_whitespace)
}

/// Collapse runs of whitespace to single spaces and trim.
fn collapse(s: &str) -> String {
    let mut out = String::with_capacity(s.len().min(1024));
    let mut space = false;
    for c in s.chars() {
        if c.is_whitespace() || c == '\u{200b}' {
            space = !out.is_empty();
        } else {
            if space {
                out.push(' ');
                space = false;
            }
            out.push(c);
        }
    }
    out
}

/// Truncate to at most `max` characters, ending with an ellipsis when cut.
fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out = out.trim_end().to_string();
    out.push('…');
    out
}

fn render(items: Vec<Item>, viewport_h: f32, max_chars: usize) -> Snapshot {
    // Size of each item once rendered (refs need up to "[999] ").
    let cost = |it: &Item| -> usize {
        it.text.chars().count()
            + match it.kind {
                Kind::Control => 6,
                Kind::Heading(l) => l + 2,
                Kind::Text => 2,
            }
    };
    let total: usize = items.iter().map(cost).sum();
    let mut keep = vec![true; items.len()];
    if total > max_chars {
        // Prefer what is on screen, then what is nearest to it.
        let distance = |it: &Item| -> f32 {
            if it.y + it.h >= 0.0 && it.y <= viewport_h {
                0.0
            } else if it.y < 0.0 {
                // Content above the viewport is slightly less useful than content below it.
                (-(it.y + it.h)) * 1.5
            } else {
                it.y - viewport_h
            }
        };
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by(|&a, &b| distance(&items[a]).partial_cmp(&distance(&items[b])).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        keep = vec![false; items.len()];
        let budget = max_chars.saturating_sub(90);
        let mut used = 0;
        for i in order {
            let c = cost(&items[i]);
            if used + c > budget {
                if c > 60 {
                    continue; // a smaller item may still fit
                }
                break;
            }
            used += c;
            keep[i] = true;
        }
    }
    let first_kept = keep.iter().position(|k| *k);
    let last_kept = keep.iter().rposition(|k| *k);
    let (mut above, mut below) = (0usize, 0usize);
    for (i, it) in items.iter().enumerate() {
        if keep[i] {
            continue;
        }
        if first_kept.is_some_and(|f| i < f) || (it.y < 0.0 && last_kept.is_some_and(|l| i < l)) {
            above += 1;
        } else {
            below += 1;
        }
    }

    let mut refs = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    if above > 0 {
        lines.push(format!("… {above} more items above (scroll up)"));
    }
    let mut line = String::new();
    let mut line_y: Option<f32> = None;
    for (i, it) in items.iter().enumerate() {
        if !keep[i] {
            continue;
        }
        let rendered = match it.kind {
            Kind::Heading(level) => format!("{} {}", "#".repeat(level), it.text),
            Kind::Control => {
                refs.push(it.node.expect("controls have nodes"));
                format!("[{}] {}", refs.len(), it.text)
            }
            Kind::Text => it.text.clone(),
        };
        let same_line = matches!(it.kind, Kind::Control | Kind::Text)
            && line_y.is_some_and(|y| (y - it.y).abs() < 6.0)
            && line.chars().count() + rendered.chars().count() + 2 <= LINE_CHARS;
        if same_line {
            line.push_str("  ");
            line.push_str(&rendered);
        } else {
            if !line.is_empty() {
                lines.push(std::mem::take(&mut line));
            }
            line = rendered;
        }
        line_y = if matches!(it.kind, Kind::Heading(_)) { None } else { Some(it.y) };
        if matches!(it.kind, Kind::Heading(_)) {
            lines.push(std::mem::take(&mut line));
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    if below > 0 {
        lines.push(format!("… {below} more items below (scroll down)"));
    }
    let mut text = lines.join("\n");
    if text.chars().count() > max_chars {
        text = clip(&text, max_chars);
        // Drop refs whose labels were cut off.
        let mut n = refs.len();
        while n > 0 && !text.contains(&format!("[{n}] ")) {
            n -= 1;
        }
        refs.truncate(n);
    }
    Snapshot { text, refs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blitz_dom::DocumentConfig;
    use blitz_html::HtmlDocument;
    use blitz_traits::shell::{ColorScheme, Viewport};

    fn fonts_dir() -> std::path::PathBuf {
        std::env::var_os("EVOLVE_BROWSER_FONTS")
            .map(Into::into)
            .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fonts"))
    }

    fn doc(html: &str) -> HtmlDocument {
        let fonts = crate::fonts::load_dir(&fonts_dir());
        assert!(!fonts.is_empty(), "fonts not found in {:?}", fonts_dir());
        let mut d = HtmlDocument::from_html(
            html,
            DocumentConfig {
                viewport: Some(Viewport::new(800, 600, 1.0, ColorScheme::Light)),
                base_url: Some("https://example.com/start".into()),
                font_ctx: Some(crate::fonts::font_context(&fonts)),
                ..Default::default()
            },
        );
        d.resolve(0.0);
        d
    }

    fn snap(html: &str, max: usize) -> Snapshot {
        let d = doc(html);
        snapshot(&d, max)
    }

    #[test]
    fn login_form_has_names_values_and_hidden_passwords() {
        let d = doc(r#"<body>
            <h1>Sign in</h1>
            <form action="/login" method="post">
              <label for="u">Username</label> <input id="u" name="acct" value="anthony">
              <label>Password <input type="password" name="pw" value="hunter2"></label>
              <label><input type="checkbox" name="remember" checked> Remember me</label>
              <select name="lang"><option>English</option><option selected>Deutsch</option></select>
              <input name="code" autocomplete="one-time-code" value="123456">
              <input type="email" placeholder="you@example.com">
              <button type="submit">Log in</button>
            </form></body>"#);
        let s = snapshot(&d, 4000);
        println!("{}", s.text);
        assert!(s.text.starts_with("# Sign in"), "{}", s.text);
        assert!(s.text.contains(r#"textbox "Username" value="anthony""#), "{}", s.text);
        assert!(s.text.contains(r#"password "Password" (value hidden)"#), "{}", s.text);
        assert!(s.text.contains(r#"checkbox "Remember me" checked"#), "{}", s.text);
        assert!(s.text.contains(r#"combobox "lang" value="Deutsch" options: "English" | "Deutsch""#), "{}", s.text);
        assert!(s.text.contains(r#"textbox "you@example.com" type=email value="""#), "{}", s.text);
        assert!(s.text.contains(r#"password "code" (value hidden)"#), "{}", s.text);
        assert!(s.text.contains(r#"button "Log in""#), "{}", s.text);
        assert!(!s.text.contains("hunter2") && !s.text.contains("123456"), "{}", s.text);
        // Label text is not repeated outside the control.
        assert_eq!(s.text.matches("Username").count(), 1, "{}", s.text);
        assert_eq!(s.refs.len(), 7);
        assert_eq!(d.get_node(s.refs[0]).unwrap().attr(local_name!("id")), Some("u"));
        assert!(is_password(&d, s.refs[1]));
        assert!(is_password(&d, s.refs[4]));
        assert!(!is_password(&d, s.refs[0]));
        for (i, r) in s.refs.iter().enumerate() {
            assert!(s.text.contains(&format!("[{}] ", i + 1)));
            // Blitz 0.3 draws no box for <select>, so only it may lack one.
            let is_select = d.get_node(*r).and_then(|n| n.element_data()).is_some_and(|e| &*e.name.local == "select");
            assert!(element_box(&d, *r).is_some() || is_select, "ref {} has no box", i + 1);
        }
    }

    #[test]
    fn links_show_short_targets() {
        let s = snap(
            r##"<p>Read <a href="/a?x=1">Alpha</a> and <a href="https://www.other.example/b/c">Beta</a>.</p>
               <ul><li><a href="javascript:void(0)">Gamma</a></li><li><a href="#top">Top</a></li>
               <li><a href="/icon"><img src="x.png" alt="Settings" width="16" height="16"></a></li>
               <li><a href="/svg" aria-label="Close dialog"><svg width="10" height="10"></svg></a></li></ul>"##,
            4000,
        );
        println!("{}", s.text);
        assert!(s.text.contains(r#"[1] link "Alpha" -> /a?x=1"#), "{}", s.text);
        assert!(s.text.contains(r#"[2] link "Beta" -> other.example/b/c"#), "{}", s.text);
        assert!(s.text.contains(r#"[3] link "Gamma""#) && !s.text.contains("javascript"), "{}", s.text);
        assert!(s.text.contains(r#"link "Top" -> /start#top"#) || s.text.contains(r#"link "Top" -> #top"#), "{}", s.text);
        assert!(s.text.contains(r#"link "Settings" -> /icon"#), "{}", s.text);
        assert!(s.text.contains(r#"link "Close dialog" -> /svg"#), "{}", s.text);
        assert!(s.text.contains("Read"), "{}", s.text);
        assert_eq!(s.refs.len(), 6);
    }

    #[test]
    fn hidden_content_is_left_out() {
        let s = snap(
            r#"<style>.gone{display:none}.invisible{visibility:hidden}</style>
               <p>Visible text</p>
               <div class="gone"><button>Hidden by class</button><p>secret paragraph</p></div>
               <div hidden><a href="/x">hidden attr</a></div>
               <div class="invisible"><button>Invisible</button></div>
               <div aria-hidden="true"><button>Aria hidden</button></div>
               <input type="hidden" name="csrf" value="tok">
               <button style="width:0;height:0;padding:0;border:0;overflow:hidden">Zero</button>
               <span style="position:absolute;width:1px;height:1px;overflow:hidden">screen reader only</span>
               <script>var notShown = 1;</script><noscript>Enable JS</noscript>
               <button>Shown</button>"#,
            4000,
        );
        println!("{}", s.text);
        assert!(s.text.contains("Visible text"));
        assert!(s.text.contains(r#"[1] button "Shown""#), "{}", s.text);
        for hidden in ["Hidden by class", "secret paragraph", "hidden attr", "Invisible", "Aria hidden", "csrf", "tok", "Zero", "notShown", "Enable JS", "screen reader only"] {
            assert!(!s.text.contains(hidden), "{hidden:?} leaked into {}", s.text);
        }
        assert_eq!(s.refs.len(), 1);
    }

    #[test]
    fn long_pages_fit_the_budget_and_prefer_the_viewport() {
        let mut html = String::from("<body style='margin:0'><h1>Top of page</h1>");
        for i in 0..300 {
            html.push_str(&format!("<p>Paragraph number {i} with some words to make it longer than a few characters.</p><a href='/item/{i}'>Item {i}</a>"));
        }
        let long = "word ".repeat(200);
        html.push_str(&format!("<p>{long}</p></body>"));
        let s = snap(&html, 1500);
        println!("{}", s.text);
        assert!(s.text.chars().count() <= 1500, "{} chars", s.text.chars().count());
        assert!(s.text.starts_with("# Top of page"), "{}", s.text);
        assert!(s.text.contains("Item 0"));
        assert!(s.text.contains("more items below (scroll down)"), "{}", s.text);
        assert!(!s.text.contains("Item 299"));
        let shown = (1..=s.refs.len()).filter(|n| s.text.contains(&format!("[{n}] "))).count();
        assert_eq!(shown, s.refs.len());
        assert!(s.refs.len() > 3);

        // A single long paragraph is truncated to about TEXT_CHARS.
        let s = snap(&format!("<p>{long}</p>"), 4000);
        assert!(s.text.chars().count() <= TEXT_CHARS && s.text.ends_with('…'), "{}", s.text);
    }

    #[test]
    fn scrolled_pages_report_content_above() {
        let mut html = String::from("<body style='margin:0'>");
        for i in 0..200 {
            html.push_str(&format!("<div style='height:40px'><a href='/n/{i}'>Entry {i}</a></div>"));
        }
        let mut d = doc(&html);
        d.set_viewport_scroll(blitz_dom::Point { x: 0.0, y: 4000.0 });
        let s = snapshot(&d, 1200);
        println!("{}", s.text);
        assert!(s.text.contains("Entry 100"), "{}", s.text);
        assert!(s.text.starts_with("… "), "{}", s.text);
        assert!(s.text.contains("more items above"), "{}", s.text);
        let (_, y, _, h) = element_box(&d, s.refs[0]).unwrap();
        assert!(h > 0.0 && y < 0.0, "first shown ref should be above the viewport: y={y}");
    }

    #[test]
    fn element_boxes_are_viewport_relative() {
        let mut d = doc("<body style='margin:0'><div style='height:50px'></div><div id='b' style='width:120px;height:30px'>Go</div><div style='height:2000px'></div></body>");
        let b = d.get_element_by_id("b").unwrap();
        let (x, y, w, h) = element_box(&d, b).unwrap();
        assert_eq!((x, y, w, h), (0.0, 50.0, 120.0, 30.0));
        d.set_viewport_scroll(blitz_dom::Point { x: 0.0, y: 20.0 });
        assert_eq!(element_box(&d, b).unwrap().1, 30.0);
    }

    #[test]
    fn generic_controls_and_focus() {
        let s = snap(
            r#"<div role="button" tabindex="0">Menu</div>
               <div onclick="go()"><a href="/inner">Inner link</a></div>
               <span tabindex="0">Focusable span</span>
               <div contenteditable="true">Draft text</div>
               <details><summary>More</summary><p>Body</p></details>
               <h2><a href="/title">Story title</a></h2>"#,
            4000,
        );
        println!("{}", s.text);
        assert!(s.text.contains(r#"button "Menu""#), "{}", s.text);
        assert!(s.text.contains(r#"link "Inner link""#) && !s.text.contains("clickable \"Inner"), "{}", s.text);
        assert!(s.text.contains(r#"clickable "Focusable span""#), "{}", s.text);
        assert!(s.text.contains(r#"textbox value="Draft text""#), "{}", s.text);
        assert!(s.text.contains(r#"summary "More""#), "{}", s.text);
        assert!(s.text.contains("## Story title") && s.text.contains(r#"link "Story title" -> /title"#), "{}", s.text);
        assert_eq!(s.text.matches("Story title").count(), 2, "{}", s.text);
    }
}
