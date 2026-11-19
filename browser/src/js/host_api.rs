//! The `__h` object: primitive operations the JavaScript DOM (prelude.js) builds on.
//! Node ids are Blitz `NodeId`s passed as numbers; -1 means "none".

use super::{js_str, with_host, HistoryOp, PageState};
use crate::host::{self, OutEvent, Task};
use crate::net::{Credentials, FetchRequest, Mode, Redirect};
use blitz_dom::{local_name, ns, BaseDocument, LocalName, NodeData, NodeId, QualName, ScrollBehavior, ScrollLogicalPosition};
use blitz_traits::events::DomEvent;
use boa_engine::object::builtins::{JsArray, JsUint8Array};
use boa_engine::object::ObjectInitializer;
use boa_engine::property::Attribute;
use boa_engine::{Context, JsArgs, JsNativeError, JsResult, JsString, JsValue, NativeFunction};
use url::Url;

type HostFn = fn(&JsValue, &[JsValue], &mut Context) -> JsResult<JsValue>;

pub fn install(ctx: &mut Context) -> JsResult<()> {
    let fns: &[(&str, HostFn, usize)] = &[
        ("identity", identity, 0),
        ("info", info, 1),
        ("parent", parent, 1),
        ("children", children, 1),
        ("documentId", document_id, 0),
        ("text", text, 1),
        ("setText", set_text, 2),
        ("getAttr", get_attr, 2),
        ("setAttr", set_attr, 3),
        ("removeAttr", remove_attr, 2),
        ("attrs", attrs, 1),
        ("create", create, 2),
        ("scriptAsync", script_async, 1),
        ("createText", create_text, 1),
        ("createComment", create_comment, 1),
        ("append", append, 2),
        ("insertBefore", insert_before, 3),
        ("remove", remove, 1),
        ("innerHTML", inner_html, 1),
        ("setInnerHTML", set_inner_html, 2),
        ("outerHTML", outer_html, 1),
        ("clone", clone_node, 2),
        ("query", query, 2),
        ("queryAll", query_all, 2),
        ("matches", matches, 2),
        ("byId", by_id, 1),
        ("rect", rect, 1),
        ("box", box_metrics, 1),
        ("setScroll", set_scroll, 3),
        ("scrollIntoView", scroll_into_view, 2),
        ("computed", computed, 2),
        ("styleSet", style_set, 3),
        ("styleRemove", style_remove, 2),
        ("styleGet", style_get, 2),
        ("value", value, 1),
        ("setValue", set_value, 2),
        ("checked", checked, 1),
        ("setChecked", set_checked, 2),
        ("focus", focus, 1),
        ("blur", blur, 1),
        ("activeElement", active_element, 0),
        ("defaultClick", default_click, 1),
        ("submit", submit, 2),
        ("title", title, 0),
        ("url", page_url, 0),
        ("referrer", referrer, 0),
        ("readyState", ready_state, 0),
        ("cookie", cookie, 0),
        ("setCookie", set_cookie, 1),
        ("navigate", navigate, 2),
        ("history", history, 3),
        ("setTimer", set_timer, 2),
        ("clearTimer", clear_timer, 1),
        ("requestFrame", request_frame, 0),
        ("fetch", fetch, 7),
        ("storage", storage, 4),
        ("log", log, 2),
        ("dialog", dialog, 2),
        ("frameWindow", frame_window, 1),
        ("postMessage", post_message, 3),
        ("now", now, 0),
        ("random", random, 1),
        ("encode", encode, 1),
        ("decode", decode, 2),
        ("parseUrl", parse_url, 2),
        ("setUrlPart", set_url_part, 3),
        ("viewport", viewport, 0),
        ("scrollTo", scroll_to, 2),
        ("listen", listen, 1),
        ("runScript", run_script, 1),
        ("write", write, 1),
        ("importMap", import_map, 1),
        ("supports", supports, 2),
        ("sheets", sheets, 0),
    ];
    let mut obj = ObjectInitializer::new(ctx);
    for (name, f, arity) in fns {
        obj.function(NativeFunction::from_fn_ptr(*f), JsString::from(*name), *arity);
    }
    let obj = obj.build();
    ctx.register_global_property(JsString::from("__h"), obj, Attribute::all())?;
    Ok(())
}

// ---- argument helpers ----

fn identity(_: &JsValue, _: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    Ok(js_str(&with_host(|_, s| s.identity.navigator_json()).unwrap_or_default()))
}

fn arg_num(args: &[JsValue], i: usize) -> f64 {
    args.get(i).and_then(|v| v.as_number()).unwrap_or(f64::NAN)
}

fn arg_id(args: &[JsValue], i: usize) -> JsResult<NodeId> {
    let n = arg_num(args, i);
    if n.is_finite() && n >= 0.0 {
        Ok(NodeId::from_u64(n as u64))
    } else {
        Err(JsNativeError::typ().with_message("Invalid node").into())
    }
}

fn arg_opt_id(args: &[JsValue], i: usize) -> Option<NodeId> {
    let n = arg_num(args, i);
    (n.is_finite() && n >= 0.0).then_some(NodeId::from_u64(n as u64))
}

fn arg_str(args: &[JsValue], i: usize, ctx: &mut Context) -> JsResult<String> {
    Ok(args.get_or_undefined(i).to_string(ctx)?.to_std_string_lossy())
}

fn arg_bool(args: &[JsValue], i: usize) -> bool {
    args.get_or_undefined(i).to_boolean()
}

fn ids(ids: impl IntoIterator<Item = NodeId>, ctx: &mut Context) -> JsValue {
    JsArray::from_iter(ids.into_iter().map(|i| JsValue::from(i.as_u64() as f64)), ctx).into()
}

fn nums(values: &[f64], ctx: &mut Context) -> JsValue {
    JsArray::from_iter(values.iter().map(|v| JsValue::from(*v)), ctx).into()
}

fn opt_id(id: Option<NodeId>) -> JsValue {
    JsValue::from(id.map(|i| i.as_u64() as f64).unwrap_or(-1.0))
}

fn no_doc() -> boa_engine::JsError {
    JsNativeError::error().with_message("The document is not available").into()
}

fn host<R>(f: impl FnOnce(&mut BaseDocument, &mut PageState) -> R) -> JsResult<R> {
    with_host(f).ok_or_else(no_doc)
}

fn node_exists(doc: &BaseDocument, id: NodeId) -> JsResult<()> {
    doc.get_node(id).map(|_| ()).ok_or_else(|| JsNativeError::typ().with_message("The node no longer exists").into())
}

fn mutated(state: &mut PageState) {
    state.layout_dirty = true;
    host::request_redraw();
}

fn ensure_layout(doc: &mut BaseDocument, state: &mut PageState) {
    if state.layout_dirty {
        doc.resolve(state.now / 1000.0);
        state.layout_dirty = false;
    }
}

fn is_dom_child(doc: &BaseDocument, id: NodeId) -> bool {
    doc.get_node(id).is_some_and(|n| !matches!(n.data, NodeData::AnonymousBlock(_)))
}

fn html_name(name: &str) -> QualName {
    QualName::new(None, ns!(html), LocalName::from(name))
}

fn attr_name(name: &str) -> QualName {
    QualName::new(None, ns!(), LocalName::from(name))
}


/// Blitz's form encoder reads `<select>` and `<textarea>` values from their `value` attribute.
/// Mirror the live values there just before a submission.
pub fn prepare_form_submission(doc: &mut BaseDocument, form: NodeId) {
    let controls: Vec<NodeId> = doc.query_selector_all_in(form, "select, textarea").map(|v| v.to_vec()).unwrap_or_default();
    let mut updates = vec![];
    for id in controls {
        let Some(node) = doc.get_node(id) else { continue };
        let Some(el) = node.element_data() else { continue };
        if el.name.local == local_name!("textarea") {
            let text = el.text_input_data().map(|t| t.editor.raw_text().to_string()).unwrap_or_else(|| node.text_content());
            updates.push((id, text));
            continue;
        }
        let options: Vec<NodeId> = doc.query_selector_all_in(id, "option").map(|v| v.to_vec()).unwrap_or_default();
        let chosen = options.iter().find(|o| doc.get_node(**o).is_some_and(|n| n.attr(local_name!("selected")).is_some())).or(options.first());
        if let Some(o) = chosen.and_then(|o| doc.get_node(*o)) {
            let value = o.attr(local_name!("value")).map(str::to_string).unwrap_or_else(|| o.text_content().split_whitespace().collect::<Vec<_>>().join(" "));
            updates.push((id, value));
        }
    }
    let mut m = doc.mutate();
    for (id, value) in updates {
        m.set_attribute(id, attr_name("value"), &value);
    }
}

pub fn form_owner_of(doc: &BaseDocument, node: NodeId) -> Option<NodeId> {
    let n = doc.get_node(node)?;
    if let Some(id) = n.attr(local_name!("form")) {
        return doc.get_element_by_id(id);
    }
    let mut cur = Some(node);
    while let Some(p) = cur {
        let pn = doc.get_node(p)?;
        if pn.data.is_element_with_tag_name(&local_name!("form")) {
            return Some(p);
        }
        cur = pn.parent;
    }
    None
}

// ---- tree ----

/// [nodeType, nodeName, namespace]
fn info(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let (kind, name, nsv) = host(|doc, _| {
        let Some(node) = doc.get_node(id) else { return (0, String::new(), String::new()) };
        match &node.data {
            NodeData::Document(_) => (9, "#document".into(), String::new()),
            NodeData::Element(el) | NodeData::AnonymousBlock(el) => {
                let nsv = if el.name.ns == ns!(svg) { "svg" } else if el.name.ns == ns!(mathml) { "math" } else { "html" };
                (1, el.name.local.to_string(), nsv.to_string())
            }
            NodeData::Text(_) => (3, "#text".into(), String::new()),
            NodeData::Comment { .. } => (8, "#comment".into(), String::new()),
        }
    })?;
    Ok(JsArray::from_iter([JsValue::from(kind), js_str(&name), js_str(&nsv)], ctx).into())
}

fn parent(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    host(|doc, _| opt_id(doc.get_node(id).and_then(|n| n.parent)))
}

fn children(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let list = host(|doc, _| {
        doc.get_node(id).map(|n| n.children.iter().copied().filter(|c| is_dom_child(doc, *c)).collect::<Vec<_>>()).unwrap_or_default()
    })?;
    Ok(ids(list, ctx))
}

fn document_id(_: &JsValue, _: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    host(|doc, _| JsValue::from(doc.root_node().id.as_u64() as f64))
}

fn text(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    host(|doc, _| match doc.get_node(id).map(|n| &n.data) {
        Some(NodeData::Text(t)) => js_str(&t.content),
        Some(NodeData::Comment { contents }) => js_str(contents),
        Some(NodeData::Element(_)) => js_str(&doc.get_node(id).unwrap().text_content()),
        _ => JsValue::null(),
    })
}

fn set_text(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let value = arg_str(args, 1, ctx)?;
    host(|doc, state| {
        node_exists(doc, id)?;
        let is_char_data = matches!(doc.get_node(id).map(|n| &n.data), Some(NodeData::Text(_) | NodeData::Comment { .. }));
        if is_char_data {
            if let Some(NodeData::Comment { contents }) = doc.get_node_mut(id).map(|n| &mut n.data) {
                *contents = value;
            } else {
                doc.mutate().set_node_text(id, &value);
            }
        } else {
            let mut m = doc.mutate();
            // Detach rather than drop: scripts may still hold the old children.
            for child in m.child_ids(id).to_vec() {
                m.remove_node(child);
            }
            if !value.is_empty() {
                let t = m.create_text_node(&value);
                m.append_children(id, &[t]);
            }
        }
        mutated(state);
        Ok(JsValue::undefined())
    })?
}

fn get_attr(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let name = arg_str(args, 1, ctx)?;
    host(|doc, _| {
        doc.get_node(id)
            .and_then(|n| n.element_data())
            .and_then(|el| el.attrs().iter().find(|a| *a.name.local == *name).map(|a| js_str(&a.value)))
            .unwrap_or(JsValue::null())
    })
}

fn set_attr(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let name = arg_str(args, 1, ctx)?;
    let value = arg_str(args, 2, ctx)?;
    if name.is_empty() || name.contains(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == '>' || c == '/' || c == '=') {
        return Err(JsNativeError::error().with_message(format!("'{name}' is not a valid attribute name")).into());
    }
    host(|doc, state| {
        node_exists(doc, id)?;
        doc.mutate().set_attribute(id, attr_name(&name), &value);
        mutated(state);
        Ok(JsValue::undefined())
    })?
}

fn remove_attr(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let name = arg_str(args, 1, ctx)?;
    host(|doc, state| {
        node_exists(doc, id)?;
        doc.mutate().clear_attribute(id, attr_name(&name));
        mutated(state);
        Ok(JsValue::undefined())
    })?
}

/// [name, value, name, value, ...]
fn attrs(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let list = host(|doc, _| {
        doc.get_node(id)
            .and_then(|n| n.element_data())
            .map(|el| el.attrs().iter().flat_map(|a| [a.name.local.to_string(), a.value.to_string()]).collect::<Vec<_>>())
            .unwrap_or_default()
    })?;
    Ok(JsArray::from_iter(list.iter().map(|s| js_str(s)), ctx).into())
}

fn create(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let tag = arg_str(args, 0, ctx)?;
    let nsv = arg_str(args, 1, ctx)?;
    let name = match nsv.as_str() {
        "http://www.w3.org/2000/svg" | "svg" => QualName::new(None, ns!(svg), LocalName::from(tag.as_str())),
        "http://www.w3.org/1998/Math/MathML" | "math" => QualName::new(None, ns!(mathml), LocalName::from(tag.as_str())),
        _ => html_name(&tag.to_ascii_lowercase()),
    };
    host(|doc, state| {
        let is_script = name.ns == ns!(html) && name.local == local_name!("script");
        let id = doc.mutate().create_element(name, vec![]);
        if is_script { state.force_async_scripts.insert(id); }
        JsValue::from(id.as_u64() as f64)
    })
}

fn script_async(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    host(|doc, state| {
        node_exists(doc, id)?;
        if args.len() > 1 {
            if arg_bool(args, 1) { state.force_async_scripts.insert(id); }
            else { state.force_async_scripts.remove(&id); }
        }
        Ok(JsValue::from(state.force_async_scripts.contains(&id)))
    })?
}

fn create_text(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let value = arg_str(args, 0, ctx)?;
    host(|doc, _| JsValue::from(doc.mutate().create_text_node(&value).as_u64() as f64))
}

fn create_comment(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let value = arg_str(args, 0, ctx)?;
    host(|doc, _| JsValue::from(doc.mutate().create_comment_node(&value).as_u64() as f64))
}

fn would_cycle(doc: &BaseDocument, parent: NodeId, child: NodeId) -> bool {
    let mut cur = Some(parent);
    while let Some(id) = cur {
        if id == child {
            return true;
        }
        cur = doc.get_node(id).and_then(|n| n.parent);
    }
    false
}

fn append(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let parent = arg_id(args, 0)?;
    let child = arg_id(args, 1)?;
    host(|doc, state| {
        node_exists(doc, parent)?;
        node_exists(doc, child)?;
        if would_cycle(doc, parent, child) {
            return Err(JsNativeError::error().with_message("HierarchyRequestError: the new child contains the parent").into());
        }
        let mut m = doc.mutate();
        if m.node_has_parent(child) {
            m.remove_node(child);
        }
        m.append_children(parent, &[child]);
        drop(m);
        mutated(state);
        Ok(JsValue::undefined())
    })?
}

fn insert_before(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let parent = arg_id(args, 0)?;
    let child = arg_id(args, 1)?;
    let anchor = arg_opt_id(args, 2);
    host(|doc, state| {
        node_exists(doc, parent)?;
        node_exists(doc, child)?;
        if would_cycle(doc, parent, child) {
            return Err(JsNativeError::error().with_message("HierarchyRequestError: the new child contains the parent").into());
        }
        if anchor == Some(child) {
            return Ok(JsValue::undefined());
        }
        let mut m = doc.mutate();
        if m.node_has_parent(child) {
            m.remove_node(child);
        }
        match anchor {
            Some(a) if m.parent_id(a) == Some(parent) => m.insert_nodes_before(a, &[child]),
            Some(_) => return Err(JsNativeError::error().with_message("NotFoundError: the reference node is not a child of this node").into()),
            None => m.append_children(parent, &[child]),
        }
        drop(m);
        mutated(state);
        Ok(JsValue::undefined())
    })?
}

fn remove(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    host(|doc, state| {
        if doc.get_node(id).is_some_and(|n| n.parent.is_some()) {
            if doc.get_focussed_node_id().is_some_and(|f| f == id || doc.get_node(f).is_some_and(|_| would_cycle(doc, f, id))) {
                doc.clear_focus();
            }
            doc.mutate().remove_node(id);
            mutated(state);
        }
        JsValue::undefined()
    })
}

fn inner_html(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    host(|doc, _| {
        let Some(node) = doc.get_node(id) else { return JsValue::null() };
        let mut out = String::new();
        for c in &node.children {
            if let Some(child) = doc.get_node(*c).filter(|_| is_dom_child(doc, *c)) {
                child.write_outer_html(&mut out);
            }
        }
        js_str(&out)
    })
}

fn set_inner_html(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let html = arg_str(args, 1, ctx)?;
    host(|doc, state| {
        node_exists(doc, id)?;
        let mut m = doc.mutate();
        for child in m.child_ids(id).to_vec() {
            m.remove_node(child);
        }
        m.set_inner_html(id, &html);
        drop(m);
        mutated(state);
        Ok(JsValue::undefined())
    })?
}

fn outer_html(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    host(|doc, _| doc.get_node(id).map(|n| js_str(&n.outer_html())).unwrap_or(JsValue::null()))
}

fn clone_node(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let deep = arg_bool(args, 1);
    host(|doc, _| {
        node_exists(doc, id)?;
        let new_id = doc.mutate().deep_clone_node(id);
        if !deep {
            doc.mutate().remove_and_drop_all_children(new_id);
        }
        Ok(JsValue::from(new_id.as_u64() as f64))
    })?
}

fn selector_error(sel: &str) -> boa_engine::JsError {
    JsNativeError::syntax().with_message(format!("SyntaxError: '{sel}' is not a valid selector")).into()
}

fn query(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let root = arg_id(args, 0)?;
    let sel = arg_str(args, 1, ctx)?;
    host(|doc, _| {
        let found = if root == doc.root_node().id { doc.query_selector(&sel) } else { doc.query_selector_in(root, &sel) };
        found.map(opt_id).map_err(|_| selector_error(&sel))
    })?
}

fn query_all(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let root = arg_id(args, 0)?;
    let sel = arg_str(args, 1, ctx)?;
    let found = host(|doc, _| {
        let found = if root == doc.root_node().id { doc.query_selector_all(&sel) } else { doc.query_selector_all_in(root, &sel) };
        found.map(|v| v.to_vec()).map_err(|_| selector_error(&sel))
    })??;
    Ok(ids(found, ctx))
}

fn matches(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let sel = arg_str(args, 1, ctx)?;
    host(|doc, _| doc.matches_selector(id, &sel).map(JsValue::from).map_err(|_| selector_error(&sel)))?
}

fn by_id(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_str(args, 0, ctx)?;
    host(|doc, _| opt_id(doc.get_element_by_id(&id)))
}

// ---- layout ----

fn rect(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let r = host(|doc, state| {
        ensure_layout(doc, state);
        doc.get_client_bounding_rect(id).map(|r| [r.x, r.y, r.width, r.height]).unwrap_or([0.0; 4])
    })?;
    Ok(nums(&r, ctx))
}

/// [offsetTop, offsetLeft, offsetWidth, offsetHeight, offsetParent, clientWidth, clientHeight,
///  scrollWidth, scrollHeight, scrollTop, scrollLeft]
fn box_metrics(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let v = host(|doc, state| {
        ensure_layout(doc, state);
        let Some(node) = doc.get_node(id) else { return [0.0; 11] };
        let rect = doc.offset_rect(id);
        let parent = node.offset_parent().map(|p| p.id.as_u64() as f64).unwrap_or(-1.0);
        let scroll = node.scroll_offset();
        [
            rect.map(|r| r.y).unwrap_or(0.0),
            rect.map(|r| r.x).unwrap_or(0.0),
            rect.map(|r| r.width).unwrap_or(0.0),
            rect.map(|r| r.height).unwrap_or(0.0),
            parent,
            node.client_width() as f64,
            node.client_height() as f64,
            node.scroll_width() as f64,
            node.scroll_height() as f64,
            scroll.y,
            scroll.x,
        ]
    })?;
    Ok(nums(&v, ctx))
}

fn set_scroll(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let top = arg_num(args, 1);
    let left = arg_num(args, 2);
    host(|doc, state| {
        ensure_layout(doc, state);
        let cur = doc.get_node(id).map(|n| *n.scroll_offset()).unwrap_or(blitz_dom::Point { x: 0.0, y: 0.0 });
        let x = if left.is_finite() { left } else { cur.x };
        let y = if top.is_finite() { top } else { cur.y };
        doc.scroll_to(id, x, y, ScrollBehavior::Instant);
        host::request_redraw();
        JsValue::undefined()
    })
}

fn scroll_into_view(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let block = arg_str(args, 1, ctx)?;
    let pos = match block.as_str() {
        "center" => ScrollLogicalPosition::Center,
        "end" => ScrollLogicalPosition::End,
        "nearest" => ScrollLogicalPosition::Nearest,
        _ => ScrollLogicalPosition::Start,
    };
    host(|doc, state| {
        ensure_layout(doc, state);
        doc.scroll_into_view(id, ScrollBehavior::Instant, pos, ScrollLogicalPosition::Nearest);
        host::request_redraw();
        JsValue::undefined()
    })
}

fn computed(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let prop = arg_str(args, 1, ctx)?;
    host(|doc, state| {
        ensure_layout(doc, state);
        if doc.get_node(id).and_then(|n| n.element_data()).is_none() {
            return js_str("");
        }
        js_str(&doc.resolved_style_value(id, &prop))
    })
}

fn style_set(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let prop = arg_str(args, 1, ctx)?;
    let raw = arg_str(args, 2, ctx)?;
    let (value, important) = match raw.trim_end().strip_suffix("!important") {
        Some(v) => (v.trim_end().to_string(), true),
        None => (raw.clone(), false),
    };
    host(|doc, state| {
        let Some(cur) = doc.get_node(id).and_then(|n| n.element_data()).map(|el| el.attr(local_name!("style")).unwrap_or("").to_string()) else {
            return JsValue::undefined();
        };
        if let Some(next) = doc.style_attr_set_property(&cur, &prop, &value, important) {
            if next != cur {
                doc.mutate().set_attribute(id, attr_name("style"), &next);
                mutated(state);
            }
        }
        JsValue::undefined()
    })
}

fn style_remove(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let prop = arg_str(args, 1, ctx)?;
    host(|doc, state| {
        let Some(cur) = doc.get_node(id).and_then(|n| n.element_data()).and_then(|el| el.attr(local_name!("style")).map(str::to_string)) else {
            return JsValue::undefined();
        };
        if let Some((next, _)) = doc.style_attr_remove_property(&cur, &prop) {
            doc.mutate().set_attribute(id, attr_name("style"), &next);
            mutated(state);
        }
        JsValue::undefined()
    })
}

/// Reads one inline style property (or the whole serialized declaration block when `prop` is "").
fn style_get(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let prop = arg_str(args, 1, ctx)?;
    host(|doc, _| {
        let style = doc.get_node(id).and_then(|n| n.attr(local_name!("style"))).unwrap_or("").to_string();
        if prop.is_empty() {
            js_str(&doc.style_attr_serialize(&style))
        } else {
            js_str(&doc.style_attr_get_property(&style, &prop))
        }
    })
}

// ---- forms and focus ----

fn value(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    host(|doc, _| {
        let Some(el) = doc.get_node(id).and_then(|n| n.element_data()) else { return JsValue::null() };
        if let Some(input) = el.text_input_data() {
            return js_str(input.editor.raw_text());
        }
        el.attr(local_name!("value")).map(js_str).unwrap_or(JsValue::null())
    })
}

fn set_value(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let v = arg_str(args, 1, ctx)?;
    host(|doc, state| {
        let has_editor = doc.get_node(id).and_then(|n| n.element_data()).is_some_and(|el| el.text_input_data().is_some());
        if has_editor {
            doc.with_text_input(id, |mut driver| {
                driver.select_all();
                if v.is_empty() {
                    driver.delete_selection();
                } else {
                    driver.insert_or_replace_selection(&v);
                }
            });
            if let Some(node) = doc.get_node_mut(id) {
                node.mark_damaged();
            }
        } else {
            doc.mutate().set_attribute(id, attr_name("value"), &v);
        }
        mutated(state);
        JsValue::undefined()
    })
}

fn checked(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    host(|doc, _| {
        doc.get_node(id)
            .and_then(|n| n.element_data())
            .and_then(|el| el.checkbox_input_checked())
            .map(JsValue::from)
            .unwrap_or(JsValue::from(false))
    })
}

fn set_checked(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    let on = arg_bool(args, 1);
    host(|doc, state| {
        let radio = doc.get_node(id).and_then(|n| n.element_data()).filter(|el| el.attr(local_name!("type")) == Some("radio")).map(|el| el.attr(local_name!("name")).unwrap_or("").to_string());
        match radio {
            Some(name) if on => doc.toggle_radio(name, id),
            _ => {
                if let Some(el) = doc.get_node_mut(id).and_then(|n| n.element_data_mut()) {
                    el.set_checkbox_input_checked(on);
                }
            }
        }
        if let Some(node) = doc.get_node_mut(id) {
            node.mark_damaged();
        }
        mutated(state);
        JsValue::undefined()
    })
}

fn focus(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    host(|doc, _| {
        let ok = doc.get_node(id).is_some_and(|n| n.is_focussable()) && doc.set_focus_to(id);
        host::request_redraw();
        JsValue::from(ok)
    })
}

fn blur(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    host(|doc, _| {
        if doc.get_focussed_node_id() == Some(id) {
            doc.clear_focus();
            host::request_redraw();
        }
        JsValue::undefined()
    })
}

fn active_element(_: &JsValue, _: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    host(|doc, _| opt_id(doc.get_focussed_node_id()))
}

/// Run the click default action (links, submit buttons, checkboxes, labels) for `el.click()`.
fn default_click(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    host(|doc, state| {
        if let Some(form) = form_owner_of(doc, id) {
            prepare_form_submission(doc, form);
        }
        let Some(node) = doc.get_node(id) else { return JsValue::undefined() };
        let data = node.synthetic_click_event(keyboard_types::Modifiers::empty());
        let mut event = DomEvent::new(id, data);
        let mut generated = vec![];
        doc.handle_dom_event(&mut event, |e| generated.push(e));
        state.pending_events.extend(generated);
        mutated(state);
        JsValue::undefined()
    })
}

fn submit(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let form = arg_id(args, 0)?;
    let submitter = arg_opt_id(args, 1).unwrap_or(form);
    host(|doc, _| {
        prepare_form_submission(doc, form);
        doc.submit_form(form, submitter);
        JsValue::undefined()
    })
}

// ---- document and page ----

fn title(_: &JsValue, _: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    host(|doc, _| js_str(doc.find_title_node().map(|n| n.text_content()).unwrap_or_default().trim()))
}

fn page_url(_: &JsValue, _: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    host(|_, state| js_str(state.url.as_str()))
}

fn referrer(_: &JsValue, _: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    host(|_, state| js_str(&state.referrer))
}

fn ready_state(_: &JsValue, _: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    host(|_, state| js_str(state.ready_state))
}

fn cookie(_: &JsValue, _: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let url = host(|_, state| state.url.clone())?;
    Ok(js_str(&host::loader().script_cookies(&url)))
}

fn set_cookie(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let value = arg_str(args, 0, ctx)?;
    let url = host(|_, state| state.url.clone())?;
    host::loader().set_script_cookie(&url, &value);
    host::emit(OutEvent::Cookies);
    Ok(JsValue::undefined())
}

fn navigate(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let url = arg_str(args, 0, ctx)?;
    let replace = arg_bool(args, 1);
    host(|_, state| {
        state.navigations.push((url, replace));
        host::wake();
        JsValue::undefined()
    })
}

/// history(op, url, state): op 0 = push, 1 = replace, 2 = go(url as delta)
fn history(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let op = arg_num(args, 0) as i32;
    let url = arg_str(args, 1, ctx)?;
    let data = arg_str(args, 2, ctx)?;
    host(|_, state| {
        if op == 2 {
            state.history.push(HistoryOp::Go(url.parse().unwrap_or(0)));
            host::wake();
            return Ok(JsValue::undefined());
        }
        let next = state.url.join(&url).map_err(|_| JsNativeError::typ().with_message("SecurityError: invalid history URL"))?;
        if next.origin() != state.url.origin() {
            return Err(JsNativeError::typ().with_message("SecurityError: history URLs must be same-origin").into());
        }
        state.url = next.clone();
        state.history.push(HistoryOp::Push { url: next.to_string(), replace: op == 1, state: data });
        host::wake();
        Ok(JsValue::undefined())
    })?
}

fn set_timer(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_num(args, 0) as u32;
    let ms = arg_num(args, 1);
    let ms = if ms.is_finite() { ms.clamp(0.0, 2_147_483_647.0) } else { 0.0 };
    host(|_, state| {
        state.timers.insert(id, state.now + ms);
        host::wake();
        JsValue::undefined()
    })
}

fn clear_timer(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_num(args, 0) as u32;
    host(|_, state| {
        state.timers.remove(&id);
        JsValue::undefined()
    })
}

fn request_frame(_: &JsValue, _: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    host(|_, state| {
        state.animation_frame = true;
        host::wake();
        JsValue::undefined()
    })
}

/// fetch(id, method, url, headers [[k, v], ...], body (string | Uint8Array | null), credentials, redirect)
fn fetch(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_num(args, 0) as u32;
    let method = arg_str(args, 1, ctx)?.to_ascii_uppercase();
    let url = arg_str(args, 2, ctx)?;
    let mut headers = Vec::new();
    if let Some(list) = args.get_or_undefined(3).as_object() {
        let list = JsArray::from_object(list)?;
        for i in 0..list.length(ctx)? {
            let pair = list.get(i, ctx)?;
            if let Some(pair) = pair.as_object() {
                let pair = JsArray::from_object(pair)?;
                let k = pair.get(0, ctx)?.to_string(ctx)?.to_std_string_lossy();
                let v = pair.get(1, ctx)?.to_string(ctx)?.to_std_string_lossy();
                headers.push((k, v));
            }
        }
    }
    let body_arg = args.get_or_undefined(4);
    let body = if body_arg.is_null_or_undefined() {
        vec![]
    } else if let Some(obj) = body_arg.as_object().filter(|o| o.is::<boa_engine::builtins::typed_array::TypedArray>()) {
        JsUint8Array::from_object(obj)?.iter(ctx).collect()
    } else {
        body_arg.to_string(ctx)?.to_std_string_lossy().into_bytes()
    };
    let credentials = match arg_str(args, 5, ctx)?.as_str() {
        "omit" => Credentials::Omit,
        "include" => Credentials::Include,
        _ => Credentials::SameOrigin,
    };
    let redirect = match arg_str(args, 6, ctx)?.as_str() {
        "manual" => Redirect::Manual,
        "error" => Redirect::Error,
        _ => Redirect::Follow,
    };
    let (page, base) = host(|_, state| (state.id, state.url.clone()))?;
    let target = base.join(&url).map_err(|_| JsNativeError::typ().with_message(format!("Failed to fetch: invalid URL {url}")))?;
    let req = FetchRequest {
        url: target,
        method,
        headers: headers.into_iter().filter(|(k, _)| !crate::net::policy::forbidden_request_header(k)).collect(),
        body,
        initiator: Some(base),
        mode: Mode::Cors,
        credentials,
        redirect,
        dest: "empty",
        user_activated: false,
    };
    host::start_fetch(req, move |result| Task::JsFetch { page, id, result });
    Ok(JsValue::undefined())
}

/// storage(kind 0 local / 1 session, op, key, value)
fn storage(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let kind = arg_num(args, 0) as i32;
    let op = arg_str(args, 1, ctx)?;
    let key = arg_str(args, 2, ctx)?;
    let value = arg_str(args, 3, ctx)?;
    let (map, keys) = host(|_, state| {
        let map = if kind == 0 { state.local.clone() } else { state.session.clone() };
        let mut m = map.borrow_mut();
        let out = match op.as_str() {
            "get" => m.get(&key).map(|v| js_str(v)).unwrap_or(JsValue::null()),
            "set" => {
                let total: usize = m.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>() + key.len() + value.len();
                if total > 5 * 1024 * 1024 {
                    return Err(boa_engine::JsError::from(JsNativeError::error().with_message("QuotaExceededError: storage is full")));
                }
                m.insert(key.clone(), value.clone());
                state.storage_changed = true;
                JsValue::undefined()
            }
            "remove" => {
                m.remove(&key);
                state.storage_changed = true;
                JsValue::undefined()
            }
            "clear" => {
                m.clear();
                state.storage_changed = true;
                JsValue::undefined()
            }
            "length" => JsValue::from(m.len() as f64),
            _ => JsValue::undefined(),
        };
        let keys: Option<Vec<String>> = (op == "keys").then(|| {
            let mut k: Vec<String> = m.keys().cloned().collect();
            k.sort();
            k
        });
        Ok((out, keys))
    })??;
    if let Some(keys) = keys {
        return Ok(JsArray::from_iter(keys.iter().map(|k| js_str(k)), ctx).into());
    }
    if op == "set" || op == "remove" || op == "clear" {
        host::emit(OutEvent::Storage);
    }
    Ok(map)
}

fn log(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let level = arg_str(args, 0, ctx)?;
    let mut text = arg_str(args, 1, ctx)?;
    if text.len() > 2000 {
        let mut cut = 2000;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        text.truncate(cut);
        text.push('…');
    }
    let allowed = with_host(|_, state| {
        if state.console_budget == 0 {
            return false;
        }
        state.console_budget -= 1;
        true
    })
    .unwrap_or(true);
    if allowed {
        host::emit(OutEvent::Console { level, text });
    }
    Ok(JsValue::undefined())
}

fn dialog(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let kind = arg_str(args, 0, ctx)?;
    let text = arg_str(args, 1, ctx)?;
    host::emit(OutEvent::Dialog { kind, text });
    Ok(JsValue::undefined())
}

fn now(_: &JsValue, _: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    Ok(JsValue::from(crate::clock::now()))
}

fn random(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let n = (arg_num(args, 0).max(0.0) as usize).min(65536);
    let mut buf = vec![0u8; n];
    getrandom::fill(&mut buf).map_err(|e| JsNativeError::error().with_message(e.to_string()))?;
    Ok(JsUint8Array::from_iter(buf, ctx)?.into())
}

fn encode(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let s = arg_str(args, 0, ctx)?;
    Ok(JsUint8Array::from_iter(s.into_bytes(), ctx)?.into())
}

fn decode(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let bytes: Vec<u8> = match args.get_or_undefined(0).as_object() {
        Some(obj) => JsUint8Array::from_object(obj)?.iter(ctx).collect(),
        None => vec![],
    };
    let label = arg_str(args, 1, ctx)?;
    let encoding = encoding_rs::Encoding::for_label(label.as_bytes()).unwrap_or(encoding_rs::UTF_8);
    let (text, _, _) = encoding.decode(&bytes);
    Ok(js_str(&text))
}

/// [href, protocol, username, password, host, hostname, port, pathname, search, hash, origin]
fn parse_url(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let href = arg_str(args, 0, ctx)?;
    let base = args.get_or_undefined(1);
    let parsed = if base.is_null_or_undefined() {
        Url::parse(&href)
    } else {
        let base = base.to_string(ctx)?.to_std_string_lossy();
        Url::parse(&base).and_then(|b| b.join(&href))
    };
    let Ok(u) = parsed else { return Ok(JsValue::null()) };
    Ok(url_parts(&u, ctx))
}

fn url_parts(u: &Url, ctx: &mut Context) -> JsValue {
    let host = match (u.host_str(), u.port()) {
        (Some(h), Some(p)) => format!("{h}:{p}"),
        (Some(h), None) => h.to_string(),
        _ => String::new(),
    };
    let parts = [
        u.as_str().to_string(),
        format!("{}:", u.scheme()),
        u.username().to_string(),
        u.password().unwrap_or("").to_string(),
        host,
        u.host_str().unwrap_or("").to_string(),
        u.port().map(|p| p.to_string()).unwrap_or_default(),
        u.path().to_string(),
        u.query().filter(|q| !q.is_empty()).map(|q| format!("?{q}")).unwrap_or_default(),
        u.fragment().filter(|f| !f.is_empty()).map(|f| format!("#{f}")).unwrap_or_default(),
        u.origin().ascii_serialization(),
    ];
    JsArray::from_iter(parts.iter().map(|p| js_str(p)), ctx).into()
}

fn set_url_part(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let href = arg_str(args, 0, ctx)?;
    let part = arg_str(args, 1, ctx)?;
    let value = arg_str(args, 2, ctx)?;
    let Ok(mut u) = Url::parse(&href) else { return Ok(JsValue::null()) };
    let _ = match part.as_str() {
        "protocol" => u.set_scheme(value.trim_end_matches(':')).map_err(|_| ()),
        "username" => u.set_username(&value),
        "password" => u.set_password(Some(&value)),
        "host" => u.set_host(Some(value.split(':').next().unwrap_or(""))).map_err(|_| ()).and_then(|_| {
            let port = value.split(':').nth(1).and_then(|p| p.parse().ok());
            u.set_port(port)
        }),
        "hostname" => u.set_host(Some(&value)).map_err(|_| ()),
        "port" => u.set_port(value.parse().ok()),
        "pathname" => {
            u.set_path(&value);
            Ok(())
        }
        "search" => {
            let q = value.trim_start_matches('?');
            u.set_query((!q.is_empty()).then_some(q));
            Ok(())
        }
        "hash" => {
            let f = value.trim_start_matches('#');
            u.set_fragment((!f.is_empty()).then_some(f));
            Ok(())
        }
        "href" => {
            if let Ok(n) = Url::parse(&value) {
                u = n;
            }
            Ok(())
        }
        _ => Ok(()),
    };
    Ok(url_parts(&u, ctx))
}

/// [innerWidth, innerHeight, scrollX, scrollY, devicePixelRatio]
fn viewport(_: &JsValue, _: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let v = host(|doc, state| {
        let s = doc.viewport_scroll();
        [state.viewport.0 as f64, state.viewport.1 as f64, s.x, s.y, state.viewport.2 as f64]
    })?;
    Ok(nums(&v, ctx))
}

fn scroll_to(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let x = arg_num(args, 0);
    let y = arg_num(args, 1);
    host(|doc, state| {
        ensure_layout(doc, state);
        let cur = doc.viewport_scroll();
        let x = if x.is_finite() { x.max(0.0) } else { cur.x };
        let y = if y.is_finite() { y.max(0.0) } else { cur.y };
        doc.set_viewport_scroll(blitz_dom::Point { x, y });
        host::request_redraw();
        JsValue::undefined()
    })
}

fn listen(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let t = arg_str(args, 0, ctx)?;
    host(|_, state| {
        state.listened.insert(t);
        JsValue::undefined()
    })
}

fn run_script(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    let id = arg_id(args, 0)?;
    host(|_, state| {
        state.dynamic_scripts.push(id);
        host::wake();
        JsValue::undefined()
    })
}

/// document.write while the page is loading: buffered and inserted after the running script.
fn write(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let html = arg_str(args, 0, ctx)?;
    host(|_, state| {
        state.write_buffer.get_or_insert_with(String::new).push_str(&html);
        JsValue::undefined()
    })
}

fn import_map(_: &JsValue, args: &[JsValue], _: &mut Context) -> JsResult<JsValue> {
    // The engine reads import maps directly from the page; kept for scripts that inject one.
    let _ = args;
    Ok(JsValue::undefined())
}

fn supports(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let prop = arg_str(args, 0, ctx)?;
    let value = arg_str(args, 1, ctx)?;
    host(|doc, _| {
        if value.is_empty() {
            JsValue::from(doc.css_supports_condition(&prop))
        } else {
            JsValue::from(doc.css_declaration_is_valid(&prop, &value))
        }
    })
}

fn sheets(_: &JsValue, _: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let n = host(|doc, _| doc.author_stylesheets().count())?;
    Ok(JsValue::from(n as f64)).map(|v| {
        let _ = ctx;
        v
    })
}

/// `__h.frameWindow(nodeId)` — the window object for an iframe node, or null. Backs
/// `iframe.contentWindow`, so a parent can hold the same object a message carries as `source`.
fn frame_window(_: &JsValue, args: &[JsValue], _ctx: &mut Context) -> JsResult<JsValue> {
    let id = arg_num(args, 0);
    if !id.is_finite() || id < 0.0 {
        return Ok(JsValue::null());
    }
    Ok(super::frame_window(id as u64).map(JsValue::from).unwrap_or(JsValue::null()))
}

/// `__h.postMessage(targetWindow, data, targetOrigin)` — delivers a message to another frame
/// (or the top window) with the posting frame as `event.source` and its origin as `event.origin`,
/// honouring `targetOrigin` ("*" or an exact origin match). The posting frame is whichever one is
/// currently executing, tracked natively so identity holds across realms.
fn post_message(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let Some(target) = args.get_or_undefined(0).as_object() else {
        return Err(JsNativeError::typ().with_message("postMessage target is not a window").into());
    };
    let data = args.get_or_undefined(1).clone();
    let target_origin = arg_str(args, 2, ctx)?;
    // Which window is the target (0 = top), from its hidden frame id.
    let target_id = target.get(JsString::from("__frameId"), ctx)?.as_number().unwrap_or(f64::NAN);
    if !target_id.is_finite() || target_id < 0.0 {
        return Ok(JsValue::undefined());
    }
    let target_id = target_id as u64;
    let Some(target_actual_origin) = super::frame_origin(target_id) else {
        return Ok(JsValue::undefined());
    };
    let (source_window, source_origin) = super::frame_source().map(|(w, o)| (JsValue::from(w), o)).unwrap_or((JsValue::null(), String::new()));
    // targetOrigin gate: "*" matches anything, "/" means the sender's own origin, else exact.
    let allowed = match target_origin.as_str() {
        "*" => true,
        "/" => source_origin == target_actual_origin,
        other => other == target_actual_origin,
    };
    if !allowed {
        return Ok(JsValue::undefined());
    }
    let deliver = target.get(JsString::from("__deliverMessage"), ctx)?;
    let Some(deliver) = deliver.as_callable() else {
        return Ok(JsValue::undefined());
    };
    let target_val = JsValue::from(target);
    let origin_val = js_str(&source_origin);
    // The target's handlers run as the target window, so a reply they post has the right source.
    super::frame_with_current(target_id, || deliver.call(&target_val, &[data, origin_val, source_window], ctx))?;
    Ok(JsValue::undefined())
}
