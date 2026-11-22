//! Parse with scripting enabled: <noscript> markup must stay inert when Boa runs scripts.
//! Blitz's default parser uses scripting_enabled=false for its script-free applications.

use blitz_dom::{BaseDocument, Document, DocumentConfig, DocumentMutator, HtmlParserProvider, NodeId};
use blitz_html::{DocumentHtmlParser, HtmlDocument};
use html5ever::{tendril::TendrilSink, tree_builder::TreeBuilderOpts, ParseOpts};

fn options() -> ParseOpts {
    ParseOpts {
        tree_builder: TreeBuilderOpts { scripting_enabled: true, drop_doctype: true, ..Default::default() },
        ..Default::default()
    }
}

pub fn document(html: &str, config: DocumentConfig) -> BaseDocument {
    // Preserve Blitz's XML path for documents explicitly identifying themselves as XHTML.
    let root = html.trim_start_matches('\u{feff}').trim_start();
    let xhtml = root.starts_with("<?xml")
        || root.starts_with("<!DOCTYPE") && root.lines().next().is_some_and(|s| s.contains("XHTML") || s.contains("xhtml"))
        || root.starts_with("<html") && root.split('>').next().is_some_and(|s| s.contains("xmlns=\"http://www.w3.org/1999/xhtml\"") || s.contains("xmlns='http://www.w3.org/1999/xhtml'"));
    if xhtml { return HtmlDocument::from_html(html, config).into_inner(); }
    let mut doc = BaseDocument::new(config);
    {
        let mut mutr = doc.mutate();
        html5ever::parse_document(DocumentHtmlParser::new(&mut mutr), options()).one(html);
    }
    doc
}

pub struct ScriptHtmlProvider;
impl HtmlParserProvider for ScriptHtmlProvider {
    fn parse_document(&self, html: &str, config: DocumentConfig) -> Box<dyn Document> {
        Box::new(document(html, config))
    }
    fn parse_inner_html<'m, 'd>(&self, mutr: &'m mut DocumentMutator<'d>, element_id: NodeId, html: &str) {
        let document_id = mutr.doc.root_node().id;
        html5ever::driver::parse_fragment_for_element(DocumentHtmlParser::new(mutr), options(), element_id, false, None).one(html);
        let fragment = mutr.last_child_id(document_id).expect("HTML fragment root");
        let children = mutr.child_ids(fragment);
        mutr.append_children(element_id, &children);
        mutr.remove_and_drop_node(fragment);
    }
}
