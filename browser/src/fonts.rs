//! Bundled fonts. WebAssembly has no system fonts, so the engine ships the Liberation families
//! (metric-compatible with Arial, Times New Roman and Courier New, SIL OFL 1.1) and DejaVu Sans
//! for wider Unicode coverage, and registers them under the names sites commonly request.

use parley::fontique::{Blob, Collection, CollectionOptions, FamilyId, FontInfoOverride, GenericFamily, SourceCache};
use parley::FontContext;
use std::sync::Arc;

const SANS: &[&str] = &[
    "Arial", "Helvetica", "Helvetica Neue", "Segoe UI", "Roboto", "system-ui", "-apple-system", "BlinkMacSystemFont",
    "Verdana", "Tahoma", "Open Sans", "Inter", "Noto Sans", "Ubuntu", "Cantarell", "Lucida Grande", "Trebuchet MS",
    "Geneva", "San Francisco", "SF Pro Text", "SF Pro Display", "Fira Sans", "Lato", "Source Sans Pro", "Montserrat",
];
const SERIF: &[&str] = &["Times", "Times New Roman", "Georgia", "Cambria", "Noto Serif", "Garamond", "Palatino", "Book Antiqua", "Baskerville"];
const MONO: &[&str] = &[
    "Courier", "Courier New", "Menlo", "Monaco", "Consolas", "SF Mono", "SFMono-Regular", "Source Code Pro", "Roboto Mono",
    "DejaVu Sans Mono", "Ubuntu Mono", "Lucida Console", "ui-monospace", "Liberation Mono", "Fira Code", "JetBrains Mono",
];

/// Build a font context from `(file name, bytes)` pairs.
pub fn font_context(files: &[(String, Arc<Vec<u8>>)]) -> FontContext {
    let mut collection = Collection::new(CollectionOptions { shared: false, system_fonts: false });
    let mut sans = None;
    let mut serif = None;
    let mut mono = None;
    let mut fallbacks: Vec<FamilyId> = vec![];
    for (name, bytes) in files {
        let blob = || Blob::new(bytes.clone() as Arc<dyn AsRef<[u8]> + Send + Sync>);
        let registered = collection.register_fonts(blob(), None);
        let family = registered.first().map(|(id, _)| *id);
        let aliases: &[&str] = if name.starts_with("LiberationSans") {
            sans = sans.or(family);
            SANS
        } else if name.starts_with("LiberationSerif") {
            serif = serif.or(family);
            SERIF
        } else if name.starts_with("LiberationMono") {
            mono = mono.or(family);
            MONO
        } else {
            // DejaVu Sans (symbols, Greek, Cyrillic, Arabic, Hebrew) before Droid Sans Fallback (CJK)
            if let Some(f) = family {
                if !fallbacks.contains(&f) {
                    if name.starts_with("Droid") {
                        fallbacks.push(f);
                    } else {
                        fallbacks.insert(0, f);
                    }
                }
            }
            &[]
        };
        for alias in aliases {
            collection.register_fonts(blob(), Some(FontInfoOverride { family_name: Some(alias), ..Default::default() }));
        }
    }
    let list = |primary: Option<FamilyId>| primary.into_iter().chain(fallbacks.iter().copied().filter(move |f| Some(*f) != primary)).collect::<Vec<_>>();
    let sans_list = list(sans.or(fallbacks.first().copied()));
    for generic in [GenericFamily::SansSerif, GenericFamily::SystemUi, GenericFamily::UiSansSerif, GenericFamily::UiRounded, GenericFamily::Cursive, GenericFamily::Fantasy, GenericFamily::Emoji, GenericFamily::Math] {
        collection.set_generic_families(generic, sans_list.iter().copied());
    }
    for generic in [GenericFamily::Serif, GenericFamily::UiSerif, GenericFamily::FangSong] {
        collection.set_generic_families(generic, list(serif.or(sans)).into_iter());
    }
    for generic in [GenericFamily::Monospace, GenericFamily::UiMonospace] {
        collection.set_generic_families(generic, list(mono.or(sans)).into_iter());
    }
    collection.register_fonts(Blob::new(Arc::new(blitz_dom::BULLET_FONT) as _), None);
    FontContext { collection, source_cache: SourceCache::default() }
}

/// Read the bundled fonts from a directory (native builds and tests).
#[cfg(not(target_arch = "wasm32"))]
pub fn load_dir(dir: &std::path::Path) -> Vec<(String, Arc<Vec<u8>>)> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "ttf" || x == "otf"))
        .filter_map(|e| Some((e.file_name().to_string_lossy().into_owned(), Arc::new(std::fs::read(e.path()).ok()?))))
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}
