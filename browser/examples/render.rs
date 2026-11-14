//! Render a saved page natively: `cargo run --example render -- <base-url> <dir> <out.png> [width height]`.
//! Every file in <dir> is served at <base-url><file name>; index.html at <base-url>.

use evolve_browser::testing::Harness;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (base, dir, out) = (&args[1], std::path::Path::new(&args[2]), &args[3]);
    let w: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let h: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(768);
    let mut t = Harness::new();
    t.engine.resize(w, h, 1.0);
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let body = std::fs::read(entry.path()).unwrap();
        let ct = match name.rsplit('.').next().unwrap_or("") {
            "html" => "text/html",
            "css" => "text/css",
            "js" => "text/javascript",
            "svg" => "image/svg+xml",
            "gif" => "image/gif",
            "png" => "image/png",
            _ => "application/octet-stream",
        };
        let url = if name == "index.html" { base.clone() } else { format!("{base}{name}") };
        t.site.pages.borrow_mut().insert(url, (200, ct.into(), body, vec![]));
    }
    // Versioned asset URLs (?v=...) resolve to the same files
    let html = String::from_utf8_lossy(&t.site.pages.borrow()[base.as_str()].2).into_owned();
    for cap in html.split(['"', '\'']).filter(|s| s.contains('?') && !s.contains(' ') && !s.starts_with("http")) {
        let file = cap.split('?').next().unwrap();
        let found = t.site.pages.borrow().get(&format!("{base}{file}")).cloned();
        if let Some(p) = found {
            t.site.pages.borrow_mut().insert(format!("{base}{cap}"), p);
        }
    }
    let start = std::time::Instant::now();
    t.open(base);
    t.engine.resize(w, h, 1.0);
    t.settle();
    eprintln!("loaded in {:?}", start.elapsed());
    for line in t.console() {
        eprintln!("{line}");
    }
    let frame = t.engine.take_frame().map(|f| f.to_vec()).unwrap_or_default();
    let (fw, fh) = t.engine.frame_size();
    let file = std::fs::File::create(out).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), fw, fh);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&frame).unwrap();
    let snap = t.engine.snapshot(1500);
    eprintln!("--- snapshot ---\n{}", snap["text"].as_str().unwrap_or(""));
}
