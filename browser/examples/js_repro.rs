use evolve_browser::testing::Harness;
fn main() {
    let mut t = Harness::new();
    t.site.add("https://a.test/", "text/html", "<p>x</p>");
    t.open("https://a.test/");
    for code in std::env::args().skip(1) {
        println!("{code}\n  => {}", t.eval(&code));
    }
}
