fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("TARGET").as_deref() == Ok("wasm32-unknown-unknown") {
        // Boa's recursive parser needs more than Rust's default 1 MiB WASM stack for
        // deeply nested, valid page scripts. This changes the WASM linear-memory stack;
        // native thread stacks are unchanged.
        println!("cargo:rustc-link-arg=-zstack-size=16777216");
    }
}
