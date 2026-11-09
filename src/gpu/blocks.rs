//! WGSL decoders for weights kept in their GGUF block layout (see `quant::to_gpu`).
//!
//! Each format defines `grp(b, j)`: the `j`-th run of eight values of the GPU block that starts
//! at word `b`, as codes plus an affine factor, `value = code * s + m`. Matrix kernels build on
//! it, so a format needs nothing else. The decoding follows ggml's reference dequantizers.

use crate::quant::{self, gpu_block};

/// Byte, half and bit-field helpers over the weight words, and the code-book table binding.
const HELPERS: &str = r#"
struct G { a: vec4<f32>, b: vec4<f32>, s: f32, m: f32 }
fn wrd(b: u32, o: u32) -> u32 { return wcodes[b + (o >> 2u)]; }
fn byt(b: u32, o: u32) -> u32 { return (wcodes[b + (o >> 2u)] >> ((o & 3u) * 8u)) & 255u; }
fn u16v(b: u32, o: u32) -> u32 { return (wcodes[b + (o >> 2u)] >> ((o & 2u) * 8u)) & 0xffffu; }
fn hlf(b: u32, o: u32) -> f32 { return unpack2x16float(u16v(b, o)).x; }
// The field of `mask` at bit `sh` of each byte of `w`.
fn fld(w: u32, sh: u32, mask: u32) -> vec4<f32> {
    return vec4<f32>((vec4<u32>(w, w >> 8u, w >> 16u, w >> 24u) >> vec4<u32>(sh)) & vec4<u32>(mask));
}
fn sb4(w: u32) -> vec4<f32> {
    let v = vec4<i32>(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w));
    return vec4<f32>(v >> vec4<u32>(24u));
}
// +1 or -1 for bits sh..sh+3 of `s`.
fn sgn(s: u32, sh: u32) -> vec4<f32> {
    return vec4<f32>(1.0) - 2.0 * vec4<f32>((vec4<u32>(s) >> vec4<u32>(sh, sh + 1u, sh + 2u, sh + 3u)) & vec4<u32>(1u));
}
fn ksign(i: u32) -> u32 { return i | ((countOneBits(i) & 1u) << 7u); }
fn lut(base: u32, q: vec4<f32>) -> vec4<f32> {
    let i = vec4<u32>(q) + vec4<u32>(base);
    return vec4<f32>(bitcast<f32>(tabs[i.x]), bitcast<f32>(tabs[i.y]), bitcast<f32>(tabs[i.z]), bitcast<f32>(tabs[i.w]));
}
fn e8m0h(e: u32) -> f32 {
    if (e < 2u) { return bitcast<f32>(0x00200000u << e); }
    return bitcast<f32>((e - 1u) << 23u);
}
// 6-bit scale and minimum of sub-block `is` of a Q4_K or Q5_K block, from the 12 bytes at `o`.
fn smk4(b: u32, o: u32, is: u32) -> vec2<f32> {
    if (is < 4u) { return vec2<f32>(f32(byt(b, o + is) & 63u), f32(byt(b, o + is + 4u) & 63u)); }
    let hi = byt(b, o + is + 4u);
    return vec2<f32>(f32((hi & 15u) | ((byt(b, o + is - 4u) >> 6u) << 4u)),
                     f32((hi >> 4u) | ((byt(b, o + is) >> 6u) << 4u)));
}
// smk4 from the scale words already loaded as the y, z, w lanes of a block's first vector.
fn hb(h: vec4<u32>, i: u32) -> u32 {
    let w = select(select(h.w, h.z, i < 8u), h.y, i < 4u);
    return (w >> (8u * (i & 3u))) & 255u;
}
fn smk4v(h: vec4<u32>, is: u32) -> vec2<f32> {
    if (is < 4u) { return vec2<f32>(f32(hb(h, is) & 63u), f32(hb(h, is + 4u) & 63u)); }
    let q = hb(h, is + 4u);
    return vec2<f32>(f32((q & 15u) | ((hb(h, is - 4u) >> 6u) << 4u)), f32((q >> 4u) | ((hb(h, is) >> 6u) << 4u)));
}
"#;

/// Eight values from a pair of 32-value blocks whose 16 bytes of nibbles sit at 16h.
const PAIR_NIBBLES: &str = r#"
    let h = j >> 2u;
    let jj = j & 3u;
    let o = 16u * h + (jj & 1u) * 8u;
    let sh = (jj >> 1u) * 4u;
    let a = fld(wrd(b, o), sh, 15u);
    let c = fld(wrd(b, o + 4u), sh, 15u);
"#;

fn body(t: u32) -> String {
    use quant::*;
    let t1 = |base: usize| format!("{base}u");
    match t {
        Q4_0 => format!("{PAIR_NIBBLES}    let s = hlf(b, 32u + 2u * h);\n    return G(a, c, s, -8.0 * s);"),
        IQ4_NL => format!(
            "{PAIR_NIBBLES}    return G(lut({0}, a), lut({0}, c), hlf(b, 32u + 2u * h), 0.0);",
            t1(Tables::IQ4NL)
        ),
        MXFP4 => format!(
            "{PAIR_NIBBLES}    return G(lut({0}, a), lut({0}, c), e8m0h(byt(b, 32u + h)), 0.0);",
            t1(Tables::MXFP4)
        ),
        Q5_0 => format!(
            r#"{PAIR_NIBBLES}    let hb = (wrd(b, 32u + 4u * h) >> (8u * jj)) & 255u;
    let s = hlf(b, 40u + 2u * h);
    return G(a + 16.0 * sgn0(hb, 0u), c + 16.0 * sgn0(hb, 4u), s, -16.0 * s);"#
        ),
        Q4_1 => r#"
    let o = 4u + (j & 1u) * 8u;
    let sh = (j >> 1u) * 4u;
    return G(fld(wrd(b, o), sh, 15u), fld(wrd(b, o + 4u), sh, 15u), hlf(b, 0u), hlf(b, 2u));"#
            .into(),
        Q5_1 => r#"
    let o = 8u + (j & 1u) * 8u;
    let sh = (j >> 1u) * 4u;
    let hb = (wrd(b, 4u) >> (8u * j)) & 255u;
    return G(fld(wrd(b, o), sh, 15u) + 16.0 * sgn0(hb, 0u), fld(wrd(b, o + 4u), sh, 15u) + 16.0 * sgn0(hb, 4u),
             hlf(b, 0u), hlf(b, 2u));"#
            .into(),
        Q8_0 => r#"
    let h = j >> 2u;
    let o = 32u * h + 8u * (j & 3u);
    return G(sb4(wrd(b, o)), sb4(wrd(b, o + 4u)), hlf(b, 64u + 2u * h), 0.0);"#
            .into(),
        Q2_K => r#"
    let e0 = 8u * j;
    let r = e0 & 127u;
    let o = 16u + 32u * (e0 >> 7u) + 16u * ((r >> 4u) & 1u) + (r & 15u);
    let sh = 2u * (r >> 5u);
    let sc = byt(b, j >> 1u);
    return G(fld(wrd(b, o), sh, 3u), fld(wrd(b, o + 4u), sh, 3u), hlf(b, 80u) * f32(sc & 15u), -hlf(b, 82u) * f32(sc >> 4u));"#
            .into(),
        Q3_K => r#"
    let e0 = 8u * j;
    let n = e0 >> 7u;
    let r = e0 & 127u;
    let si = r >> 5u;
    let l = 16u * ((r >> 4u) & 1u) + (r & 15u);
    let o = 32u + 32u * n + l;
    let hbit = 4u * n + si;
    let a = fld(wrd(b, o), 2u * si, 3u) + 4.0 * fld(wrd(b, l), hbit, 1u);
    let c = fld(wrd(b, o + 4u), 2u * si, 3u) + 4.0 * fld(wrd(b, l + 4u), hbit, 1u);
    let sub = j >> 1u;
    let sc = ((byt(b, 96u + (sub & 7u)) >> (4u * (sub >> 3u))) & 15u) | (((byt(b, 104u + (sub & 3u)) >> (2u * (sub >> 2u))) & 3u) << 4u);
    let s = hlf(b, 108u) * (f32(sc) - 32.0);
    return G(a, c, s, -4.0 * s);"#
            .into(),
        Q4_K | Q5_K => {
            let (qs, high) = if t == Q5_K {
                (
                    "48u",
                    "    a += 16.0 * fld(wrd(b, 16u + l0), is, 1u);\n    c += 16.0 * fld(wrd(b, 20u + l0), is, 1u);\n",
                )
            } else {
                ("16u", "")
            };
            format!(
                r#"
    let e0 = 8u * j;
    let r = e0 & 63u;
    let l0 = r & 31u;
    let o = {qs} + 32u * (e0 >> 6u) + l0;
    let sh = 4u * (r >> 5u);
    let is = j >> 2u;
    var a = fld(wrd(b, o), sh, 15u);
    var c = fld(wrd(b, o + 4u), sh, 15u);
{high}    let sm = smk4(b, 4u, is);
    return G(a, c, hlf(b, 0u) * sm.x, -hlf(b, 2u) * sm.y);"#
            )
        }
        Q6_K => r#"
    let e0 = 8u * j;
    let n = e0 >> 7u;
    let r = e0 & 127u;
    let qt = r >> 5u;
    let l0 = r & 31u;
    let o = 64u * n + l0 + 32u * (qt & 1u);
    let sh = 4u * (qt >> 1u);
    let qo = 128u + 32u * n + l0;
    let a = fld(wrd(b, o), sh, 15u) + 16.0 * fld(wrd(b, qo), 2u * qt, 3u);
    let c = fld(wrd(b, o + 4u), sh, 15u) + 16.0 * fld(wrd(b, qo + 4u), 2u * qt, 3u);
    let sc = bitcast<i32>(byt(b, 192u + (j >> 1u)) << 24u) >> 24u;
    let s = hlf(b, 208u) * f32(sc);
    return G(a, c, s, -32.0 * s);"#
            .into(),
        IQ4_XS => format!(
            r#"
    let ib = j >> 2u;
    let jj = j & 3u;
    let ls = ((byt(b, 4u + (ib >> 1u)) >> (4u * (ib & 1u))) & 15u) | (((u16v(b, 2u) >> (2u * ib)) & 3u) << 4u);
    let o = 8u + 16u * ib + (jj & 1u) * 8u;
    let sh = (jj >> 1u) * 4u;
    return G(lut({0}, fld(wrd(b, o), sh, 15u)), lut({0}, fld(wrd(b, o + 4u), sh, 15u)), hlf(b, 0u) * (f32(ls) - 32.0), 0.0);"#,
            t1(Tables::IQ4NL)
        ),
        IQ2_XXS => format!(
            r#"
    let ib = j >> 2u;
    let l = j & 3u;
    let a0 = wrd(b, 8u * ib);
    let a1 = wrd(b, 8u * ib + 4u);
    let gi = {0} + 2u * ((a0 >> (8u * l)) & 255u);
    let sg = ksign((a1 >> (7u * l)) & 127u);
    return G(fld(tabs[gi], 0u, 255u) * sgn(sg, 0u), fld(tabs[gi + 1u], 0u, 255u) * sgn(sg, 4u),
             hlf(b, 64u) * (0.5 + f32(a1 >> 28u)) * 0.25, 0.0);"#,
            t1(Tables::IQ2XXS)
        ),
        IQ2_XS => format!(
            r#"
    let ib = j >> 2u;
    let l = j & 3u;
    let q = u16v(b, 8u * ib + 2u * l);
    let gi = {0} + 2u * (q & 511u);
    let sg = ksign(q >> 9u);
    let s = hlf(b, 72u) * (0.5 + f32((byt(b, 64u + ib) >> (4u * (l >> 1u))) & 15u)) * 0.25;
    return G(fld(tabs[gi], 0u, 255u) * sgn(sg, 0u), fld(tabs[gi + 1u], 0u, 255u) * sgn(sg, 4u), s, 0.0);"#,
            t1(Tables::IQ2XS)
        ),
        IQ2_S => format!(
            r#"
    let ib = j >> 2u;
    let l = j & 3u;
    let gi = {0} + 2u * (byt(b, 4u * ib + l) | ((byt(b, 64u + ib) << (8u - 2u * l)) & 0x300u));
    let sg = byt(b, 32u + 4u * ib + l);
    let s = hlf(b, 80u) * (0.5 + f32((byt(b, 72u + ib) >> (4u * (l >> 1u))) & 15u)) * 0.25;
    return G(fld(tabs[gi], 0u, 255u) * sgn(sg, 0u), fld(tabs[gi + 1u], 0u, 255u) * sgn(sg, 4u), s, 0.0);"#,
            t1(Tables::IQ2S)
        ),
        IQ3_XXS => format!(
            r#"
    let ib = j >> 2u;
    let l = j & 3u;
    let aux = wrd(b, 64u + 4u * ib);
    let sg = ksign((aux >> (7u * l)) & 127u);
    let g1 = tabs[{0} + byt(b, 8u * ib + 2u * l)];
    let g2 = tabs[{0} + byt(b, 8u * ib + 2u * l + 1u)];
    return G(fld(g1, 0u, 255u) * sgn(sg, 0u), fld(g2, 0u, 255u) * sgn(sg, 4u), hlf(b, 96u) * (0.5 + f32(aux >> 28u)) * 0.5, 0.0);"#,
            t1(Tables::IQ3XXS)
        ),
        IQ3_S => format!(
            r#"
    let ib = j >> 2u;
    let l = j & 3u;
    let qh = byt(b, 64u + ib);
    let g1 = tabs[{0} + (byt(b, 8u * ib + 2u * l) | ((qh << (8u - 2u * l)) & 256u))];
    let g2 = tabs[{0} + (byt(b, 8u * ib + 2u * l + 1u) | ((qh << (7u - 2u * l)) & 256u))];
    let sg = byt(b, 72u + 4u * ib + l);
    let s = hlf(b, 108u) * f32(1u + 2u * ((byt(b, 104u + (ib >> 1u)) >> (4u * (ib & 1u))) & 15u));
    return G(fld(g1, 0u, 255u) * sgn(sg, 0u), fld(g2, 0u, 255u) * sgn(sg, 4u), s, 0.0);"#,
            t1(Tables::IQ3S)
        ),
        IQ1_S => format!(
            r#"
    let ib = j >> 2u;
    let l = j & 3u;
    let qh = u16v(b, 32u + 2u * ib);
    let s = hlf(b, 48u) * f32(2u * ((qh >> 12u) & 7u) + 1u);
    let delta = select(0.125, -0.125, (qh & 0x8000u) != 0u);
    let gi = {0} + 2u * (byt(b, 4u * ib + l) | (((qh >> (3u * l)) & 7u) << 8u));
    return G(sb4(tabs[gi]), sb4(tabs[gi + 1u]), s, s * delta);"#,
            t1(Tables::IQ1S)
        ),
        IQ1_M => format!(
            r#"
    let ib = j >> 2u;
    let l = j & 3u;
    let bits = (u16v(b, 48u) >> 12u) | ((u16v(b, 50u) >> 8u) & 0xf0u) | ((u16v(b, 52u) >> 4u) & 0xf00u) | (u16v(b, 54u) & 0xf000u);
    let scw = u16v(b, 48u + 2u * (ib >> 1u));
    let s = unpack2x16float(bits).x * f32(2u * ((scw >> (6u * (ib & 1u) + 3u * (l >> 1u))) & 7u) + 1u);
    let qhb = byt(b, 32u + 2u * ib + (l >> 1u));
    let odd = (l & 1u) == 1u;
    let gi = {0} + 2u * (byt(b, 4u * ib + l) | ((qhb << select(8u, 4u, odd)) & 0x700u));
    let delta = select(0.125, -0.125, (qhb & select(0x08u, 0x80u, odd)) != 0u);
    return G(sb4(tabs[gi]), sb4(tabs[gi + 1u]), s, s * delta);"#,
            t1(Tables::IQ1S)
        ),
        TQ2_0 => r#"
    let e0 = 8u * j;
    let r = e0 & 127u;
    let o = 32u * (e0 >> 7u) + (r & 31u);
    let sh = 2u * (r >> 5u);
    let s = hlf(b, 64u);
    return G(fld(wrd(b, o), sh, 3u), fld(wrd(b, o + 4u), sh, 3u), s, -s);"#
            .into(),
        TQ1_0 => r#"
    let e0 = 8u * j;
    return G(vec4<f32>(tq1(b, e0), tq1(b, e0 + 1u), tq1(b, e0 + 2u), tq1(b, e0 + 3u)),
             vec4<f32>(tq1(b, e0 + 4u), tq1(b, e0 + 5u), tq1(b, e0 + 6u), tq1(b, e0 + 7u)), hlf(b, 52u), 0.0);"#
            .into(),
        _ => unreachable!("no block decoder for {t}"),
    }
}

/// Extra helpers a format needs beyond `HELPERS`.
fn extras(t: u32) -> &'static str {
    use quant::*;
    match t {
        Q5_0 | Q5_1 => {
            r#"
// Bits sh..sh+3 of `v` as 0 or 1.
fn sgn0(v: u32, sh: u32) -> vec4<f32> {
    return vec4<f32>((vec4<u32>(v) >> vec4<u32>(sh, sh + 1u, sh + 2u, sh + 3u)) & vec4<u32>(1u));
}
"#
        }
        TQ1_0 => {
            r#"
// One ternary value: five or four values share a byte as base-3 digits.
fn tq1(b: u32, e: u32) -> f32 {
    var o = e & 31u;
    var n = e >> 5u;
    if (e >= 240u) { o = 48u + ((e - 240u) & 3u); n = (e - 240u) >> 2u; }
    else if (e >= 160u) { o = 32u + ((e - 160u) & 15u); n = (e - 160u) >> 4u; }
    var pw = 1u;
    for (var i = 0u; i < n; i++) { pw *= 3u; }
    let q = (byt(b, o) * pw) & 255u;
    return f32((q * 3u) >> 8u) - 1.0;
}
"#
        }
        _ => "",
    }
}

/// `BV` values in `BW` words per GPU block, the helpers and `grp(b, j)` for one format.
pub fn decoder(t: u32) -> String {
    let (vals, words, _) = gpu_block(t).expect("block format");
    format!(
        "const BV: u32 = {vals}u;\nconst BW: u32 = {words}u;\n{HELPERS}{}\nfn grp(b: u32, j: u32) -> G {{{}\n}}\n",
        extras(t),
        body(t)
    )
}

/// A hand-written `chunk_dot` (32 values) for formats where the generic one built on `grp`
/// repeats loads and scale decoding. Each word of codes is loaded once.
pub fn chunk_dot(t: u32) -> Option<&'static str> {
    use quant::*;
    Some(match t {
        Q8_0 => r#"
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let b = (row * ((p.K + 63u) / 64u) + (c >> 1u)) * 17u;
    let h = c & 1u;
    let o = b + 8u * h;
    let xi = xb + c * 8u;
    var d = 0.0;
    for (var i = 0u; i < 8u; i++) { d += dot(sb4(wcodes[o + i]), x[xi + i]); }
    return hlf(b, 64u + 2u * h) * d;
}
"#,
        Q4_0 | IQ4_NL | MXFP4 => match t {
            Q4_0 => r#"
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let b = (row * ((p.K + 63u) / 64u) + (c >> 1u)) * 9u;
    let h = c & 1u;
    let o = b + 4u * h;
    let xi = xb + c * 8u;
    var d = 0.0;
    var xs = vec4<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let w = wcodes[o + i];
        let x0 = x[xi + i];
        let x1 = x[xi + 4u + i];
        d += dot(fld(w, 0u, 15u), x0) + dot(fld(w, 4u, 15u), x1);
        xs += x0 + x1;
    }
    return hlf(b, 32u + 2u * h) * (d - 8.0 * dot(xs, vec4<f32>(1.0)));
}
"#,
            IQ4_NL => r#"
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let b = (row * ((p.K + 63u) / 64u) + (c >> 1u)) * 9u;
    let h = c & 1u;
    let o = b + 4u * h;
    let xi = xb + c * 8u;
    var d = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let w = wcodes[o + i];
        d += dot(lut(0u, fld(w, 0u, 15u)), x[xi + i]) + dot(lut(0u, fld(w, 4u, 15u)), x[xi + 4u + i]);
    }
    return hlf(b, 32u + 2u * h) * d;
}
"#,
            _ => r#"
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let b = (row * ((p.K + 63u) / 64u) + (c >> 1u)) * 9u;
    let h = c & 1u;
    let o = b + 4u * h;
    let xi = xb + c * 8u;
    var d = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let w = wcodes[o + i];
        d += dot(lut(16u, fld(w, 0u, 15u)), x[xi + i]) + dot(lut(16u, fld(w, 4u, 15u)), x[xi + 4u + i]);
    }
    return e8m0h(byt(b, 32u + h)) * d;
}
"#,
        },
        Q5_0 => r#"
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let b = (row * ((p.K + 63u) / 64u) + (c >> 1u)) * 11u;
    let h = c & 1u;
    let o = b + 4u * h;
    let qh = wcodes[b + 8u + h];
    let xi = xb + c * 8u;
    var d = 0.0;
    var xs = vec4<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let w = wcodes[o + i];
        let x0 = x[xi + i];
        let x1 = x[xi + 4u + i];
        d += dot(fld(w, 0u, 15u) + 16.0 * sgn0(qh, 4u * i), x0) + dot(fld(w, 4u, 15u) + 16.0 * sgn0(qh, 16u + 4u * i), x1);
        xs += x0 + x1;
    }
    return hlf(b, 40u + 2u * h) * (d - 16.0 * dot(xs, vec4<f32>(1.0)));
}
"#,
        Q4_K | Q5_K => match t {
            Q4_K => r#"
const CHUNK: u32 = 256u;
// A whole super-block per step: its scales are decoded once, its codes read as 16-byte vectors.
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let v = (row * (p.K >> 8u) + c) * 9u;
    let h = wv4[v];
    let dm = unpack2x16float(h.x);
    let xi = xb + c * 64u;
    let one = vec4<f32>(1.0);
    var acc = 0.0;
    var mins = 0.0;
    for (var j = 0u; j < 4u; j++) {
        let w0 = wv4[v + 1u + 2u * j];
        let w1 = wv4[v + 2u + 2u * j];
        let xo = xi + 16u * j;
        var lo = 0.0;
        var hi = 0.0;
        var sl = vec4<f32>(0.0);
        var sh = vec4<f32>(0.0);
        for (var i = 0u; i < 4u; i++) {
            let xa = x[xo + i];
            let xb2 = x[xo + 4u + i];
            let xc = x[xo + 8u + i];
            let xd = x[xo + 12u + i];
            lo += dot(fld(w0[i], 0u, 15u), xa) + dot(fld(w1[i], 0u, 15u), xb2);
            hi += dot(fld(w0[i], 4u, 15u), xc) + dot(fld(w1[i], 4u, 15u), xd);
            sl += xa + xb2;
            sh += xc + xd;
        }
        let s0 = smk4v(h, 2u * j);
        let s1 = smk4v(h, 2u * j + 1u);
        acc += s0.x * lo + s1.x * hi;
        mins += s0.y * dot(sl, one) + s1.y * dot(sh, one);
    }
    return dm.x * acc - dm.y * mins;
}
"#,
            _ => r#"
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let b = (row * (p.K >> 8u) + (c >> 3u)) * 44u;
    let is = c & 7u;
    let o = b + 12u + 8u * (is >> 1u);
    let sh = 4u * (is & 1u);
    let xi = xb + c * 8u;
    var d = 0.0;
    var xs = vec4<f32>(0.0);
    for (var i = 0u; i < 8u; i++) {
        let xv = x[xi + i];
        d += dot(fld(wcodes[o + i], sh, 15u) + 16.0 * fld(wcodes[b + 4u + i], is, 1u), xv);
        xs += xv;
    }
    let dm = unpack2x16float(wcodes[b]);
    let sm = smk4(b, 4u, is);
    return dm.x * sm.x * d - dm.y * sm.y * dot(xs, vec4<f32>(1.0));
}
"#,
        },
        Q6_K => r#"
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let b = (row * (p.K >> 8u) + (c >> 3u)) * 53u;
    let r = c & 7u;
    let n = r >> 2u;
    let qt = r & 3u;
    let o = b + 16u * n + 8u * (qt & 1u);
    let sh = 4u * (qt >> 1u);
    let qo = b + 32u + 8u * n;
    let xi = xb + c * 8u;
    var d0 = 0.0;
    var d1 = 0.0;
    var s0 = vec4<f32>(0.0);
    var s1 = vec4<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let xa = x[xi + i];
        let xc = x[xi + 4u + i];
        d0 += dot(fld(wcodes[o + i], sh, 15u) + 16.0 * fld(wcodes[qo + i], 2u * qt, 3u), xa);
        d1 += dot(fld(wcodes[o + 4u + i], sh, 15u) + 16.0 * fld(wcodes[qo + 4u + i], 2u * qt, 3u), xc);
        s0 += xa;
        s1 += xc;
    }
    let sub = 2u * r;
    let c0 = f32(bitcast<i32>(byt(b, 192u + sub) << 24u) >> 24u);
    let c1 = f32(bitcast<i32>(byt(b, 193u + sub) << 24u) >> 24u);
    let one = vec4<f32>(1.0);
    return hlf(b, 208u) * (c0 * (d0 - 32.0 * dot(s0, one)) + c1 * (d1 - 32.0 * dot(s1, one)));
}
"#,
        Q2_K => r#"
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let b = (row * (p.K >> 8u) + (c >> 3u)) * 21u;
    let r = c & 7u;
    let o = b + 4u + 8u * (r >> 2u);
    let sh = 2u * (r & 3u);
    let xi = xb + c * 8u;
    var d0 = 0.0;
    var d1 = 0.0;
    var s0 = vec4<f32>(0.0);
    var s1 = vec4<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let xa = x[xi + i];
        let xc = x[xi + 4u + i];
        d0 += dot(fld(wcodes[o + i], sh, 3u), xa);
        d1 += dot(fld(wcodes[o + 4u + i], sh, 3u), xc);
        s0 += xa;
        s1 += xc;
    }
    let sc0 = byt(b, 2u * r);
    let sc1 = byt(b, 2u * r + 1u);
    let dm = unpack2x16float(wcodes[b + 20u]);
    let one = vec4<f32>(1.0);
    return dm.x * (f32(sc0 & 15u) * d0 + f32(sc1 & 15u) * d1) - dm.y * (f32(sc0 >> 4u) * dot(s0, one) + f32(sc1 >> 4u) * dot(s1, one));
}
"#,
        Q3_K => r#"
fn q3scale(b: u32, sub: u32) -> f32 {
    let sc = ((byt(b, 96u + (sub & 7u)) >> (4u * (sub >> 3u))) & 15u) | (((byt(b, 104u + (sub & 3u)) >> (2u * (sub >> 2u))) & 3u) << 4u);
    return f32(sc) - 32.0;
}
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let b = (row * (p.K >> 8u) + (c >> 3u)) * 28u;
    let r = c & 7u;
    let n = r >> 2u;
    let si = r & 3u;
    let o = b + 8u + 8u * n;
    let hbit = 4u * n + si;
    let xi = xb + c * 8u;
    var d0 = 0.0;
    var d1 = 0.0;
    var s0 = vec4<f32>(0.0);
    var s1 = vec4<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let xa = x[xi + i];
        let xc = x[xi + 4u + i];
        d0 += dot(fld(wcodes[o + i], 2u * si, 3u) + 4.0 * fld(wcodes[b + i], hbit, 1u), xa);
        d1 += dot(fld(wcodes[o + 4u + i], 2u * si, 3u) + 4.0 * fld(wcodes[b + 4u + i], hbit, 1u), xc);
        s0 += xa;
        s1 += xc;
    }
    let one = vec4<f32>(1.0);
    return hlf(b, 108u) * (q3scale(b, 2u * r) * (d0 - 4.0 * dot(s0, one)) + q3scale(b, 2u * r + 1u) * (d1 - 4.0 * dot(s1, one)));
}
"#,
        IQ4_XS => r#"
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let b = (row * (p.K >> 8u) + (c >> 3u)) * 34u;
    let ib = c & 7u;
    let o = b + 2u + 4u * ib;
    let xi = xb + c * 8u;
    var d = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let w = wcodes[o + i];
        d += dot(lut(0u, fld(w, 0u, 15u)), x[xi + i]) + dot(lut(0u, fld(w, 4u, 15u)), x[xi + 4u + i]);
    }
    let ls = ((byt(b, 4u + (ib >> 1u)) >> (4u * (ib & 1u))) & 15u) | (((u16v(b, 2u) >> (2u * ib)) & 3u) << 4u);
    return hlf(b, 0u) * (f32(ls) - 32.0) * d;
}
"#,
        _ => return None,
    })
}
