//! WGSL compute kernels. Weight formats are compiled in as constants, so each
//! matrix layout gets its own specialised matrix-vector and matrix-matrix kernel.

use super::blocks;
use crate::quant::GpuFormat;

/// Bindings shared by every kernel: per-step values and per-dispatch parameters.
const COMMON: &str = r#"
struct Step { n_tokens: u32, pos0: u32, kv_len: u32, last: u32 }
@group(0) @binding(0) var<uniform> step: Step;
"#;

/// Binding 3 of the matrix kernels: f32 scales of an expanded layout, or the code books.
fn binding3(fmt: GpuFormat) -> &'static str {
    match fmt {
        GpuFormat::Block(_) => "@group(0) @binding(3) var<storage, read> tabs: array<u32>;\n",
        _ => "@group(0) @binding(3) var<storage, read> wscales: array<f32>;\n",
    }
}

/// Matrix kernels: params, weight codes, weight scales, input, output, bias.
const MATMUL_BINDINGS: &str = r#"
struct P { K: u32, N: u32, out_stride: u32, out_offset: u32, mode: u32, has_bias: u32, x_offset: u32, x_stride: u32,
           group_rows: u32, group_stride: u32, n_used: u32, slot0: u32, exp_rows: u32, map_offset: u32, w_offset: u32, pad1: u32 }
@group(0) @binding(1) var<uniform> p: P;
// Input offset of output row `r` (of this shard): rows in groups read their own slice of x.
fn xbase(r: u32) -> u32 {
    if (p.group_rows == 0u) { return p.x_offset; }
    return p.x_offset + ((p.out_offset + r) / p.group_rows) * p.group_stride;
}
@group(0) @binding(2) var<storage, read> wcodes: array<u32>;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<storage, read> bias: array<f32>;

struct W8 { a: vec4<f32>, b: vec4<f32> }

fn finish(t: u32, r: u32, acc: f32) {
    var v = acc;
    if (p.has_bias == 1u) { v += bias[p.out_offset + r]; }
    let i = t * p.out_stride + p.out_offset + r;
    if (p.mode == 0u) {
        y[i] = v;
    } else if (p.mode == 1u) {
        y[i] = y[i] + v;
    } else {
        let g = y[i];
        y[i] = g / (1.0 + exp(-g)) * v;
    }
}
"#;

/// Constants and the code decoder shared by every kernel for the nibble layouts.
fn q4_common(group: u8, high: u8, offset: u8, bias: bool) -> String {
    format!(
        r#"
const OFF: f32 = {offset}.0;
const SB: bool = {bias};
const STRIDE: u32 = {stride}u;
const GROUP: u32 = {group}u;
const HIGH: u32 = {high}u;
// Four codes: nibbles of `w` from bit `ns`, plus extra bits of `h` from bit `hs`.
fn vals(w: u32, ns: u32, h: u32, hs: u32) -> vec4<f32> {{
    var q = vec4<u32>(w >> ns, w >> (ns + 4u), w >> (ns + 8u), w >> (ns + 12u)) & vec4<u32>(15u);
    if (HIGH == 1u) {{
        q |= (vec4<u32>(h >> hs, h >> (hs + 1u), h >> (hs + 2u), h >> (hs + 3u)) & vec4<u32>(1u)) << vec4<u32>(4u);
    }} else if (HIGH == 2u) {{
        q |= (vec4<u32>(h >> hs, h >> (hs + 2u), h >> (hs + 4u), h >> (hs + 6u)) & vec4<u32>(3u)) << vec4<u32>(4u);
    }}
    return vec4<f32>(q);
}}
"#,
        stride = 1 + bias as u32
    )
}

/// `load8(row, k)` for one weight format: eight consecutive values of a row.
fn loader(fmt: GpuFormat) -> String {
    match fmt {
        GpuFormat::Q4 {
            group,
            high,
            offset,
            bias,
        } => {
            q4_common(group, high, offset, bias)
                + r#"
fn load8(r: u32, k: u32) -> W8 {
    let e = r * p.K + k;
    let w = wcodes[e >> 3u];
    let g = (e / GROUP) * STRIDE;
    let s = wscales[g];
    let hb = p.N * (p.K >> 3u);
    var h = 0u;
    var hs = 0u;
    var step = 0u;
    if (HIGH == 1u) { h = wcodes[hb + (e >> 5u)]; hs = e & 31u; step = 4u; }
    if (HIGH == 2u) { h = wcodes[hb + (e >> 4u)]; hs = (e & 15u) * 2u; step = 8u; }
    var o = W8((vals(w, 0u, h, hs) - vec4<f32>(OFF)) * s, (vals(w, 16u, h, hs + step) - vec4<f32>(OFF)) * s);
    if (SB) {
        let b = wscales[g + 1u];
        o.a += vec4<f32>(b);
        o.b += vec4<f32>(b);
    }
    return o;
}
"#
        }
        GpuFormat::Q8 { group, bias } => format!(
            r#"
const SB: bool = {bias};
const STRIDE: u32 = {stride}u;
const GROUP: u32 = {group}u;
fn i8x4(w: u32) -> vec4<f32> {{
    let v = vec4<i32>(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w));
    return vec4<f32>(v >> vec4<u32>(24u));
}}
fn load8(r: u32, k: u32) -> W8 {{
    let e = r * p.K + k;
    let wi = e >> 2u;
    let g = (e / GROUP) * STRIDE;
    let s = wscales[g];
    var o = W8(i8x4(wcodes[wi]) * s, i8x4(wcodes[wi + 1u]) * s);
    if (SB) {{
        let b = wscales[g + 1u];
        o.a += vec4<f32>(b);
        o.b += vec4<f32>(b);
    }}
    return o;
}}
"#,
            stride = 1 + bias as u32
        ),
        GpuFormat::Block(t) => {
            blocks::decoder(t)
                + r#"
fn load8(r: u32, k: u32) -> W8 {
    let b = (r * ((p.K + BV - 1u) / BV) + k / BV) * BW;
    let g = grp(b, (k % BV) >> 3u);
    return W8(g.a * g.s + vec4<f32>(g.m), g.b * g.s + vec4<f32>(g.m));
}
"#
        }
        GpuFormat::F16 => r#"
fn load8(r: u32, k: u32) -> W8 {
    let e = (r * p.K + k) >> 1u;
    let a = vec4<f32>(unpack2x16float(wcodes[e]), unpack2x16float(wcodes[e + 1u]));
    let b = vec4<f32>(unpack2x16float(wcodes[e + 2u]), unpack2x16float(wcodes[e + 3u]));
    return W8(a, b);
}
"#
        .to_string(),
    }
}

/// Dot product of one chunk of a weight row with the matching inputs.
fn gemv_chunk(fmt: GpuFormat) -> String {
    match fmt {
        GpuFormat::Q4 {
            group,
            high,
            offset,
            bias,
        } => {
            q4_common(group, high, offset, bias)
                + r#"
const CHUNK: u32 = 32u;
fn word(i: u32) -> u32 {
    return wc4[i >> 2u][i & 3u];
}
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let ci = row * (p.K >> 5u) + c;
    let w = wc4[ci];
    let hb = p.N * (p.K >> 3u);
    var h0 = 0u;
    var h1 = 0u;
    if (HIGH == 1u) { h0 = word(hb + ci); h1 = h0; }
    if (HIGH == 2u) { h0 = word(hb + 2u * ci); h1 = word(hb + 2u * ci + 1u); }
    // Bit position of the extra bits for each run of four values.
    let st = select(4u, 8u, HIGH == 2u);
    let b2 = select(16u, 0u, HIGH == 2u);
    let xi = xb + c * 8u;
    let x0 = x[xi]; let x1 = x[xi + 1u]; let x2 = x[xi + 2u]; let x3 = x[xi + 3u];
    let x4 = x[xi + 4u]; let x5 = x[xi + 5u]; let x6 = x[xi + 6u]; let x7 = x[xi + 7u];
    let d0 = dot(vals(w.x, 0u, h0, 0u), x0) + dot(vals(w.x, 16u, h0, st), x1)
           + dot(vals(w.y, 0u, h0, 2u * st), x2) + dot(vals(w.y, 16u, h0, 3u * st), x3);
    let d1 = dot(vals(w.z, 0u, h1, b2), x4) + dot(vals(w.z, 16u, h1, b2 + st), x5)
           + dot(vals(w.w, 0u, h1, b2 + 2u * st), x6) + dot(vals(w.w, 16u, h1, b2 + 3u * st), x7);
    let one = vec4<f32>(1.0);
    let s0 = dot(x0 + x1 + x2 + x3, one);
    let s1 = dot(x4 + x5 + x6 + x7, one);
    if (GROUP == 32u) {
        let g = ci * STRIDE;
        var r = wscales[g] * (d0 + d1 - OFF * (s0 + s1));
        if (SB) { r += wscales[g + 1u] * (s0 + s1); }
        return r;
    }
    let g = 2u * ci * STRIDE;
    var r = wscales[g] * (d0 - OFF * s0) + wscales[g + STRIDE] * (d1 - OFF * s1);
    if (SB) { r += wscales[g + 1u] * s0 + wscales[g + STRIDE + 1u] * s1; }
    return r;
}
"#
        }
        GpuFormat::Q8 { group, bias } => format!(
            r#"
const CHUNK: u32 = 16u;
fn i8x4(w: u32) -> vec4<f32> {{
    let v = vec4<i32>(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w));
    return vec4<f32>(v >> vec4<u32>(24u));
}}
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {{
    let ci = row * (p.K >> 4u) + c;
    let w = wc4[ci];
    let xi = xb + c * 4u;
    let x0 = x[xi]; let x1 = x[xi + 1u]; let x2 = x[xi + 2u]; let x3 = x[xi + 3u];
    let d = dot(i8x4(w.x), x0) + dot(i8x4(w.y), x1) + dot(i8x4(w.z), x2) + dot(i8x4(w.w), x3);
    let g = (ci * 16u) / {group}u;
    var r = wscales[g * {stride}u] * d;
    if ({bias}) {{ r += wscales[g * {stride}u + 1u] * dot(x0 + x1 + x2 + x3, vec4<f32>(1.0)); }}
    return r;
}}
"#,
            stride = 1 + bias as u32
        ),
        GpuFormat::Block(t) if blocks::chunk_dot(t).is_some() => {
            let body = blocks::chunk_dot(t).unwrap();
            let chunk = if body.contains("const CHUNK") { "" } else { "const CHUNK: u32 = 32u;\n" };
            format!("{}\n{chunk}{body}", blocks::decoder(t))
        }
        GpuFormat::Block(t) => {
            blocks::decoder(t)
                + r#"
const CHUNK: u32 = 32u;
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let e = c * 32u;
    let b = (row * ((p.K + BV - 1u) / BV) + e / BV) * BW;
    let j = (e % BV) >> 3u;
    let xi = xb + c * 8u;
    var r = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let g = grp(b, j + i);
        let x0 = x[xi + 2u * i];
        let x1 = x[xi + 2u * i + 1u];
        r += g.s * (dot(g.a, x0) + dot(g.b, x1)) + g.m * dot(x0 + x1, vec4<f32>(1.0));
    }
    return r;
}
"#
        }
        GpuFormat::F16 => r#"
const CHUNK: u32 = 8u;
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {
    let w = wc4[row * (p.K >> 3u) + c];
    let xi = xb + c * 2u;
    let a = vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));
    let b = vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w));
    return dot(a, x[xi]) + dot(b, x[xi + 1u]);
}
"#
        .to_string(),
    }
}

/// One token: `256 / lanes` rows per workgroup, `lanes` lanes per row, one chunk per lane and step.
pub fn gemv(fmt: GpuFormat, lanes: u32) -> String {
    let bindings = match fmt {
        // The same weights again as 16-byte vectors, for kernels whose code arrays are aligned.
        GpuFormat::Block(_) => MATMUL_BINDINGS.to_string()
            + binding3(fmt)
            + "@group(0) @binding(7) var<storage, read> wv4: array<vec4<u32>>;\n",
        _ => MATMUL_BINDINGS.replace(
            "var<storage, read> wcodes: array<u32>;",
            "var<storage, read> wc4: array<vec4<u32>>;",
        ) + binding3(fmt),
    };
    format!(
        "{COMMON}{bindings}{}{}{}",
        gemv_chunk(fmt),
        format!(
            "\nconst LANES: u32 = {lanes}u;\nconst ROWS: u32 = {}u;\n",
            256 / lanes
        ),
        r#"
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let row = (wg.x + wg.y * nwg.x) * ROWS + li / LANES;
    let lane = li % LANES;
    var acc = 0.0;
    if (row < p.N) {
        let xb = xbase(row) >> 2u;
        let chunks = p.K / CHUNK;
        for (var c = lane; c < chunks; c += LANES) {
            acc += chunk_dot(row, c, xb);
        }
    }
    red[li] = acc;
    workgroupBarrier();
    for (var s = LANES / 2u; s > 0u; s >>= 1u) {
        if (lane < s) { red[li] += red[li + s]; }
        workgroupBarrier();
    }
    if (lane == 0u && row < p.N) { finish(0u, row, red[li]); }
}
"#
    )
}

/// Several tokens: 64 rows x 32 tokens per workgroup, through shared-memory tiles.
pub fn gemm(fmt: GpuFormat) -> String {
    format!(
        "{COMMON}{MATMUL_BINDINGS}{}{}{}",
        binding3(fmt),
        loader(fmt),
        r#"
const WS: u32 = 9u; // padded row stride in vec4s, avoids shared-memory bank conflicts
var<workgroup> xs: array<vec4<f32>, 256>;
var<workgroup> ws: array<vec4<f32>, 576>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row0 = wg.x * 64u;
    let t0 = wg.y * 32u;
    let T = step.n_tokens;
    let tr = li & 15u;
    let tt = li >> 4u;
    var a00 = 0.0; var a01 = 0.0; var a10 = 0.0; var a11 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a30 = 0.0; var a31 = 0.0;
    let xt = t0 + (li >> 3u);
    let wr = row0 + (li >> 2u);
    let wslot = (li >> 2u) * WS + (li & 3u) * 2u;
    for (var k0 = 0u; k0 < p.K; k0 += 32u) {
        var xv = vec4<f32>(0.0);
        if (xt < T) { xv = x[((xbase(row0) + xt * p.x_stride + k0) >> 2u) + (li & 7u)]; }
        xs[li] = xv;
        var w = W8(vec4<f32>(0.0), vec4<f32>(0.0));
        if (wr < p.N) { w = load8(wr, k0 + (li & 3u) * 8u); }
        ws[wslot] = w.a;
        ws[wslot + 1u] = w.b;
        workgroupBarrier();
        for (var kv = 0u; kv < 8u; kv++) {
            let x0 = xs[tt * 8u + kv];
            let x1 = xs[(tt + 16u) * 8u + kv];
            let w0 = ws[tr * WS + kv];
            let w1 = ws[(tr + 16u) * WS + kv];
            let w2 = ws[(tr + 32u) * WS + kv];
            let w3 = ws[(tr + 48u) * WS + kv];
            a00 += dot(w0, x0); a01 += dot(w0, x1);
            a10 += dot(w1, x0); a11 += dot(w1, x1);
            a20 += dot(w2, x0); a21 += dot(w2, x1);
            a30 += dot(w3, x0); a31 += dot(w3, x1);
        }
        workgroupBarrier();
    }
    let ta = t0 + tt;
    let tb = t0 + tt + 16u;
    let r0 = row0 + tr;
    if (r0 < p.N) { if (ta < T) { finish(ta, r0, a00); } if (tb < T) { finish(tb, r0, a01); } }
    if (r0 + 16u < p.N) { if (ta < T) { finish(ta, r0 + 16u, a10); } if (tb < T) { finish(tb, r0 + 16u, a11); } }
    if (r0 + 32u < p.N) { if (ta < T) { finish(ta, r0 + 32u, a20); } if (tb < T) { finish(tb, r0 + 32u, a21); } }
    if (r0 + 48u < p.N) { if (ta < T) { finish(ta, r0 + 48u, a30); } if (tb < T) { finish(tb, r0 + 48u, a31); } }
}
"#
    )
}

/// Token embedding lookup from a (possibly sharded) quantized table.
pub fn embed(fmt: GpuFormat) -> String {
    format!(
        "{COMMON}{MATMUL_BINDINGS}{}{}{}",
        binding3(fmt),
        loader(fmt),
        r#"
@group(0) @binding(7) var<storage, read> tokens: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    // Reuses the matrix parameters: K = width, N = rows in this shard, out_offset = first row.
    let t = wg.y;
    let c = wg.x * 64u + li;
    if (t >= step.n_tokens || c * 8u >= p.K) { return; }
    let tok = tokens[t];
    if (tok < p.out_offset || tok >= p.out_offset + p.N) { return; }
    let w = load8(tok - p.out_offset, c * 8u);
    let o = t * p.K + c * 8u;
    y[o] = w.a.x; y[o + 1u] = w.a.y; y[o + 2u] = w.a.z; y[o + 3u] = w.a.w;
    y[o + 4u] = w.b.x; y[o + 5u] = w.b.y; y[o + 6u] = w.b.z; y[o + 7u] = w.b.w;
}
"#
    )
}

/// RMSNorm over rows of `len` values. `inplace` normalises one buffer (per-head norms).
pub fn rmsnorm(inplace: bool) -> String {
    let (bindings, read, write) = if inplace {
        ("@group(0) @binding(2) var<storage, read_write> buf: array<f32>;\n@group(0) @binding(3) var<storage, read> w: array<f32>;", "buf", "buf")
    } else {
        ("@group(0) @binding(2) var<storage, read> src: array<f32>;\n@group(0) @binding(3) var<storage, read> w: array<f32>;\n@group(0) @binding(4) var<storage, read_write> dst: array<f32>;", "src", "dst")
    };
    format!(
        r#"{COMMON}
struct P {{ len: u32, rows_per_token: u32, last_only: u32, eps: f32 }}
@group(0) @binding(1) var<uniform> p: P;
{bindings}
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
    var row = wg.x;
    var out_row = wg.x;
    if (p.last_only == 1u) {{
        row = step.last;
        out_row = 0u;
    }}
    let base = row * p.len;
    let ob = out_row * p.len;
    var s = 0.0;
    for (var i = li; i < p.len; i += 256u) {{
        let v = {read}[base + i];
        s += v * v;
    }}
    red[li] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {{
        if (li < k) {{ red[li] += red[li + k]; }}
        workgroupBarrier();
    }}
    let scale = inverseSqrt(red[0] / f32(p.len) + p.eps);
    for (var i = li; i < p.len; i += 256u) {{
        {write}[ob + i] = {read}[base + i] * scale * w[i];
    }}
}}
"#
    )
}

/// Rotary position embedding on queries (in place) and keys, then store keys
/// and values in the f16 cache at their positions.
pub const ROPE: &str = r#"
struct P { n_head: u32, n_kv: u32, head_dim: u32, half_rot: u32, neox: u32, kv_dim: u32, q_dim: u32, rot_offset: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read> k: array<f32>;
@group(0) @binding(4) var<storage, read> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> kc: array<u32>;
@group(0) @binding(6) var<storage, read_write> vc: array<u32>;
@group(0) @binding(7) var<storage, read> table: array<vec2<f32>>;

fn rot_k(base: u32, d: u32, cs_base: u32) -> f32 {
    let x = k[base + d];
    if (d >= 2u * p.half_rot) { return x; }
    if (p.neox == 1u) {
        let h = p.half_rot;
        if (d < h) {
            let cs = table[cs_base + d];
            return x * cs.x - k[base + d + h] * cs.y;
        }
        let cs = table[cs_base + d - h];
        return k[base + d - h] * cs.y + x * cs.x;
    }
    let cs = table[cs_base + d / 2u];
    if ((d & 1u) == 0u) { return x * cs.x - k[base + d + 1u] * cs.y; }
    return k[base + d - 1u] * cs.y + x * cs.x;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.y;
    let pos = step.pos0 + t;
    let cs_base = pos * p.half_rot;
    let hd = p.head_dim;
    if (wg.x < p.n_head) {
        let base = t * p.q_dim + wg.x * hd + p.rot_offset;
        for (var i = li; i < p.half_rot; i += 64u) {
            let cs = table[cs_base + i];
            var a = base + 2u * i;
            var b = a + 1u;
            if (p.neox == 1u) { a = base + i; b = base + i + p.half_rot; }
            let x0 = q[a];
            let x1 = q[b];
            q[a] = x0 * cs.x - x1 * cs.y;
            q[b] = x0 * cs.y + x1 * cs.x;
        }
        return;
    }
    let kvh = wg.x - p.n_head;
    let base = t * p.kv_dim + kvh * hd;
    let cache = (pos * p.kv_dim + kvh * hd) >> 1u;
    for (var e = li; e < hd / 2u; e += 64u) {
        let d = 2u * e;
        kc[cache + e] = pack2x16float(vec2<f32>(rot_k(base, d, cs_base), rot_k(base, d + 1u, cs_base)));
        vc[cache + e] = pack2x16float(vec2<f32>(v[base + d], v[base + d + 1u]));
    }
}
"#;

/// Causal attention for one (head, token) per workgroup with a streaming
/// softmax, so context length is bounded only by the cache.
pub const ATTENTION: &str = r#"
struct P { n_head: u32, n_kv: u32, head_dim: u32, kv_dim: u32, q_dim: u32, scale: f32, pad0: u32, pad1: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kc: array<u32>;
@group(0) @binding(4) var<storage, read> vc: array<u32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;

const WG: u32 = 256u;
var<workgroup> qs: array<f32, 256>;
var<workgroup> pr: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.x;
    let t = wg.y;
    let hd = p.head_dim;
    let kvh = h / (p.n_head / p.n_kv);
    if (li < hd) { qs[li] = q[t * p.q_dim + h * hd + li]; }
    workgroupBarrier();
    let n_keys = step.pos0 + t + 1u;
    let groups = WG / hd;
    let d = li % hd;
    let g = li / hd;
    let live = g < groups;
    let kbase = kvh * hd;
    var m = -3.0e38;
    var l = 0.0;
    var acc = 0.0;
    for (var j0 = 0u; j0 < n_keys; j0 += WG) {
        let j = j0 + li;
        var s = -3.0e38;
        if (j < n_keys) {
            let kb = (j * p.kv_dim + kbase) >> 1u;
            var dv = 0.0;
            for (var e = 0u; e < hd / 2u; e++) {
                let kk = unpack2x16float(kc[kb + e]);
                dv += qs[2u * e] * kk.x + qs[2u * e + 1u] * kk.y;
            }
            s = dv * p.scale;
        }
        red[li] = s;
        workgroupBarrier();
        for (var r = 128u; r > 0u; r >>= 1u) {
            if (li < r) { red[li] = max(red[li], red[li + r]); }
            workgroupBarrier();
        }
        let m_new = max(m, red[0]);
        workgroupBarrier();
        var pj = 0.0;
        if (j < n_keys) { pj = exp(s - m_new); }
        pr[li] = pj;
        red[li] = pj;
        workgroupBarrier();
        for (var r = 128u; r > 0u; r >>= 1u) {
            if (li < r) { red[li] += red[li + r]; }
            workgroupBarrier();
        }
        let corr = exp(m - m_new);
        l = l * corr + red[0];
        acc = acc * corr;
        if (live) {
            let nk = min(WG, n_keys - j0);
            for (var jj = g; jj < nk; jj += groups) {
                let word = vc[((j0 + jj) * p.kv_dim + kbase + d) >> 1u];
                let pair = unpack2x16float(word);
                acc += pr[jj] * select(pair.x, pair.y, (d & 1u) == 1u);
            }
        }
        m = m_new;
        workgroupBarrier();
    }
    red[li] = select(0.0, acc, live);
    workgroupBarrier();
    if (live && g == 0u) {
        var total = 0.0;
        for (var gg = 0u; gg < groups; gg++) { total += red[gg * hd + d]; }
        o[t * p.q_dim + h * hd + d] = total / l;
    }
}
"#;

pub fn rope() -> String {
    format!("{COMMON}{ROPE}")
}

pub fn attention() -> String {
    format!("{COMMON}{ATTENTION}")
}

/// Expert matrices: the input and output of one (token, chosen expert) pair per workgroup row.
fn id_bindings(fmt: GpuFormat) -> String {
    let base = match fmt {
        GpuFormat::Block(_) => MATMUL_BINDINGS.to_string()
            + binding3(fmt)
            + "@group(0) @binding(7) var<storage, read> wv4: array<vec4<u32>>;\n",
        _ => MATMUL_BINDINGS.replace(
            "var<storage, read> wcodes: array<u32>;",
            "var<storage, read> wc4: array<vec4<u32>>;",
        ) + binding3(fmt),
    };
    base.replace(
        "@group(0) @binding(6) var<storage, read> bias: array<f32>;",
        "// Chosen experts per (token, slot), then their weights as f32 bits from `w_offset`.\n@group(0) @binding(6) var<storage, read> route: array<u32>;\n// Expert to resident slot, or 0xffffffff when the expert is not on the GPU.\n@group(0) @binding(8) var<storage, read> emap: array<u32>;",
    )
    .replace("    if (p.has_bias == 1u) { v += bias[p.out_offset + r]; }\n", "")
}

/// Gate or up projection of chosen experts: one (token, expert) pair per workgroup column.
pub fn gemv_id(fmt: GpuFormat, lanes: u32) -> String {
    format!(
        "{COMMON}{}{}\nconst LANES: u32 = {lanes}u;\nconst ROWS: u32 = {}u;\n{}",
        id_bindings(fmt),
        gemv_chunk(fmt),
        256 / lanes,
        r#"
var<workgroup> red: array<f32, 256>;
var<workgroup> wslot: u32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let pair = wg.y;
    if (li == 0u) { wslot = emap[p.map_offset + route[pair]]; }
    let slot = workgroupUniformLoad(&wslot);
    let in_shard = slot != 0xffffffffu && slot >= p.slot0 && slot < p.slot0 + p.N / p.exp_rows;
    let row = wg.x * ROWS + li / LANES;
    let lane = li % LANES;
    var acc = 0.0;
    if (in_shard && row < p.exp_rows) {
        let wrow = (slot - p.slot0) * p.exp_rows + row;
        let xb = (p.x_offset + (pair / p.n_used) * p.x_stride) >> 2u;
        for (var c = lane; c < p.K / CHUNK; c += LANES) { acc += chunk_dot(wrow, c, xb); }
    }
    red[li] = acc;
    workgroupBarrier();
    for (var s = LANES / 2u; s > 0u; s >>= 1u) {
        if (lane < s) { red[li] += red[li + s]; }
        workgroupBarrier();
    }
    if (in_shard && lane == 0u && row < p.exp_rows) {
        let i = pair * p.out_stride + row;
        if (p.mode == 0u) { y[i] = red[li]; } else { let g = y[i]; y[i] = g / (1.0 + exp(-g)) * red[li]; }
    }
}
"#
    )
}

/// Down projection of chosen experts, weighted and summed into the residual of each token.
pub fn gemv_id_down(fmt: GpuFormat, lanes: u32) -> String {
    format!(
        "{COMMON}{}{}\nconst LANES: u32 = {lanes}u;\nconst ROWS: u32 = {}u;\n{}",
        id_bindings(fmt),
        gemv_chunk(fmt),
        256 / lanes,
        r#"
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.y;
    let row = wg.x * ROWS + li / LANES;
    let lane = li % LANES;
    let per_shard = p.N / p.exp_rows;
    var acc = 0.0;
    for (var s = 0u; s < p.n_used; s++) {
        let pair = t * p.n_used + s;
        let slot = emap[p.map_offset + route[pair]];
        if (slot != 0xffffffffu && slot >= p.slot0 && slot < p.slot0 + per_shard && row < p.exp_rows) {
            let wrow = (slot - p.slot0) * p.exp_rows + row;
            let xb = (p.x_offset + pair * p.x_stride) >> 2u;
            var d = 0.0;
            for (var c = lane; c < p.K / CHUNK; c += LANES) { d += chunk_dot(wrow, c, xb); }
            acc += bitcast<f32>(route[p.w_offset + pair]) * d;
        }
    }
    red[li] = acc;
    workgroupBarrier();
    for (var s = LANES / 2u; s > 0u; s >>= 1u) {
        if (lane < s) { red[li] += red[li + s]; }
        workgroupBarrier();
    }
    if (lane == 0u && row < p.exp_rows) {
        let i = t * p.out_stride + p.out_offset + row;
        y[i] = y[i] + red[li];
    }
}
"#
    )
}

/// Expert routing per token: probabilities, optional group limit, top-k and weights.
pub const TOPK: &str = r#"
struct P { n_expert: u32, n_used: u32, gating: u32, norm: u32, scale: f32, has_bias: u32, n_group: u32, n_group_used: u32,
           w_offset: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> logits: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> route: array<u32>;

const NEG: f32 = -3.4e38;
var<workgroup> pr: array<f32, 1024>;
var<workgroup> sel: array<f32, 1024>;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
var<workgroup> gsum: array<f32, 64>;
var<workgroup> gkeep: array<u32, 64>;
var<workgroup> chosen: array<u32, 32>;
var<workgroup> cw: array<f32, 32>;

fn reduce_max(li: u32) {
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) { rv[li] = max(rv[li], rv[li + s]); }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.x;
    let n = p.n_expert;
    var mx = NEG;
    for (var i = li; i < n; i += 256u) { let v = logits[t * n + i]; pr[i] = v; mx = max(mx, v); }
    if (p.gating == 0u) {
        rv[li] = mx;
        workgroupBarrier();
        reduce_max(li);
        let m = rv[0];
        workgroupBarrier();
        var sm = 0.0;
        for (var i = li; i < n; i += 256u) { let e = exp(pr[i] - m); pr[i] = e; sm += e; }
        rv[li] = sm;
        workgroupBarrier();
        for (var s = 128u; s > 0u; s >>= 1u) {
            if (li < s) { rv[li] += rv[li + s]; }
            workgroupBarrier();
        }
        let total = rv[0];
        for (var i = li; i < n; i += 256u) { pr[i] = pr[i] / total; }
    } else {
        for (var i = li; i < n; i += 256u) { pr[i] = 1.0 / (1.0 + exp(-pr[i])); }
    }
    workgroupBarrier();
    for (var i = li; i < n; i += 256u) {
        var v = pr[i];
        if (p.has_bias == 1u) { v += bias[i]; }
        sel[i] = v;
    }
    workgroupBarrier();
    if (p.n_group > 1u) {
        // Each group scores the sum of its two best experts; only the best groups stay.
        let per = n / p.n_group;
        if (li < p.n_group) {
            var a = NEG;
            var b = NEG;
            for (var i = 0u; i < per; i++) {
                let v = sel[li * per + i];
                if (v > a) { b = a; a = v; } else if (v > b) { b = v; }
            }
            gsum[li] = a + b;
            gkeep[li] = 0u;
        }
        workgroupBarrier();
        if (li == 0u) {
            for (var k = 0u; k < p.n_group_used; k++) {
                var best = NEG;
                var bi = 0u;
                for (var g = 0u; g < p.n_group; g++) {
                    if (gkeep[g] == 0u && gsum[g] > best) { best = gsum[g]; bi = g; }
                }
                gkeep[bi] = 1u;
            }
        }
        workgroupBarrier();
        for (var i = li; i < n; i += 256u) {
            if (gkeep[i / per] == 0u) { sel[i] = NEG; }
        }
        workgroupBarrier();
    }
    for (var k = 0u; k < p.n_used; k++) {
        var bv = NEG;
        var bi = 0xffffffffu;
        for (var i = li; i < n; i += 256u) {
            if (sel[i] > bv) { bv = sel[i]; bi = i; }
        }
        rv[li] = bv;
        ri[li] = bi;
        workgroupBarrier();
        for (var s = 128u; s > 0u; s >>= 1u) {
            if (li < s) {
                let ov = rv[li + s];
                let oi = ri[li + s];
                if (ov > rv[li] || (ov == rv[li] && oi < ri[li])) { rv[li] = ov; ri[li] = oi; }
            }
            workgroupBarrier();
        }
        if (li == 0u) {
            let w = ri[0];
            chosen[k] = w;
            cw[k] = pr[w];
            sel[w] = NEG;
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        var sum = 0.0;
        for (var k = 0u; k < p.n_used; k++) { sum += cw[k]; }
        var div = 1.0;
        if (p.norm == 1u) { div = max(sum, 6.103515625e-5); }
        for (var k = 0u; k < p.n_used; k++) {
            route[t * p.n_used + k] = chosen[k];
            route[p.w_offset + t * p.n_used + k] = bitcast<u32>(cw[k] / div * p.scale);
        }
    }
}
"#;

/// MLA: normalise the latent half of `kv_a`, rotate its key half, and store both in the f16 cache.
pub const MLA_STORE: &str = r#"
struct P { lat: u32, rope: u32, cdim: u32, neox: u32, eps: f32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> kva: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> table: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> cache: array<u32>;
var<workgroup> red: array<f32, 256>;

fn rot(base: u32, d: u32, csb: u32) -> f32 {
    let x = kva[base + d];
    let h = p.rope / 2u;
    if (p.neox == 1u) {
        if (d < h) { let cs = table[csb + d]; return x * cs.x - kva[base + d + h] * cs.y; }
        let cs = table[csb + d - h];
        return kva[base + d - h] * cs.y + x * cs.x;
    }
    let cs = table[csb + d / 2u];
    if ((d & 1u) == 0u) { return x * cs.x - kva[base + d + 1u] * cs.y; }
    return kva[base + d - 1u] * cs.y + x * cs.x;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.x;
    let pos = step.pos0 + t;
    let base = t * p.cdim;
    var s = 0.0;
    for (var i = li; i < p.lat; i += 256u) { let v = kva[base + i]; s += v * v; }
    red[li] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (li < k) { red[li] += red[li + k]; }
        workgroupBarrier();
    }
    let scale = inverseSqrt(red[0] / f32(p.lat) + p.eps);
    let cb = pos * (p.cdim / 2u);
    for (var e = li; e < p.lat / 2u; e += 256u) {
        let d = 2u * e;
        cache[cb + e] = pack2x16float(vec2<f32>(kva[base + d] * scale * w[d], kva[base + d + 1u] * scale * w[d + 1u]));
    }
    let csb = pos * (p.rope / 2u);
    for (var e = li; e < p.rope / 2u; e += 256u) {
        let d = 2u * e;
        cache[cb + p.lat / 2u + e] = pack2x16float(vec2<f32>(rot(base + p.lat, d, csb), rot(base + p.lat, d + 1u, csb)));
    }
}
"#;

/// MLA attention: several query heads per workgroup share each read of the latent cache.
/// Scores use [absorbed query | rotated query] against [latent | rotated key]; values are the latent.
pub const MLA_ATTENTION: &str = r#"
struct P { n_head: u32, lat: u32, rope: u32, qstride: u32, qrope: u32, cdim: u32, scale: f32, heads: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> qlat: array<f32>;
@group(0) @binding(3) var<storage, read> q: array<f32>;
@group(0) @binding(4) var<storage, read> cache: array<u32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;

const HM: u32 = 4u;
const NEG: f32 = -3.4e38;
var<workgroup> qs: array<f32, 2304>;
var<workgroup> pr: array<f32, 1024>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.y;
    let h0 = wg.x * p.heads;
    let D = p.cdim;
    let L = p.lat;
    for (var i = li; i < p.heads * D; i += 256u) {
        let h = h0 + i / D;
        let d = i % D;
        if (d < L) { qs[i] = qlat[(t * p.n_head + h) * L + d]; }
        else { qs[i] = q[(t * p.n_head + h) * p.qstride + p.qrope + d - L]; }
    }
    workgroupBarrier();
    let n_keys = step.pos0 + t + 1u;
    let nd = (L + 255u) / 256u;
    var m = array<f32, 4>(NEG, NEG, NEG, NEG);
    var l = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
    for (var j0 = 0u; j0 < n_keys; j0 += 256u) {
        let j = j0 + li;
        var sc = array<f32, 4>(NEG, NEG, NEG, NEG);
        if (j < n_keys) {
            let kb = j * (D / 2u);
            var dv = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
            for (var e = 0u; e < D / 2u; e++) {
                let kk = unpack2x16float(cache[kb + e]);
                for (var h = 0u; h < HM; h++) {
                    if (h < p.heads) { dv[h] += qs[h * D + 2u * e] * kk.x + qs[h * D + 2u * e + 1u] * kk.y; }
                }
            }
            for (var h = 0u; h < HM; h++) { sc[h] = dv[h] * p.scale; }
        }
        for (var h = 0u; h < HM; h++) {
            if (h >= p.heads) { break; }
            red[li] = sc[h];
            workgroupBarrier();
            for (var r = 128u; r > 0u; r >>= 1u) {
                if (li < r) { red[li] = max(red[li], red[li + r]); }
                workgroupBarrier();
            }
            let m_new = max(m[h], red[0]);
            workgroupBarrier();
            var pj = 0.0;
            if (j < n_keys) { pj = exp(sc[h] - m_new); }
            pr[li * HM + h] = pj;
            red[li] = pj;
            workgroupBarrier();
            for (var r = 128u; r > 0u; r >>= 1u) {
                if (li < r) { red[li] += red[li + r]; }
                workgroupBarrier();
            }
            let corr = exp(m[h] - m_new);
            l[h] = l[h] * corr + red[0];
            for (var qd = 0u; qd < 4u; qd++) { acc[h * 4u + qd] *= corr; }
            m[h] = m_new;
            workgroupBarrier();
        }
        let nk = min(256u, n_keys - j0);
        for (var jj = 0u; jj < nk; jj++) {
            let vb = (j0 + jj) * (D / 2u);
            for (var qd = 0u; qd < nd; qd++) {
                let d = li + 256u * qd;
                if (d < L) {
                    let pair = unpack2x16float(cache[vb + d / 2u]);
                    let v = select(pair.x, pair.y, (d & 1u) == 1u);
                    for (var h = 0u; h < HM; h++) {
                        if (h < p.heads) { acc[h * 4u + qd] += pr[jj * HM + h] * v; }
                    }
                }
            }
        }
        workgroupBarrier();
    }
    for (var h = 0u; h < p.heads; h++) {
        for (var qd = 0u; qd < nd; qd++) {
            let d = li + 256u * qd;
            if (d < L) { o[(t * p.n_head + h0 + h) * L + d] = acc[h * 4u + qd] / l[h]; }
        }
    }
}
"#;

pub fn topk() -> String {
    format!("{COMMON}{TOPK}")
}

pub fn mla_store() -> String {
    format!("{COMMON}{MLA_STORE}")
}

pub fn mla_attention() -> String {
    format!("{COMMON}{MLA_ATTENTION}")
}

/// Prompt tokens' expert projections as matrix-matrix products: one workgroup per (64 rows,
/// 32 of the tokens routed to expert `wg.z`), using the per-expert token lists of `route_sort`.
/// Gate/up write each pair's hidden row; `down` writes each pair's output row for `moe_sum`.
pub fn gemm_id(fmt: GpuFormat) -> String {
    // Word-addressed weights, as `load8` reads them.
    let mut base = MATMUL_BINDINGS.to_string() + binding3(fmt);
    if matches!(fmt, GpuFormat::Block(_)) {
        base += "@group(0) @binding(7) var<storage, read> wv4: array<vec4<u32>>;\n";
    }
    let bindings = base
        .replace(
            "@group(0) @binding(6) var<storage, read> bias: array<f32>;",
            "@group(0) @binding(6) var<storage, read> route: array<u32>;\n@group(0) @binding(8) var<storage, read> emap: array<u32>;",
        )
        .replace("    if (p.has_bias == 1u) { v += bias[p.out_offset + r]; }\n", "")
        .replace(
        "@group(0) @binding(8) var<storage, read> emap: array<u32>;",
        "@group(0) @binding(8) var<storage, read> emap: array<u32>;\n// Per expert: token count, then list offset, then the pair indices grouped by expert.\n@group(0) @binding(9) var<storage, read> rsort: array<u32>;",
    );
    format!(
        "{COMMON}{bindings}{}{}",
        loader(fmt),
        r#"
const WS: u32 = 9u;
var<workgroup> xs: array<vec4<f32>, 256>;
var<workgroup> ws: array<vec4<f32>, 576>;
var<workgroup> pairs: array<u32, 32>;
var<workgroup> u_slot: u32;
var<workgroup> u_count: u32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let e = wg.z;
    let n_expert = p.pad1;
    if (li == 0u) { u_slot = emap[p.map_offset + e]; u_count = rsort[e]; }
    let slot = workgroupUniformLoad(&u_slot);
    let count = workgroupUniformLoad(&u_count);
    let t0 = wg.y * 32u;
    if (slot == 0xffffffffu || slot < p.slot0 || slot >= p.slot0 + p.N / p.exp_rows || t0 >= count) { return; }
    let off = rsort[n_expert + e];
    if (li < 32u) {
        var pr = 0xffffffffu;
        if (t0 + li < count) { pr = rsort[2u * n_expert + off + t0 + li]; }
        pairs[li] = pr;
    }
    workgroupBarrier();
    let row0 = wg.x * 64u;
    let wbase = (slot - p.slot0) * p.exp_rows;
    let tr = li & 15u;
    let tt = li >> 4u;
    var a00 = 0.0; var a01 = 0.0; var a10 = 0.0; var a11 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a30 = 0.0; var a31 = 0.0;
    let xp = pairs[li >> 3u];
    // Gate/up read the token's input row (x_offset 0); down reads the pair's hidden row (mode 3).
    var xrow = 0u;
    if (xp != 0xffffffffu) { xrow = select(xp / p.n_used, xp, p.mode == 3u); }
    let wr = row0 + (li >> 2u);
    let wslot = (li >> 2u) * WS + (li & 3u) * 2u;
    for (var k0 = 0u; k0 < p.K; k0 += 32u) {
        var xv = vec4<f32>(0.0);
        if (xp != 0xffffffffu) { xv = x[((xrow * p.x_stride + k0) >> 2u) + (li & 7u)]; }
        xs[li] = xv;
        var w = W8(vec4<f32>(0.0), vec4<f32>(0.0));
        if (wr < p.exp_rows) { w = load8(wbase + wr, k0 + (li & 3u) * 8u); }
        ws[wslot] = w.a;
        ws[wslot + 1u] = w.b;
        workgroupBarrier();
        for (var kv = 0u; kv < 8u; kv++) {
            let x0 = xs[tt * 8u + kv];
            let x1 = xs[(tt + 16u) * 8u + kv];
            let w0 = ws[tr * WS + kv];
            let w1 = ws[(tr + 16u) * WS + kv];
            let w2 = ws[(tr + 32u) * WS + kv];
            let w3 = ws[(tr + 48u) * WS + kv];
            a00 += dot(w0, x0); a01 += dot(w0, x1);
            a10 += dot(w1, x0); a11 += dot(w1, x1);
            a20 += dot(w2, x0); a21 += dot(w2, x1);
            a30 += dot(w3, x0); a31 += dot(w3, x1);
        }
        workgroupBarrier();
    }
    let pa = pairs[tt];
    let pb = pairs[tt + 16u];
    let r0 = row0 + tr;
    put(pa, r0, a00); put(pb, r0, a01);
    put(pa, r0 + 16u, a10); put(pb, r0 + 16u, a11);
    put(pa, r0 + 32u, a20); put(pb, r0 + 32u, a21);
    put(pa, r0 + 48u, a30); put(pb, r0 + 48u, a31);
}

fn put(pair: u32, r: u32, v: f32) {
    if (pair == 0xffffffffu || r >= p.exp_rows) { return; }
    let i = pair * p.out_stride + r;
    if (p.mode == 2u) { let g = y[i]; y[i] = g / (1.0 + exp(-g)) * v; } else { y[i] = v; }
}
"#
    )
}

/// Group a prompt's (token, expert) pairs by expert: counts, offsets and pair lists.
pub const ROUTE_SORT: &str = r#"
struct P { n_expert: u32, n_used: u32, pad0: u32, pad1: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> route: array<u32>;
@group(0) @binding(3) var<storage, read_write> rsort: array<u32>;
var<workgroup> cnt: array<atomic<u32>, 1024>;
var<workgroup> cur: array<atomic<u32>, 1024>;
var<workgroup> offs: array<u32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    let n = step.n_tokens * p.n_used;
    for (var e = li; e < p.n_expert; e += 256u) { atomicStore(&cnt[e], 0u); atomicStore(&cur[e], 0u); }
    workgroupBarrier();
    for (var i = li; i < n; i += 256u) { atomicAdd(&cnt[route[i]], 1u); }
    workgroupBarrier();
    if (li == 0u) {
        var run = 0u;
        for (var e = 0u; e < p.n_expert; e++) { offs[e] = run; run += atomicLoad(&cnt[e]); }
    }
    workgroupBarrier();
    for (var e = li; e < p.n_expert; e += 256u) { rsort[e] = atomicLoad(&cnt[e]); rsort[p.n_expert + e] = offs[e]; }
    for (var i = li; i < n; i += 256u) {
        let e = route[i];
        rsort[2u * p.n_expert + offs[e] + atomicAdd(&cur[e], 1u)] = i;
    }
}
"#;

/// Weighted sum of each token's expert outputs (from `gemm_id` down) into the residual, for the
/// experts resident in this pass.
pub const MOE_SUM: &str = r#"
struct P { n_embd: u32, n_used: u32, w_offset: u32, map_offset: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> route: array<u32>;
@group(0) @binding(3) var<storage, read> emap: array<u32>;
@group(0) @binding(4) var<storage, read> dout: array<f32>;
@group(0) @binding(5) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.y;
    let r = wg.x * 256u + li;
    if (r >= p.n_embd) { return; }
    var acc = 0.0;
    for (var s = 0u; s < p.n_used; s++) {
        let pair = t * p.n_used + s;
        if (emap[p.map_offset + route[pair]] != 0xffffffffu) {
            acc += bitcast<f32>(route[p.w_offset + pair]) * dout[pair * p.n_embd + r];
        }
    }
    x[t * p.n_embd + r] = x[t * p.n_embd + r] + acc;
}
"#;

pub fn route_sort() -> String {
    format!("{COMMON}{ROUTE_SORT}")
}

pub fn moe_sum() -> String {
    format!("{COMMON}{MOE_SUM}")
}
