//! WGSL compute kernels. Weight formats are compiled in as constants, so each
//! matrix layout gets its own specialised matrix-vector and matrix-matrix kernel.

use crate::quant::GpuFormat;

/// Bindings shared by every kernel: per-step values and per-dispatch parameters.
const COMMON: &str = r#"
struct Step { n_tokens: u32, pos0: u32, kv_len: u32, last: u32 }
@group(0) @binding(0) var<uniform> step: Step;
"#;

/// Matrix kernels: params, weight codes, weight scales, input, output, bias.
const MATMUL_BINDINGS: &str = r#"
struct P { K: u32, N: u32, out_stride: u32, out_offset: u32, mode: u32, has_bias: u32, x_offset: u32, pad: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> wcodes: array<u32>;
@group(0) @binding(3) var<storage, read> wscales: array<f32>;
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

/// `load8(row, k)` for one weight format: eight consecutive values of a row.
fn loader(fmt: GpuFormat) -> String {
    match fmt {
        GpuFormat::Q4 { offset, bias } => format!(
            r#"
const OFF: f32 = {offset}.0;
const SB: bool = {bias};
const STRIDE: u32 = {stride}u;
fn load8(r: u32, k: u32) -> W8 {{
    let e = r * p.K + k;
    let w = wcodes[e >> 3u];
    let g = (e >> 5u) * STRIDE;
    let s = wscales[g];
    let lo = vec4<f32>(vec4<u32>(w, w >> 4u, w >> 8u, w >> 12u) & vec4<u32>(15u)) - vec4<f32>(OFF);
    let hi = vec4<f32>(vec4<u32>(w >> 16u, w >> 20u, w >> 24u, w >> 28u) & vec4<u32>(15u)) - vec4<f32>(OFF);
    var o = W8(lo * s, hi * s);
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

/// Dot product of one 16-byte block of weight codes with the matching inputs.
fn gemv_chunk(fmt: GpuFormat) -> String {
    match fmt {
        GpuFormat::Q4 { offset, bias } => format!(
            r#"
const CHUNK: u32 = 32u;
fn nib(w: u32, s: u32) -> vec4<f32> {{
    return vec4<f32>(vec4<u32>(w >> s, w >> (s + 4u), w >> (s + 8u), w >> (s + 12u)) & vec4<u32>(15u));
}}
fn chunk_dot(row: u32, c: u32, xb: u32) -> f32 {{
    let g = row * (p.K >> 5u) + c;
    let w = wc4[g];
    let xi = xb + c * 8u;
    let x0 = x[xi]; let x1 = x[xi + 1u]; let x2 = x[xi + 2u]; let x3 = x[xi + 3u];
    let x4 = x[xi + 4u]; let x5 = x[xi + 5u]; let x6 = x[xi + 6u]; let x7 = x[xi + 7u];
    let d = dot(nib(w.x, 0u), x0) + dot(nib(w.x, 16u), x1) + dot(nib(w.y, 0u), x2) + dot(nib(w.y, 16u), x3)
          + dot(nib(w.z, 0u), x4) + dot(nib(w.z, 16u), x5) + dot(nib(w.w, 0u), x6) + dot(nib(w.w, 16u), x7);
    let one = vec4<f32>(1.0);
    let xs = dot(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7, one);
    var r = wscales[g * {stride}u] * (d - {offset}.0 * xs);
    if ({bias}) {{ r += wscales[g * {stride}u + 1u] * xs; }}
    return r;
}}
"#,
            stride = 1 + bias as u32
        ),
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

/// One token: sixteen rows per workgroup, sixteen lanes per row, one
/// 16-byte weight load per lane and step.
pub fn gemv(fmt: GpuFormat, lanes: u32) -> String {
    let bindings = MATMUL_BINDINGS.replace(
        "var<storage, read> wcodes: array<u32>;",
        "var<storage, read> wc4: array<vec4<u32>>;",
    );
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
        let xb = p.x_offset >> 2u;
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
        "{COMMON}{MATMUL_BINDINGS}{}{}",
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
        if (xt < T) { xv = x[((p.x_offset + xt * p.K + k0) >> 2u) + (li & 7u)]; }
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
        "{COMMON}{MATMUL_BINDINGS}{}{}",
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
struct P { n_head: u32, n_kv: u32, head_dim: u32, half_rot: u32, neox: u32, kv_dim: u32, q_dim: u32, pad: u32 }
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
        let base = t * p.q_dim + wg.x * hd;
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
