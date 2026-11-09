//! GGML tensor formats: reference dequantization and GPU repacking.
//!
//! Every quantized block is decoded into integer codes plus a scale and an
//! optional bias per group, so `value = scale * code + bias`. The arithmetic
//! matches ggml's reference dequantization. The GPU keeps the same codes in
//! word-aligned arrays, which lets WGSL read any supported format with plain
//! `u32` loads.

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q4_0: u32 = 2;
pub const Q4_1: u32 = 3;
pub const Q5_0: u32 = 6;
pub const Q5_1: u32 = 7;
pub const Q8_0: u32 = 8;
pub const Q2_K: u32 = 10;
pub const Q3_K: u32 = 11;
pub const Q4_K: u32 = 12;
pub const Q5_K: u32 = 13;
pub const Q6_K: u32 = 14;
pub const IQ2_XXS: u32 = 16;
pub const IQ2_XS: u32 = 17;
pub const IQ3_XXS: u32 = 18;
pub const IQ1_S: u32 = 19;
pub const IQ4_NL: u32 = 20;
pub const IQ3_S: u32 = 21;
pub const IQ2_S: u32 = 22;
pub const IQ4_XS: u32 = 23;
pub const IQ1_M: u32 = 29;
pub const BF16: u32 = 30;
pub const TQ1_0: u32 = 34;
pub const TQ2_0: u32 = 35;
pub const MXFP4: u32 = 39;

/// Non-linear 4-bit code book shared by IQ4_NL and IQ4_XS.
const IQ4: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// E2M1 values of MXFP4, doubled (the shared E8M0 scale is halved to match).
pub const MXFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// IQ4_NL code book as floats, for the GPU tables.
pub fn iq4_values() -> [i8; 16] {
    IQ4
}

pub fn type_name(t: u32) -> &'static str {
    match t {
        F32 => "F32",
        F16 => "F16",
        Q4_0 => "Q4_0",
        Q4_1 => "Q4_1",
        Q5_0 => "Q5_0",
        Q5_1 => "Q5_1",
        Q8_0 => "Q8_0",
        9 => "Q8_1",
        Q2_K => "Q2_K",
        Q3_K => "Q3_K",
        Q4_K => "Q4_K",
        Q5_K => "Q5_K",
        Q6_K => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        29 => "IQ1_M",
        BF16 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        _ => "unknown",
    }
}

/// (values per block, bytes per block) for supported formats.
pub fn block_info(t: u32) -> Option<(usize, usize)> {
    Some(match t {
        F32 => (1, 4),
        F16 | BF16 => (1, 2),
        Q4_0 => (32, 18),
        Q4_1 => (32, 20),
        Q5_0 => (32, 22),
        Q5_1 => (32, 24),
        Q8_0 => (32, 34),
        Q2_K => (256, 84),
        Q3_K => (256, 110),
        Q4_K => (256, 144),
        Q5_K => (256, 176),
        Q6_K => (256, 210),
        IQ4_NL => (32, 18),
        IQ4_XS => (256, 136),
        IQ2_XXS => (256, 66),
        IQ2_XS => (256, 74),
        IQ2_S => (256, 82),
        IQ3_XXS => (256, 98),
        IQ3_S => (256, 110),
        IQ1_S => (256, 50),
        IQ1_M => (256, 56),
        TQ1_0 => (256, 54),
        TQ2_0 => (256, 66),
        MXFP4 => (32, 17),
        _ => return None,
    })
}

pub fn row_bytes(t: u32, k: usize) -> Result<usize, String> {
    let (vals, bytes) =
        block_info(t).ok_or_else(|| format!("{} tensors are not supported yet", type_name(t)))?;
    if k % vals != 0 {
        return Err(format!(
            "row of {k} values is not a whole number of {} blocks",
            type_name(t)
        ));
    }
    Ok(k / vals * bytes)
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Round-to-nearest-even conversion used for F32/BF16 matrices stored as F16 on the GPU.
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut out = m >> shift;
        if rem > half || (rem == half && out & 1 == 1) {
            out += 1;
        }
        return sign | out as u16;
    }
    let mut out = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && out & 1 == 1) {
        out += 1;
    }
    sign | out as u16
}

fn half(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

/// Codes and per-group factors for one decoded block.
struct Block {
    codes: [i8; 256],
    scales: [f32; 16],
    biases: [f32; 16],
}

impl Block {
    fn new() -> Self {
        Block {
            codes: [0; 256],
            scales: [0.0; 16],
            biases: [0.0; 16],
        }
    }
}

/// Group size of the decoded representation.
fn group_size(t: u32) -> usize {
    match t {
        Q2_K | Q3_K | Q6_K => 16,
        _ => 32,
    }
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn decode_block(t: u32, b: &[u8], out: &mut Block) {
    let c = &mut out.codes;
    match t {
        Q4_0 => {
            out.scales[0] = half(b, 0);
            out.biases[0] = 0.0;
            for j in 0..16 {
                c[j] = (b[2 + j] & 0xf) as i8 - 8;
                c[j + 16] = (b[2 + j] >> 4) as i8 - 8;
            }
        }
        Q4_1 => {
            out.scales[0] = half(b, 0);
            out.biases[0] = half(b, 2);
            for j in 0..16 {
                c[j] = (b[4 + j] & 0xf) as i8;
                c[j + 16] = (b[4 + j] >> 4) as i8;
            }
        }
        Q5_0 | Q5_1 => {
            let (qh_at, qs_at) = if t == Q5_0 { (2, 6) } else { (4, 8) };
            out.scales[0] = half(b, 0);
            out.biases[0] = if t == Q5_1 { half(b, 2) } else { 0.0 };
            let qh = u32::from_le_bytes(b[qh_at..qh_at + 4].try_into().unwrap());
            let sub = if t == Q5_0 { 16 } else { 0 };
            for j in 0..16 {
                let h0 = ((qh >> j) << 4) & 0x10;
                let h1 = (qh >> (j + 12)) & 0x10;
                c[j] = (((b[qs_at + j] & 0xf) as u32 | h0) as i32 - sub) as i8;
                c[j + 16] = (((b[qs_at + j] >> 4) as u32 | h1) as i32 - sub) as i8;
            }
        }
        Q8_0 => {
            out.scales[0] = half(b, 0);
            out.biases[0] = 0.0;
            for j in 0..32 {
                c[j] = b[2 + j] as i8;
            }
        }
        Q2_K => {
            let scales = &b[0..16];
            let qs = &b[16..80];
            let d = half(b, 80);
            let min = half(b, 82);
            let mut y = 0;
            let mut is = 0;
            for n in 0..2 {
                let q = &qs[n * 32..n * 32 + 32];
                for shift in [0u32, 2, 4, 6] {
                    for half_index in 0..2 {
                        let sc = scales[is];
                        out.scales[is] = d * (sc & 0xf) as f32;
                        out.biases[is] = -(min * (sc >> 4) as f32);
                        for l in 0..16 {
                            c[y] = ((q[l + 16 * half_index] >> shift) & 3) as i8;
                            y += 1;
                        }
                        is += 1;
                    }
                }
            }
        }
        Q3_K => {
            let hm = &b[0..32];
            let qs = &b[32..96];
            let raw = &b[96..108];
            let d_all = half(b, 108);
            let mut aux = [0u32; 4];
            for (i, chunk) in raw.chunks(4).enumerate() {
                aux[i] = u32::from_le_bytes(chunk.try_into().unwrap());
            }
            let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
            aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
            aux[0] = (aux[0] & k2) | ((tmp & k1) << 4);
            aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
            let mut sc = [0i8; 16];
            for i in 0..16 {
                sc[i] = (aux[i / 4] >> (8 * (i % 4))) as u8 as i8;
            }
            let mut y = 0;
            let mut is = 0;
            let mut m = 1u8;
            for n in 0..2 {
                let q = &qs[n * 32..n * 32 + 32];
                for shift in [0u32, 2, 4, 6] {
                    for half_index in 0..2 {
                        out.scales[is] = d_all * (sc[is] as i32 - 32) as f32;
                        out.biases[is] = 0.0;
                        for l in 0..16 {
                            let idx = l + 16 * half_index;
                            let low = ((q[idx] >> shift) & 3) as i8;
                            c[y] = low - if hm[idx] & m != 0 { 0 } else { 4 };
                            y += 1;
                        }
                        is += 1;
                    }
                    m = m.wrapping_shl(1);
                }
            }
        }
        Q4_K | Q5_K => {
            let d = half(b, 0);
            let min = half(b, 2);
            let scales = &b[4..16];
            let (qh, ql) = if t == Q5_K {
                (&b[16..48], &b[48..176])
            } else {
                (&b[0..0], &b[16..144])
            };
            let mut y = 0;
            let mut is = 0;
            let (mut u1, mut u2) = (1u8, 2u8);
            for j in 0..4 {
                let q = &ql[j * 32..j * 32 + 32];
                let (s0, m0) = scale_min_k4(is, scales);
                let (s1, m1) = scale_min_k4(is + 1, scales);
                out.scales[is] = d * s0 as f32;
                out.biases[is] = -(min * m0 as f32);
                out.scales[is + 1] = d * s1 as f32;
                out.biases[is + 1] = -(min * m1 as f32);
                for l in 0..32 {
                    let hi = if t == Q5_K && qh[l] & u1 != 0 { 16 } else { 0 };
                    c[y + l] = ((q[l] & 0xf) + hi) as i8;
                }
                for l in 0..32 {
                    let hi = if t == Q5_K && qh[l] & u2 != 0 { 16 } else { 0 };
                    c[y + 32 + l] = ((q[l] >> 4) + hi) as i8;
                }
                y += 64;
                is += 2;
                u1 = u1.wrapping_shl(2);
                u2 = u2.wrapping_shl(2);
            }
        }
        Q6_K => {
            let ql_all = &b[0..128];
            let qh_all = &b[128..192];
            let sc = &b[192..208];
            let d = half(b, 208);
            for n in 0..2 {
                let ql = &ql_all[n * 64..n * 64 + 64];
                let qh = &qh_all[n * 32..n * 32 + 32];
                let base = n * 128;
                for l in 0..32 {
                    let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                    c[base + l] = q1;
                    c[base + l + 32] = q2;
                    c[base + l + 64] = q3;
                    c[base + l + 96] = q4;
                }
                // Groups of 16 values, in element order, use scales sc[8n + g].
                for g in 0..8 {
                    out.scales[n * 8 + g] = d * (sc[n * 8 + g] as i8) as f32;
                    out.biases[n * 8 + g] = 0.0;
                }
            }
        }
        IQ4_NL => {
            out.scales[0] = half(b, 0);
            out.biases[0] = 0.0;
            for j in 0..16 {
                c[j] = IQ4[(b[2 + j] & 0xf) as usize];
                c[j + 16] = IQ4[(b[2 + j] >> 4) as usize];
            }
        }
        IQ4_XS => {
            let d = half(b, 0);
            let scales_h = u16::from_le_bytes([b[2], b[3]]) as u32;
            let scales_l = &b[4..8];
            let qs = &b[8..136];
            for ib in 0..8 {
                let ls = ((scales_l[ib / 2] as u32 >> (4 * (ib % 2))) & 0xf)
                    | (((scales_h >> (2 * ib)) & 3) << 4);
                out.scales[ib] = d * (ls as i32 - 32) as f32;
                out.biases[ib] = 0.0;
                let q = &qs[ib * 16..ib * 16 + 16];
                for j in 0..16 {
                    c[ib * 32 + j] = IQ4[(q[j] & 0xf) as usize];
                    c[ib * 32 + j + 16] = IQ4[(q[j] >> 4) as usize];
                }
            }
        }
        _ => unreachable!("decode_block called for {t}"),
    }
}

fn u16le(b: &[u8], at: usize) -> u32 {
    u16::from_le_bytes([b[at], b[at + 1]]) as u32
}

fn u32le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

/// Sign bits of an IQ2/IQ3 group: seven stored bits plus even parity in the eighth.
fn ksign(i: u32) -> u32 {
    i | ((i.count_ones() & 1) << 7)
}

fn signed(signs: u32, j: usize) -> f32 {
    if (signs >> j) & 1 == 1 {
        -1.0
    } else {
        1.0
    }
}

/// Half of an E8M0 exponent scale, as ggml's MXFP4 uses with doubled E2M1 values.
pub fn e8m0_half(e: u8) -> f32 {
    f32::from_bits(if e < 2 {
        0x0020_0000u32 << e
    } else {
        ((e as u32) - 1) << 23
    })
}

/// Formats decoded straight to floats: code books, ternary and MXFP4.
fn is_direct(t: u32) -> bool {
    matches!(
        t,
        IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS | IQ3_S | IQ1_S | IQ1_M | TQ1_0 | TQ2_0 | MXFP4
    )
}

/// One block of a direct format, following ggml's reference dequantizers.
fn decode_direct(t: u32, b: &[u8], y: &mut [f32]) {
    use crate::iq_tables::*;
    let byte = |g: u64, j: usize| ((g >> (8 * j)) & 255) as f32;
    match t {
        IQ2_XXS => {
            let d = half(b, 0);
            for ib in 0..8 {
                let a0 = u32le(b, 2 + 8 * ib);
                let a1 = u32le(b, 6 + 8 * ib);
                let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
                for l in 0..4 {
                    let grid = IQ2XXS_GRID[((a0 >> (8 * l)) & 255) as usize];
                    let signs = ksign((a1 >> (7 * l)) & 127);
                    for j in 0..8 {
                        y[ib * 32 + l * 8 + j] = db * byte(grid, j) * signed(signs, j);
                    }
                }
            }
        }
        IQ2_XS => {
            let d = half(b, 0);
            for ib in 0..8 {
                let sc = b[66 + ib] as u32;
                for l in 0..4 {
                    let q = u16le(b, 2 + 2 * (4 * ib + l));
                    let db = d * (0.5 + ((sc >> (4 * (l / 2))) & 15) as f32) * 0.25;
                    let grid = IQ2XS_GRID[(q & 511) as usize];
                    let signs = ksign(q >> 9);
                    for j in 0..8 {
                        y[ib * 32 + l * 8 + j] = db * byte(grid, j) * signed(signs, j);
                    }
                }
            }
        }
        IQ2_S => {
            let d = half(b, 0);
            let (qs, qh, sc) = (&b[2..66], &b[66..74], &b[74..82]);
            for ib in 0..8 {
                for l in 0..4 {
                    let idx = qs[4 * ib + l] as u32 | (((qh[ib] as u32) << (8 - 2 * l)) & 0x300);
                    let signs = qs[32 + 4 * ib + l] as u32;
                    let db = d * (0.5 + ((sc[ib] as u32 >> (4 * (l / 2))) & 15) as f32) * 0.25;
                    let grid = IQ2S_GRID[idx as usize];
                    for j in 0..8 {
                        y[ib * 32 + l * 8 + j] = db * byte(grid, j) * signed(signs, j);
                    }
                }
            }
        }
        IQ3_XXS => {
            let d = half(b, 0);
            let (qs, sas) = (&b[2..66], &b[66..98]);
            for ib in 0..8 {
                let aux = u32le(sas, 4 * ib);
                let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let signs = ksign((aux >> (7 * l)) & 127);
                    let g1 = IQ3XXS_GRID[qs[8 * ib + 2 * l] as usize] as u64;
                    let g2 = IQ3XXS_GRID[qs[8 * ib + 2 * l + 1] as usize] as u64;
                    for j in 0..4 {
                        y[ib * 32 + l * 8 + j] = db * byte(g1, j) * signed(signs, j);
                        y[ib * 32 + l * 8 + j + 4] = db * byte(g2, j) * signed(signs, j + 4);
                    }
                }
            }
        }
        IQ3_S => {
            let d = half(b, 0);
            let (qs, qh, signs, sc) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
            for ib in 0..8 {
                let db = d * (1 + 2 * ((sc[ib / 2] as u32 >> (4 * (ib % 2))) & 15)) as f32;
                let h = qh[ib] as u32;
                for l in 0..4 {
                    let i1 = qs[8 * ib + 2 * l] as u32 | ((h << (8 - 2 * l)) & 256);
                    let i2 = qs[8 * ib + 2 * l + 1] as u32 | ((h << (7 - 2 * l)) & 256);
                    let (g1, g2) = (IQ3S_GRID[i1 as usize] as u64, IQ3S_GRID[i2 as usize] as u64);
                    let s = signs[4 * ib + l] as u32;
                    for j in 0..4 {
                        y[ib * 32 + l * 8 + j] = db * byte(g1, j) * signed(s, j);
                        y[ib * 32 + l * 8 + j + 4] = db * byte(g2, j) * signed(s, j + 4);
                    }
                }
            }
        }
        IQ1_S => {
            let d = half(b, 0);
            let qs = &b[2..34];
            for ib in 0..8 {
                let qh = u16le(b, 34 + 2 * ib);
                let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
                let delta = if qh & 0x8000 != 0 { -0.125 } else { 0.125 };
                for l in 0..4 {
                    let idx = qs[4 * ib + l] as u32 | (((qh >> (3 * l)) & 7) << 8);
                    let grid = IQ1S_GRID[idx as usize];
                    for j in 0..8 {
                        let g = (grid >> (8 * j)) as u8 as i8 as f32;
                        y[ib * 32 + l * 8 + j] = dl * (g + delta);
                    }
                }
            }
        }
        IQ1_M => {
            let (qs, qh) = (&b[0..32], &b[32..48]);
            let sc = [u16le(b, 48), u16le(b, 50), u16le(b, 52), u16le(b, 54)];
            let bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0xf0) | ((sc[2] >> 4) & 0xf00) | (sc[3] & 0xf000);
            let d = f16_to_f32(bits as u16);
            for ib in 0..8 {
                let s = sc[ib / 2];
                let dl1 = d * (2 * ((s >> (6 * (ib % 2))) & 7) + 1) as f32;
                let dl2 = d * (2 * ((s >> (6 * (ib % 2) + 3)) & 7) + 1) as f32;
                let (h0, h1) = (qh[2 * ib] as u32, qh[2 * ib + 1] as u32);
                let idx = [
                    qs[4 * ib] as u32 | ((h0 << 8) & 0x700),
                    qs[4 * ib + 1] as u32 | ((h0 << 4) & 0x700),
                    qs[4 * ib + 2] as u32 | ((h1 << 8) & 0x700),
                    qs[4 * ib + 3] as u32 | ((h1 << 4) & 0x700),
                ];
                let flag = [h0 & 0x08, h0 & 0x80, h1 & 0x08, h1 & 0x80];
                for l in 0..4 {
                    let dl = if l < 2 { dl1 } else { dl2 };
                    let delta = if flag[l] != 0 { -0.125 } else { 0.125 };
                    let grid = IQ1S_GRID[idx[l] as usize];
                    for j in 0..8 {
                        let g = (grid >> (8 * j)) as u8 as i8 as f32;
                        y[ib * 32 + l * 8 + j] = dl * (g + delta);
                    }
                }
            }
        }
        MXFP4 => {
            let d = e8m0_half(b[0]);
            for j in 0..16 {
                y[j] = MXFP4_VALUES[(b[1 + j] & 15) as usize] as f32 * d;
                y[j + 16] = MXFP4_VALUES[(b[1 + j] >> 4) as usize] as f32 * d;
            }
        }
        TQ1_0 => {
            let d = half(b, 52);
            let pow3 = [1u8, 3, 9, 27, 81, 243];
            let tern = |q: u8, n: usize| (((q.wrapping_mul(pow3[n]) as u16) * 3) >> 8) as f32 - 1.0;
            let mut o = 0;
            for n in 0..5 {
                for m in 0..32 {
                    y[o] = tern(b[m], n) * d;
                    o += 1;
                }
            }
            for n in 0..5 {
                for m in 0..16 {
                    y[o] = tern(b[32 + m], n) * d;
                    o += 1;
                }
            }
            for n in 0..4 {
                for j in 0..4 {
                    y[o] = tern(b[48 + j], n) * d;
                    o += 1;
                }
            }
        }
        TQ2_0 => {
            let d = half(b, 64);
            for jb in [0usize, 32] {
                for l in 0..4 {
                    for m in 0..32 {
                        let q = (b[jb + m] >> (2 * l)) & 3;
                        y[jb * 4 + l * 32 + m] = (q as f32 - 1.0) * d;
                    }
                }
            }
        }
        _ => unreachable!("decode_direct called for {t}"),
    }
}

fn float_value(t: u32, b: &[u8], i: usize) -> f32 {
    match t {
        F32 => f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()),
        F16 => half(b, i * 2),
        BF16 => f32::from_bits((u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]) as u32) << 16),
        _ => unreachable!(),
    }
}

fn is_float(t: u32) -> bool {
    matches!(t, F32 | F16 | BF16)
}

/// Reference dequantization of `out.len()` values.
pub fn dequantize(t: u32, bytes: &[u8], out: &mut [f32]) -> Result<(), String> {
    let (vals, size) =
        block_info(t).ok_or_else(|| format!("{} tensors are not supported yet", type_name(t)))?;
    if out.len() % vals != 0 || bytes.len() < out.len() / vals * size {
        return Err(format!("{} data has the wrong length", type_name(t)));
    }
    if is_float(t) {
        for (i, o) in out.iter_mut().enumerate() {
            *o = float_value(t, bytes, i);
        }
        return Ok(());
    }
    if is_direct(t) {
        for (bi, chunk) in out.chunks_mut(vals).enumerate() {
            decode_direct(t, &bytes[bi * size..(bi + 1) * size], chunk);
        }
        return Ok(());
    }
    let g = group_size(t);
    let mut block = Block::new();
    for (bi, chunk) in out.chunks_mut(vals).enumerate() {
        decode_block(t, &bytes[bi * size..(bi + 1) * size], &mut block);
        for (i, o) in chunk.iter_mut().enumerate() {
            *o = block.scales[i / g] * block.codes[i] as f32 + block.biases[i / g];
        }
    }
    Ok(())
}

/// How to trade GPU memory for speed when choosing weight layouts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Layout {
    /// Expanded layouts wherever a format has one: fastest, up to a third larger than the file.
    #[default]
    Fast,
    /// File-sized blocks, except where the expanded layout costs under 15% more memory.
    Compact,
    /// File-sized blocks for every quantized format (tests use it).
    Blocks,
}

/// The layout a weight matrix uses on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuFormat {
    // Expanded layouts: codes in planes with one f32 scale (and bias) per group. Fastest to
    // read, but larger than the file; used when the model fits the GPU comfortably.
    /// Unsigned codes split into a plane of 4-bit nibbles (eight per word) and,
    /// for 5- and 6-bit formats, a plane of `high` extra bits per value:
    /// `code = nibble | high_bits << 4`, `value = scale * (code - offset) + bias`.
    Q4 {
        group: u8,
        high: u8,
        offset: u8,
        bias: bool,
    },
    /// Signed 8-bit codes, four per word: `value = scale * code + bias`.
    Q8 { group: u8, bias: bool },
    /// Two IEEE half floats per word, for float tensors.
    F16,
    /// GGUF blocks of this type with their fields moved to word-aligned offsets. Kernels
    /// decode them in place, so a quantized matrix takes the same memory on the GPU as in the file.
    Block(u32),
}

/// (values, 32-bit words, GGUF blocks) of one GPU block. Formats with 32-value blocks pair
/// two of them so that their 18-, 22- and 34-byte blocks fill whole words without padding.
pub fn gpu_block(t: u32) -> Option<(usize, usize, usize)> {
    Some(match t {
        Q4_0 | IQ4_NL | MXFP4 => (64, 9, 2),
        Q5_0 => (64, 11, 2),
        Q8_0 => (64, 17, 2),
        Q4_1 => (32, 5, 1),
        Q5_1 => (32, 6, 1),
        Q2_K | IQ2_S => (256, 21, 1),
        Q3_K | IQ3_S => (256, 28, 1),
        Q4_K => (256, 36, 1),
        Q5_K => (256, 44, 1),
        Q6_K => (256, 53, 1),
        IQ4_XS => (256, 34, 1),
        IQ2_XXS | TQ2_0 => (256, 17, 1),
        IQ2_XS => (256, 19, 1),
        IQ3_XXS => (256, 25, 1),
        IQ1_S => (256, 13, 1),
        IQ1_M | TQ1_0 => (256, 14, 1),
        _ => return None,
    })
}

/// (source offset, length, destination offset) of each field of GGUF block `h` of a GPU block.
fn field_moves(t: u32, h: usize) -> Vec<(usize, usize, usize)> {
    match t {
        Q4_0 | IQ4_NL => vec![(2, 16, 16 * h), (0, 2, 32 + 2 * h)],
        MXFP4 => vec![(1, 16, 16 * h), (0, 1, 32 + h)],
        Q5_0 => vec![(6, 16, 16 * h), (2, 4, 32 + 4 * h), (0, 2, 40 + 2 * h)],
        Q8_0 => vec![(2, 32, 32 * h), (0, 2, 64 + 2 * h)],
        IQ2_XXS => vec![(2, 64, 0), (0, 2, 64)],
        IQ2_XS => vec![(2, 64, 0), (66, 8, 64), (0, 2, 72)],
        IQ2_S => vec![(2, 64, 0), (66, 8, 64), (74, 8, 72), (0, 2, 80)],
        IQ3_XXS => vec![(2, 96, 0), (0, 2, 96)],
        IQ3_S => vec![(2, 64, 0), (66, 8, 64), (74, 32, 72), (106, 4, 104), (0, 2, 108)],
        IQ1_S => vec![(2, 32, 0), (34, 16, 32), (0, 2, 48)],
        _ => vec![(0, block_info(t).unwrap().1, 0)],
    }
}

/// GPU bytes of one matrix: nibble or code plane, extra-bit plane, scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Planes {
    pub low: u64,
    pub high: u64,
    pub scales: u64,
}

impl GpuFormat {
    /// The fastest layout for a type: expanded where one exists, else blocks.
    pub fn for_type(t: u32) -> Option<GpuFormat> {
        Self::expanded(t).or_else(|| Self::compact(t))
    }

    /// The layout that takes the same memory as the file (half floats for float tensors).
    pub fn compact(t: u32) -> Option<GpuFormat> {
        match t {
            F32 | F16 | BF16 => Some(GpuFormat::F16),
            _ => gpu_block(t).map(|_| GpuFormat::Block(t)),
        }
    }

    /// Expanded codes and scales, for the formats where that costs at most a third more memory.
    pub fn expanded(t: u32) -> Option<GpuFormat> {
        let q4 = |group, high, offset, bias| GpuFormat::Q4 {
            group,
            high,
            offset,
            bias,
        };
        Some(match t {
            Q4_0 => q4(32, 0, 8, false),
            Q4_1 | Q4_K => q4(32, 0, 0, true),
            Q5_0 => q4(32, 1, 16, false),
            Q5_1 | Q5_K => q4(32, 1, 0, true),
            Q6_K => q4(16, 2, 32, false),
            Q8_0 => GpuFormat::Q8 {
                group: 32,
                bias: false,
            },
            F32 | F16 | BF16 => GpuFormat::F16,
            _ => return None,
        })
    }

    /// The layout for a type under a layout policy.
    pub fn choose(t: u32, layout: Layout) -> Option<GpuFormat> {
        let fast = Self::for_type(t)?;
        let small = Self::compact(t)?;
        match layout {
            Layout::Fast => return Some(fast),
            Layout::Blocks => return Some(small),
            Layout::Compact => {}
        }
        let (fb, sb) = (fast.bytes(256, 1), small.bytes(256, 1));
        Some(if fb * 100 <= sb * 115 { fast } else { small })
    }

    /// True for the expanded layouts, which keep scales in a second buffer.
    pub fn has_scales(self) -> bool {
        matches!(self, GpuFormat::Q4 { .. } | GpuFormat::Q8 { .. })
    }

    pub fn group(self) -> usize {
        match self {
            GpuFormat::Q4 { group, .. } | GpuFormat::Q8 { group, .. } => group as usize,
            _ => 0,
        }
    }

    /// Floats stored per group: scale, then bias when present.
    pub fn scale_stride(self) -> usize {
        match self {
            GpuFormat::Q4 { bias, .. } | GpuFormat::Q8 { bias, .. } => 1 + bias as usize,
            _ => 0,
        }
    }

    pub fn key(self) -> String {
        match self {
            GpuFormat::Q4 {
                group,
                high,
                offset,
                bias,
            } => format!("q4g{group}h{high}o{offset}b{}", bias as u8),
            GpuFormat::Q8 { group, bias } => format!("q8g{group}b{}", bias as u8),
            GpuFormat::F16 => "f16".into(),
            GpuFormat::Block(t) => type_name(t).to_ascii_lowercase(),
        }
    }

    /// GPU bytes for an expanded matrix of `rows` x `k` values, by plane.
    pub fn planes(self, k: usize, rows: usize) -> Planes {
        let n = (k * rows) as u64;
        match self {
            GpuFormat::Q4 { group, high, .. } => Planes {
                low: n / 2,
                high: n * high as u64 / 8,
                scales: n / group as u64 * self.scale_stride() as u64 * 4,
            },
            GpuFormat::Q8 { group, .. } => Planes {
                low: n,
                high: 0,
                scales: n / group as u64 * self.scale_stride() as u64 * 4,
            },
            GpuFormat::F16 | GpuFormat::Block(_) => Planes {
                low: self.bytes(k, rows),
                high: 0,
                scales: 0,
            },
        }
    }

    /// (code bytes including any extra-bit plane, scale bytes).
    pub fn sizes(self, k: usize, rows: usize) -> (u64, u64) {
        let p = self.planes(k, rows);
        (p.low + p.high, p.scales)
    }

    /// GPU bytes of one row of `k` values, all planes together.
    pub fn row_bytes(self, k: usize) -> u64 {
        self.bytes(k, 1)
    }

    /// GPU bytes for a matrix of `rows` x `k` values, all planes together.
    pub fn bytes(self, k: usize, rows: usize) -> u64 {
        match self {
            GpuFormat::F16 => (k * rows) as u64 * 2,
            GpuFormat::Block(t) => {
                let (vals, words, _) = gpu_block(t).unwrap();
                (k.div_ceil(vals) * words * 4 * rows) as u64
            }
            _ => {
                let (c, s) = self.sizes(k, rows);
                c + s
            }
        }
    }
}

/// True when a format's compact GPU layout is byte-for-byte the file layout, so streamed
/// weights can go to the GPU without being rewritten.
pub fn gpu_is_file_layout(t: u32) -> bool {
    match (gpu_block(t), block_info(t)) {
        (Some((_, words, 1)), Some((_, size))) => words * 4 == size && field_moves(t, 0) == vec![(0, size, 0)],
        _ => false,
    }
}

/// `to_gpu` as bytes, borrowing the input when the layouts already match.
pub fn to_gpu_bytes(t: u32, bytes: &[u8], k: usize, rows: usize) -> Result<std::borrow::Cow<'_, [u8]>, String> {
    if gpu_is_file_layout(t) && k % block_info(t).unwrap().0 == 0 {
        if bytes.len() != row_bytes(t, k)? * rows {
            return Err("weight rows have the wrong length".into());
        }
        return Ok(std::borrow::Cow::Borrowed(bytes));
    }
    if let (Some(GpuFormat::Block(_)), Some((vals, size))) = (GpuFormat::compact(t), block_info(t)) {
        // Copy each block's fields straight to their GPU offsets.
        let rb = row_bytes(t, k)?;
        if bytes.len() != rb * rows {
            return Err("weight rows have the wrong length".into());
        }
        let (gvals, words, pair) = gpu_block(t).unwrap();
        let nb = k / vals;
        let per_row = k.div_ceil(gvals);
        let gb_bytes = words * 4;
        let moves: Vec<_> = (0..pair).map(|h| field_moves(t, h)).collect();
        let mut out = vec![0u8; rows * per_row * gb_bytes];
        for r in 0..rows {
            let src_row = &bytes[r * rb..(r + 1) * rb];
            let dst_row = &mut out[r * per_row * gb_bytes..(r + 1) * per_row * gb_bytes];
            for bi in 0..nb {
                let (gb, h) = (bi / pair, bi % pair);
                let src = &src_row[bi * size..(bi + 1) * size];
                let dst = &mut dst_row[gb * gb_bytes..(gb + 1) * gb_bytes];
                for &(from, len, to) in &moves[h] {
                    dst[to..to + len].copy_from_slice(&src[from..from + len]);
                }
            }
        }
        return Ok(std::borrow::Cow::Owned(out));
    }
    let w = to_gpu(t, bytes, k, rows)?;
    Ok(std::borrow::Cow::Owned(w.iter().flat_map(|v| v.to_le_bytes()).collect()))
}

/// Convert whole rows of a GGML tensor into their GPU layout (32-bit words).
pub fn to_gpu(t: u32, bytes: &[u8], k: usize, rows: usize) -> Result<Vec<u32>, String> {
    let fmt = GpuFormat::compact(t)
        .ok_or_else(|| format!("{} tensors are not supported yet", type_name(t)))?;
    let rb = row_bytes(t, k)?;
    if bytes.len() != rb * rows {
        return Err(format!(
            "expected {} bytes for {rows} rows, got {}",
            rb * rows,
            bytes.len()
        ));
    }
    if k % 8 != 0 {
        return Err(format!("row length {k} does not fit the GPU layout"));
    }
    let mut out = vec![0u32; (fmt.bytes(k, rows) / 4) as usize];
    match fmt {
        GpuFormat::Q4 { .. } | GpuFormat::Q8 { .. } => unreachable!(),
        GpuFormat::F16 => {
            for i in 0..k * rows {
                let h = match t {
                    F16 => u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]),
                    _ => f32_to_f16(float_value(t, bytes, i)),
                };
                out[i / 2] |= (h as u32) << (16 * (i % 2));
            }
        }
        GpuFormat::Block(_) => {
            let (vals, size) = block_info(t).unwrap();
            let (gvals, words, pair) = gpu_block(t).unwrap();
            let nb = k / vals;
            let per_row = k.div_ceil(gvals);
            let moves: Vec<_> = (0..pair).map(|h| field_moves(t, h)).collect();
            let mut block = vec![0u8; words * 4];
            for r in 0..rows {
                for gb in 0..per_row {
                    block.fill(0);
                    for (h, mv) in moves.iter().enumerate() {
                        let bi = gb * pair + h;
                        if bi >= nb {
                            continue;
                        }
                        let src = &bytes[r * rb + bi * size..r * rb + (bi + 1) * size];
                        for &(from, len, to) in mv {
                            block[to..to + len].copy_from_slice(&src[from..from + len]);
                        }
                    }
                    let at = (r * per_row + gb) * words;
                    for (w, c) in out[at..at + words].iter_mut().zip(block.chunks_exact(4)) {
                        *w = u32::from_le_bytes(c.try_into().unwrap());
                    }
                }
            }
        }
    }
    Ok(out)
}

pub struct Packed {
    /// Nibbles, 8-bit codes or halves.
    pub codes: Vec<u32>,
    /// Extra high bits of 5- and 6-bit codes, `high` bits per value.
    pub high: Vec<u32>,
    pub scales: Vec<f32>,
}

/// Convert whole rows of a GGML tensor into its GPU layout.
pub fn repack(t: u32, bytes: &[u8], k: usize, rows: usize) -> Result<Packed, String> {
    let fmt = GpuFormat::expanded(t)
        .ok_or_else(|| format!("{} has no expanded GPU layout", type_name(t)))?;
    let rb = row_bytes(t, k)?;
    if bytes.len() != rb * rows {
        return Err(format!(
            "expected {} bytes for {rows} rows, got {}",
            rb * rows,
            bytes.len()
        ));
    }
    let n = k * rows;
    if n % 8 != 0 || (fmt != GpuFormat::F16 && k % 32 != 0) {
        return Err(format!("row length {k} does not fit the GPU layout"));
    }
    let planes = fmt.planes(k, rows);
    let mut codes = vec![0u32; (planes.low / 4) as usize];
    let mut high = vec![0u32; (planes.high / 4) as usize];
    if fmt == GpuFormat::F16 {
        for i in 0..n {
            let h = match t {
                F16 => u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]),
                _ => f32_to_f16(float_value(t, bytes, i)),
            };
            codes[i / 2] |= (h as u32) << (16 * (i % 2));
        }
        return Ok(Packed {
            codes,
            high,
            scales: Vec::new(),
        });
    }
    let (vals, size) = block_info(t).unwrap();
    let src_group = group_size(t);
    debug_assert_eq!(src_group, fmt.group());
    let stride = fmt.scale_stride();
    let mut scales = vec![0f32; n / src_group * stride];
    let mut block = Block::new();
    for bi in 0..n / vals {
        decode_block(t, &bytes[bi * size..(bi + 1) * size], &mut block);
        let base = bi * vals;
        for g in 0..vals / src_group {
            let gi = base / src_group + g;
            scales[gi * stride] = block.scales[g];
            if stride == 2 {
                scales[gi * stride + 1] = block.biases[g];
            }
        }
        match fmt {
            GpuFormat::Q4 {
                high: hb, offset, ..
            } => {
                let mask = (1u32 << (4 + hb)) - 1;
                for i in 0..vals {
                    let q = (block.codes[i] as i32 + offset as i32) as u32;
                    debug_assert_eq!(q & !mask, 0, "{} code out of range", type_name(t));
                    let e = base + i;
                    codes[e / 8] |= (q & 0xf) << (4 * (e % 8));
                    match hb {
                        1 => high[e / 32] |= ((q >> 4) & 1) << (e % 32),
                        2 => high[e / 16] |= ((q >> 4) & 3) << (2 * (e % 16)),
                        _ => {}
                    }
                }
            }
            GpuFormat::Q8 { .. } => {
                for i in 0..vals {
                    let e = base + i;
                    codes[e / 4] |= (block.codes[i] as u8 as u32) << (8 * (e % 4));
                }
            }
            GpuFormat::F16 | GpuFormat::Block(_) => unreachable!(),
        }
    }
    Ok(Packed {
        codes,
        high,
        scales,
    })
}

/// Expand a GPU-packed matrix back to floats (used by tests to check the layout).
pub fn unpack(fmt: GpuFormat, p: &Packed, n: usize) -> Vec<f32> {
    (0..n)
        .map(|e| match fmt {
            GpuFormat::F16 => f16_to_f32((p.codes[e / 2] >> (16 * (e % 2))) as u16),
            GpuFormat::Q4 {
                group,
                high,
                offset,
                bias,
            } => {
                let g = e / group as usize * (1 + bias as usize);
                let mut q = (p.codes[e / 8] >> (4 * (e % 8))) & 0xf;
                q |= match high {
                    1 => ((p.high[e / 32] >> (e % 32)) & 1) << 4,
                    2 => ((p.high[e / 16] >> (2 * (e % 16))) & 3) << 4,
                    _ => 0,
                };
                p.scales[g] * (q as f32 - offset as f32) + if bias { p.scales[g + 1] } else { 0.0 }
            }
            GpuFormat::Q8 { group, bias } => {
                let g = e / group as usize * (1 + bias as usize);
                let code = (p.codes[e / 4] >> (8 * (e % 4))) as u8 as i8 as f32;
                p.scales[g] * code + if bias { p.scales[g + 1] } else { 0.0 }
            }
            GpuFormat::Block(_) => panic!("use from_gpu for block layouts"),
        })
        .collect()
}

/// Values held by a GPU layout (tests use it to check that `to_gpu` loses nothing).
pub fn from_gpu(t: u32, words: &[u32], k: usize, rows: usize) -> Result<Vec<f32>, String> {
    let fmt = GpuFormat::compact(t).ok_or("unsupported type")?;
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    match fmt {
        GpuFormat::Q4 { .. } | GpuFormat::Q8 { .. } => unreachable!(),
        GpuFormat::F16 => Ok((0..k * rows)
            .map(|i| half(&bytes, i * 2))
            .collect()),
        GpuFormat::Block(_) => {
            let (vals, size) = block_info(t).unwrap();
            let (gvals, words, pair) = gpu_block(t).unwrap();
            let nb = k / vals;
            let per_row = k.div_ceil(gvals);
            let mut gguf = vec![0u8; rows * nb * size];
            for r in 0..rows {
                for bi in 0..nb {
                    let (gb, h) = (bi / pair, bi % pair);
                    let at = (r * per_row + gb) * words * 4;
                    let dst = &mut gguf[(r * nb + bi) * size..(r * nb + bi + 1) * size];
                    for (from, len, to) in field_moves(t, h) {
                        dst[from..from + len].copy_from_slice(&bytes[at + to..at + to + len]);
                    }
                }
            }
            let mut out = vec![0f32; k * rows];
            dequantize(t, &gguf, &mut out)?;
            Ok(out)
        }
    }
}

/// Code books the block kernels index, as one table of 32-bit words.
pub struct Tables;

impl Tables {
    pub const IQ4NL: usize = 0;
    pub const MXFP4: usize = 16;
    pub const IQ2XXS: usize = 32;
    pub const IQ2XS: usize = Self::IQ2XXS + 512;
    pub const IQ2S: usize = Self::IQ2XS + 1024;
    pub const IQ3XXS: usize = Self::IQ2S + 2048;
    pub const IQ3S: usize = Self::IQ3XXS + 256;
    pub const IQ1S: usize = Self::IQ3S + 512;
    pub const LEN: usize = Self::IQ1S + 4096;

    pub fn words() -> Vec<u32> {
        use crate::iq_tables::*;
        let mut t = Vec::with_capacity(Self::LEN);
        t.extend(IQ4.iter().map(|&v| (v as f32).to_bits()));
        t.extend(MXFP4_VALUES.iter().map(|&v| (v as f32).to_bits()));
        let split = |t: &mut Vec<u32>, g: &[u64]| {
            for &v in g {
                t.push(v as u32);
                t.push((v >> 32) as u32);
            }
        };
        split(&mut t, &IQ2XXS_GRID);
        split(&mut t, &IQ2XS_GRID);
        split(&mut t, &IQ2S_GRID);
        t.extend_from_slice(&IQ3XXS_GRID);
        t.extend_from_slice(&IQ3S_GRID);
        split(&mut t, &IQ1S_GRID);
        debug_assert_eq!(t.len(), Self::LEN);
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_round_trip() {
        for &v in &[
            0.0f32,
            -0.0,
            1.0,
            -2.5,
            65504.0,
            6.1035156e-5,
            5.9604645e-8,
            0.333_333_34,
        ] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!(
                (back - v).abs() <= v.abs() * 1e-3 + 1e-7,
                "{v} -> {h:#x} -> {back}"
            );
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert!(f16_to_f32(f32_to_f16(f32::INFINITY)).is_infinite());
    }

    #[test]
    fn q8_0_block_dequantizes() {
        let mut b = vec![0u8; 34];
        b[0..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        for j in 0..32 {
            b[2 + j] = (j as i8 - 16) as u8;
        }
        let mut out = vec![0f32; 32];
        dequantize(Q8_0, &b, &mut out).unwrap();
        assert_eq!(out[0], -8.0);
        assert_eq!(out[31], 7.5);
        let w = to_gpu(Q8_0, &b, 32, 1).unwrap();
        assert_eq!(w.len(), 17);
        assert_eq!(from_gpu(Q8_0, &w, 32, 1).unwrap(), out);
        let p = repack(Q8_0, &b, 32, 1).unwrap();
        assert_eq!(unpack(GpuFormat::expanded(Q8_0).unwrap(), &p, 32), out);
    }

    #[test]
    fn q4_0_offsets_are_preserved() {
        let mut b = vec![0u8; 18];
        b[0..2].copy_from_slice(&f32_to_f16(2.0).to_le_bytes());
        for j in 0..16 {
            b[2 + j] = (j as u8) | ((15 - j as u8) << 4);
        }
        let mut out = vec![0f32; 32];
        dequantize(Q4_0, &b, &mut out).unwrap();
        assert_eq!(out[0], -16.0);
        assert_eq!(out[16], 14.0);
        let w = to_gpu(Q4_0, &b, 32, 1).unwrap();
        assert_eq!(from_gpu(Q4_0, &w, 32, 1).unwrap(), out);
        let p = repack(Q4_0, &b, 32, 1).unwrap();
        assert_eq!(unpack(GpuFormat::expanded(Q4_0).unwrap(), &p, 32), out);
    }

    #[test]
    fn gpu_blocks_fill_whole_words() {
        for t in [Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS,
                  IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, TQ1_0, TQ2_0, MXFP4] {
            let (vals, size) = block_info(t).unwrap();
            let (gvals, words, pair) = gpu_block(t).unwrap();
            assert_eq!(gvals, vals * pair, "{}", type_name(t));
            assert!(words * 4 >= size * pair && words * 4 - size * pair < 4, "{}", type_name(t));
            let mut used = vec![false; words * 4];
            for h in 0..pair {
                let total: usize = field_moves(t, h).iter().map(|m| m.1).sum();
                assert_eq!(total, size, "{}", type_name(t));
                for (_, len, to) in field_moves(t, h) {
                    for u in &mut used[to..to + len] {
                        assert!(!*u, "{} fields overlap", type_name(t));
                        *u = true;
                    }
                }
            }
        }
    }

    #[test]
    fn file_layout_formats_stream_unchanged() {
        for t in [Q2_K, Q4_K, Q5_K, IQ4_XS, IQ1_M, Q4_1, Q5_1] {
            assert!(gpu_is_file_layout(t), "{}", type_name(t));
        }
        for t in [Q3_K, Q6_K, Q8_0, Q4_0, IQ2_XXS, MXFP4, F16] {
            assert!(!gpu_is_file_layout(t), "{}", type_name(t));
        }
        let b: Vec<u8> = (0..144u32).map(|i| (i * 7) as u8).collect();
        let w = to_gpu(Q4_K, &b, 256, 1).unwrap();
        assert_eq!(&*to_gpu_bytes(Q4_K, &b, 256, 1).unwrap(), &words_bytes(&w)[..]);
        // The byte path agrees with the word path for formats that move fields, rows of odd pairs too.
        for (t, k) in [(Q6_K, 512), (Q8_0, 96), (IQ2_XS, 256), (MXFP4, 96), (Q3_K, 256)] {
            let (vals, size) = block_info(t).unwrap();
            let n = k / vals * size * 3;
            let b: Vec<u8> = (0..n as u32).map(|i| (i * 13 + 5) as u8).collect();
            let w = to_gpu(t, &b, k, 3).unwrap();
            assert_eq!(&*to_gpu_bytes(t, &b, k, 3).unwrap(), &words_bytes(&w)[..], "{}", type_name(t));
        }
    }

    fn words_bytes(w: &[u32]) -> Vec<u8> {
        w.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn rejects_partial_blocks() {
        assert!(row_bytes(Q4_K, 896).is_err());
        assert_eq!(row_bytes(Q8_0, 896).unwrap(), 28 * 34);
        assert!(row_bytes(15, 256).unwrap_err().contains("Q8_K"));
    }
}
