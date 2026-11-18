#!/usr/bin/env python3
"""Reference logits for a Kimi K3.1 GGUF (llama.cpp's kimi-k31 architecture) in float64 NumPy.

Written from llama.cpp's graph (src/models/kimi-k31.cpp, the gated delta net op and
build_moe_ffn), independently of the engine, because transformers has no Kimi K3.1 model and
no other implementation runs without the original weights. It writes the fixture that
tests/tiny.rs checks the GPU against, for the tiny random model
akopytko/tiny-kimi-k3.1-pro-GGUF (tiny-k31-bf16.gguf).

Covered: Kimi delta attention (short convolution, L2-normalised q and k, the bounded decay gate,
beta, the gated delta rule, per-head norm and sigmoid gate), latent attention without rotation
and with its output gate, the DSA lightning indexer and its top-k selection, residuals that
attend across blocks, latent experts with sigmoid routing and a selection bias, shared experts
and the situ activation.

The indexer only matters past `indexer.top_k` (2048) cached positions. `--top-k N` writes a copy
of the model with that header value changed to N (same size, patched in place), so a short prompt
exercises the selection, and computes the reference for the copy.

usage: tools/ref_kimi_k31.py model.gguf fixture.json [--top-k N]
"""
import base64
import json
import struct
import sys

import numpy as np


def read_gguf(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'GGUF'
    n_t, n_kv = struct.unpack_from('<QQ', b, 8)
    off = 24

    def rstr():
        nonlocal off
        (n,) = struct.unpack_from('<Q', b, off)
        off += 8
        s = b[off:off + n].decode('utf-8', 'replace')
        off += n
        return s

    fmts = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}

    def rval(t):
        nonlocal off
        if t == 8:
            return rstr()
        if t == 9:
            et, n = struct.unpack_from('<IQ', b, off)
            off += 12
            return [rval(et) for _ in range(n)]
        f = fmts[t]
        (v,) = struct.unpack_from('<' + f, b, off)
        off += struct.calcsize(f)
        return v

    kv = {}
    for _ in range(n_kv):
        k = rstr()
        (t,) = struct.unpack_from('<I', b, off)
        off += 4
        kv[k] = rval(t)
    infos = []
    for _ in range(n_t):
        name = rstr()
        (nd,) = struct.unpack_from('<I', b, off)
        off += 4
        dims = list(struct.unpack_from('<' + 'Q' * nd, b, off))
        off += 8 * nd
        t, o = struct.unpack_from('<IQ', b, off)
        off += 12
        infos.append((name, dims, t, o))
    align = kv.get('general.alignment', 32)
    base = (off + align - 1) // align * align
    tensors = {}
    for name, dims, t, o in infos:
        tensors[name] = (dims, t, base + o)
    return b, kv, tensors


MXFP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float64)


def tensor(g, name):
    b, _, tensors = g
    dims, t, at = tensors[name]
    n = int(np.prod(dims))
    if t == 0:  # F32
        v = np.frombuffer(b, dtype='<f4', count=n, offset=at).astype(np.float64)
    elif t == 30:  # BF16
        h = np.frombuffer(b, dtype='<u2', count=n, offset=at).astype(np.uint32) << 16
        v = h.view(np.float32).astype(np.float64)
    elif t == 39:  # MXFP4: E8M0 scale, then 16 bytes of 4-bit codes, per 32 values
        blocks = np.frombuffer(b, dtype=np.uint8, count=n // 32 * 17, offset=at).reshape(-1, 17)
        d = np.ldexp(1.0, blocks[:, 0].astype(np.int64) - 128)
        lo = MXFP4[blocks[:, 1:] & 15]
        hi = MXFP4[blocks[:, 1:] >> 4]
        v = (np.concatenate([lo, hi], axis=1) * d[:, None]).reshape(-1)
    else:
        raise SystemExit(f'{name}: ggml type {t} is not handled by this reference')
    return v.reshape(list(reversed(dims)))


def patch_top_k(path, top_k):
    """A copy of the model with attention.indexer.top_k set to `top_k`."""
    b = bytearray(open(path, 'rb').read())
    key = b'kimi-k31.attention.indexer.top_k'
    at = b.index(struct.pack('<Q', len(key)) + key) + 8 + len(key)
    (t,) = struct.unpack_from('<I', b, at)
    assert t == 4, f'top_k is ggml type {t}, expected u32'
    struct.pack_into('<I', b, at + 4, top_k)
    out = path.replace('.gguf', f'-topk{top_k}.gguf')
    open(out, 'wb').write(b)
    return out


def main():
    path, out_path = sys.argv[1], sys.argv[2]
    if '--top-k' in sys.argv:
        path = patch_top_k(path, int(sys.argv[sys.argv.index('--top-k') + 1]))
    g = read_gguf(path)
    kv = g[1]
    a = kv['general.architecture']
    assert a == 'kimi-k31', a
    k = lambda s: kv[f'{a}.{s}']
    W = lambda n: tensor(g, n)
    n_layer = k('block_count')
    nh = k('attention.head_count')
    hd = k('kda.head_dim')
    lb = k('kda.gate_lower_bound')
    eps = k('attention.layer_norm_rms_epsilon')
    bs = k('attn_res.block_size')
    kv_lora = k('attention.kv_lora_rank')
    k_mla = k('attention.key_length_mla')
    rope = k('rope.dimension_count')
    nope = k_mla - rope
    n_used = k('expert_used_count')
    lead = k('leading_dense_block_count')
    beta_s, beta_l = k('activation.situ_beta'), k('activation.situ_linear_beta')
    w_norm = k('expert_weights_norm')
    w_scale = k('expert_weights_scale')
    recurrent = [v == 0 for v in k('attention.head_count_kv')]
    top_k = kv.get(f'{a}.attention.indexer.top_k', 0)
    ix_full = kv.get(f'{a}.attention.indexer.types', [True] * n_layer)
    ix_eps = kv.get(f'{a}.attention.layer_norm_epsilon', 1e-5)
    selection = {}  # the last indexer's choice, for layers that share it
    d = nh * hd

    def rms(x, w=None):
        y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        return y if w is None else y * w

    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    silu = lambda x: x * sig(x)

    def situ(gt, up):
        a = beta_s * np.tanh(gt / beta_s) * sig(gt)
        if beta_l > 0:
            up = beta_l * np.tanh(up / beta_l)
        return a * up

    def res_mix(cur, stack, w):
        if not stack:
            return cur
        srcs = stack + [cur]
        scores = np.stack([np.sum(rms(s) * w, axis=-1) for s in srcs], axis=-1)  # (T, n)
        p = np.exp(scores - scores.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        return sum(p[:, i:i + 1] * s for i, s in enumerate(srcs))

    def kda(xn, l):
        T = xn.shape[0]
        n = lambda s: f'blk.{l}.{s}'
        proj = {}
        for c in 'qkv':
            x = xn @ W(n(f'attn_{c}.weight')).T  # (T, d)
            w = W(n(f'ssm_conv1d_{c}.weight')).reshape(d, 4)
            pad = np.concatenate([np.zeros((3, d)), x], axis=0)
            conv = sum(pad[j:j + T] * w[:, j] for j in range(4))
            proj[c] = silu(conv).reshape(T, nh, hd)
        l2 = lambda x: x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)
        q, kk, v = l2(proj['q']), l2(proj['k']), proj['v']
        gr = (xn @ W(n('ssm_f_a.weight')).T) @ W(n('ssm_f_b.weight')).T + W(n('ssm_dt.bias'))
        A = W(n('ssm_a')).reshape(nh)  # -exp(A_log)
        gate = lb * sig(-(gr.reshape(T, nh, hd) * A[None, :, None]))
        beta = sig(xn @ W(n('ssm_beta.weight')).T)  # (T, nh)
        S = np.zeros((nh, hd, hd))  # [head, key channel i, value channel j]
        o = np.zeros((T, nh, hd))
        for t in range(T):
            for h in range(nh):
                S[h] *= np.exp(gate[t, h])[:, None]
                delta = (v[t, h] - S[h].T @ kk[t, h]) * beta[t, h]
                S[h] += np.outer(kk[t, h], delta)
                o[t, h] = S[h].T @ q[t, h] / np.sqrt(hd)
        o = rms(o, W(n('ssm_norm.weight'))) * sig((xn @ W(n('ssm_g.weight')).T).reshape(T, nh, hd))
        return o.reshape(T, d) @ W(n('attn_output.weight')).T

    def mla(xn, l):
        T = xn.shape[0]
        n = lambda s: f'blk.{l}.{s}'
        qa = rms(xn @ W(n('attn_q_a.weight')).T, W(n('attn_q_a_norm.weight')))
        Q = (qa @ W(n('attn_q_b.weight')).T).reshape(T, nh, k_mla)
        q_nope, q_pe = Q[..., :nope], Q[..., nope:]
        kva = xn @ W(n('attn_kv_a_mqa.weight')).T
        lat = rms(kva[:, :kv_lora], W(n('attn_kv_a_norm.weight')))
        kpe = kva[:, kv_lora:]  # not rotated: Kimi K3 latent attention has no position encoding
        wk_b = W(n('attn_k_b.weight'))  # (nh, kv_lora, nope)
        wv_b = W(n('attn_v_b.weight'))  # (nh, v, kv_lora)
        q_abs = np.einsum('hcn,thn->thc', wk_b, q_nope)
        scores = (np.einsum('thc,sc->ths', q_abs, lat) + np.einsum('thr,sr->ths', q_pe, kpe)) / np.sqrt(k_mla)
        mask = np.triu(np.ones((T, T), dtype=bool), 1)
        if top_k and ix_full[l]:
            # Lightning indexer: relu'd per-head scores against LayerNorm'd keys, weighted per
            # head; each query keeps its top_k causal positions.
            iq = (qa @ W(n('indexer.attn_q_b.weight')).T).reshape(T, -1, W(n('indexer.k_norm.weight')).shape[0])
            ik = xn @ W(n('indexer.attn_k.weight')).T
            ik = (ik - ik.mean(-1, keepdims=True)) / np.sqrt(ik.var(-1, keepdims=True) + ix_eps)
            ik = ik * W(n('indexer.k_norm.weight')) + W(n('indexer.k_norm.bias'))
            # Cached in f16, as the engine does: the selection can sit within f16 precision of
            # the cut (it does for this test's random weights), so round the same way.
            ik = ik.astype(np.float16).astype(np.float64)
            iw = (xn @ W(n('indexer.proj.weight')).T) / np.sqrt(iq.shape[1] * iq.shape[2])
            sc = np.einsum('th,ths->ts', iw, np.maximum(np.einsum('thd,sd->ths', iq, ik), 0))
            sc[mask] = -np.inf
            keep = np.zeros((T, T), dtype=bool)
            for t in range(T):
                order = np.argsort(-sc[t], kind='stable')[:min(top_k, t + 1)]
                keep[t, order] = True
            selection['keep'] = keep
        if top_k and 'keep' in selection:
            mask = mask | ~selection['keep']
        scores[:, :, :][np.broadcast_to(mask[:, None, :], scores.shape)] = -np.inf
        p = np.exp(scores - scores.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        o_lat = np.einsum('ths,sc->thc', p, lat)
        o = np.einsum('hvc,thc->thv', wv_b, o_lat).reshape(T, -1)
        o = o * sig(xn @ W(n('attn_gate.weight')).T)
        return o @ W(n('attn_output.weight')).T

    def ffn(xn, l):
        n = lambda s: f'blk.{l}.{s}'
        if l < lead:
            return situ(xn @ W(n('ffn_gate.weight')).T, xn @ W(n('ffn_up.weight')).T) @ W(n('ffn_down.weight')).T
        xl = xn @ W(n('ffn_routed_down.weight')).T
        probs = sig(xn @ W(n('ffn_gate_inp.weight')).T)  # (T, n_expert)
        sel = np.argsort(-(probs + W(n('exp_probs_b.bias'))), axis=-1, kind='stable')[:, :n_used]
        wts = np.take_along_axis(probs, sel, axis=-1)
        if w_norm:
            wts = wts / np.maximum(wts.sum(axis=-1, keepdims=True), 6.103515625e-5)
        wts = wts * w_scale
        G, U, D = W(n('ffn_gate_exps.weight')), W(n('ffn_up_exps.weight')), W(n('ffn_down_exps.weight'))
        moe = np.zeros_like(xl)
        for t in range(xl.shape[0]):
            for e, w in zip(sel[t], wts[t]):
                moe[t] += w * (D[e] @ situ(G[e] @ xl[t], U[e] @ xl[t]))
        moe = rms(moe, W(n('ffn_routed_norm.weight'))) @ W(n('ffn_routed_up.weight')).T
        sh = situ(xn @ W(n('ffn_gate_shexp.weight')).T, xn @ W(n('ffn_up_shexp.weight')).T) @ W(n('ffn_down_shexp.weight')).T
        return moe + sh

    tokens = [163584, 3, 1029, 77, 2024, 99, 12345, 163585, 5, 6, 4096, 31337, 8, 9, 100000, 42, 7, 2048, 1, 163586]
    x = W('token_embd.weight')[tokens]
    stack = []
    for l in range(n_layer):
        n = lambda s: f'blk.{l}.{s}'
        P = x
        A = res_mix(P, stack, W(n('attn_res_score.weight')))
        banked = l % bs == 0
        if banked:
            stack.append(P)  # the raw layer input, not the mix
        xn = rms(A, W(n('attn_norm.weight')))
        a_out = kda(xn, l) if recurrent[l] else mla(xn, l)
        P = a_out if banked else P + a_out
        F = res_mix(P, stack, W(n('ffn_res_score.weight')))
        P = P + ffn(rms(F, W(n('ffn_norm.weight'))), l)
        x = P
    F = res_mix(x, stack, W('output_res_score.weight'))
    logits = rms(F, W('output_norm.weight')) @ W('output.weight').T
    positions = [3, 9, len(tokens) - 1]
    enc = lambda v: base64.b64encode(np.asarray(v, dtype='<f4').tobytes()).decode()
    last = logits[-1]
    fixture = {
        'file': path.split('/')[-1],
        'tokens': tokens,
        'positions': positions,
        'logits': [enc(logits[p]) for p in positions],
        'range': float(last.max() - last.min()),
    }
    json.dump(fixture, open(out_path, "w"))
    print(f'{len(tokens)} tokens; last-position top token {int(last.argmax())}, logit range {fixture["range"]:.3f}')


if __name__ == '__main__':
    main()
