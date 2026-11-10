// Read-only public web access. No ambient cookies, credentials, redirects or private URLs.
export function publicUrl(value) {
  const u = new URL(String(value || '').trim());
  if (!['https:', 'http:'].includes(u.protocol) || u.username || u.password) throw new Error('Use a public HTTP(S) URL without credentials.');
  const host = u.hostname.toLowerCase().replace(/^\[|\]$/g, '').replace(/\.$/, '');
  if (host === 'localhost' || !host.includes('.') || /(?:^|\.)(?:localhost|local|internal|test|invalid)$/.test(host) || host.includes(':') ||
      /^(?:0|10|127|169\.254|192\.168|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))\./.test(host))
    throw new Error('Private and local network addresses are unavailable.');
  if (/^\d+\.\d+\.\d+\.\d+$/.test(host)) {
    const [a,b,c]=host.split('.').map(Number);
    if (a>=224 || (a===192 && b===0 && [0,2].includes(c)) || (a===198 && [18,19].includes(b)) || (a===198 && b===51 && c===100) || (a===203 && b===0 && c===113)) throw new Error('Reserved network addresses are unavailable.');
  }
  u.hash = ''; return u.href;
}
export function requestOptions(opts = {}) {
  const timeout = AbortSignal.timeout(Math.min(30000, Math.max(1, opts.timeoutMs || 15000)));
  const signal = opts.signal ? AbortSignal.any([opts.signal, timeout]) : timeout;
  return { signal, credentials: 'omit', redirect: 'error', referrerPolicy: 'no-referrer', cache: 'no-store' };
}
export async function boundedText(res, maxBytes = 65536, truncate = false) {
  if (!res.body?.getReader) {
    const text = await res.text();
    if (new TextEncoder().encode(text).byteLength > maxBytes && !truncate) throw new Error('Response exceeds the size limit.');
    return text.slice(0, maxBytes);
  }
  const reader = res.body.getReader(), decoder = new TextDecoder(); let count = 0, out = '';
  try {
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      const remaining = maxBytes - count;
      if (value.length > remaining) {
        if (!truncate) throw new Error('Response exceeds the size limit.');
        out += decoder.decode(value.subarray(0, remaining), { stream: true }); break;
      }
      count += value.length; out += decoder.decode(value, { stream: true });
      if (count === maxBytes && truncate) break;
    }
    return out + decoder.decode();
  } finally { await reader.cancel().catch(() => {}); }
}
