import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { qrMatrix, qrSvg } from '../web/qr.mjs';

// Matrices python-qrcode (an independent encoder) made for the same text, version, level and mask.
const cases = JSON.parse(readFileSync(new URL('./fixtures/qr-reference.json', import.meta.url)));

test('QR matrices match python-qrcode for versions 1 to 40 and every error correction level', () => {
  for (const c of cases) {
    const m = qrMatrix(c.text, { ecl: c.ecl, mask: c.mask, version: c.version });
    const rows = m.map((r) => r.map((b) => (b ? '1' : '0')).join(''));
    assert.deepEqual(rows, c.rows, `version ${c.version}-${c.ecl}, mask ${c.mask}`);
  }
});

test('the smallest version that fits is chosen, and the SVG has a quiet zone', () => {
  const link = 'https://www.evolvetech.dev/play/evolve-llm/#remote=ABCD-EFGH-JKLM-NPQR';
  assert.equal(qrMatrix(link).length, 37); // version 5 at level M, as other encoders pick
  const svg = qrSvg(link, { title: 'Pairing <link>' });
  assert.match(svg, /^<svg[^>]+viewBox="0 0 45 45"/);
  assert.match(svg, /aria-label="Pairing &#60;link&#62;"/);
});
