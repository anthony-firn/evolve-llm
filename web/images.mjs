/**
 * Images for vision models: what an image encoder reads is RGB with both sides multiples of its
 * unit (32 for Qwen3-VL: 16-pixel patches merged 2x2), and each unit costs a prompt token.
 */

/** An RGBA image (width*height*4 bytes) resized by area averaging to fit within maxSide, both
 *  sides multiples of `unit`, as RGB bytes. */
export function fitImage(rgba, width, height, { unit = 32, maxSide = 1024 } = {}) {
  if (!(width > 0 && height > 0) || rgba.length < width * height * 4) throw new Error('The image data does not match its size.');
  const scale = Math.min(1, maxSide / Math.max(width, height));
  const W = Math.max(unit, Math.round((width * scale) / unit) * unit);
  const H = Math.max(unit, Math.round((height * scale) / unit) * unit);
  const rgb = new Uint8Array(W * H * 3);
  const sx = width / W, sy = height / H;
  for (let y = 0; y < H; y++) {
    const y0 = Math.min(height - 1, Math.floor(y * sy)), y1 = Math.max(y0 + 1, Math.min(height, Math.floor((y + 1) * sy)));
    for (let x = 0; x < W; x++) {
      const x0 = Math.min(width - 1, Math.floor(x * sx)), x1 = Math.max(x0 + 1, Math.min(width, Math.floor((x + 1) * sx)));
      let r = 0, g = 0, b = 0;
      for (let yy = y0; yy < y1; yy++) {
        for (let xx = x0; xx < x1; xx++) {
          const i = (yy * width + xx) * 4;
          r += rgba[i]; g += rgba[i + 1]; b += rgba[i + 2];
        }
      }
      const n = (y1 - y0) * (x1 - x0), o = (y * W + x) * 3;
      rgb[o] = Math.round(r / n); rgb[o + 1] = Math.round(g / n); rgb[o + 2] = Math.round(b / n);
    }
  }
  return { rgb, width: W, height: H };
}
