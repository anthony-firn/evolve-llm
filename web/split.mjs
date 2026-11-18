// Sharing one model among a person's devices: which layers each runs.

/**
 * `lb`: the model's sizes (`layer_bytes` from the engine). `caps`: bytes of weight memory per
 * device, this one first (it also holds the embedding and the head). Each device gets a run of
 * layers in proportion to its memory, in order. Returns [{first, end}] per device; a device
 * whose share rounds to nothing gets first === end.
 */
export function planSplit(lb, caps) {
  const n = lb.n_layer;
  const fixed = lb.embed + (lb.tied ? 0 : lb.head);
  const total = lb.layers.reduce((a, b) => a + b, 0);
  const room = caps.map((c, i) => Math.max(0, i === 0 ? c - fixed : c));
  const sum = room.reduce((a, b) => a + b, 0) || 1;
  const stages = [];
  let l = 0;
  let acc = 0;
  let target = 0;
  for (let d = 0; d < caps.length; d++) {
    target += (total * room[d]) / sum;
    const first = l;
    if (d === caps.length - 1) l = n;
    else while (l < n && acc + lb.layers[l] / 2 <= target) acc += lb.layers[l++];
    stages.push({ first, end: l });
  }
  return stages;
}

/** Bytes of weights a device can hold, from its memory setting ('auto', 'all' or GB). */
export function capacityOf(pref) {
  if (pref === 'all') return 16 * 2 ** 30;
  if (pref && pref !== 'auto' && Number(pref) > 0) return Number(pref) * 2 ** 30;
  return 4 * 2 ** 30;
}
