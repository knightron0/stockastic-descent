// Seeded noise keeps simulations stable across polls and page reloads.
export function seededNoise(seed: number) {
  function uniform() {
    seed = (seed + 0x6d2b79f5) | 0;
    let bits = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    bits = (bits + Math.imul(bits ^ (bits >>> 7), 61 | bits)) ^ bits;
    return ((bits ^ (bits >>> 14)) >>> 0) / 4294967296;
  }
  const gaussian = () => Math.sqrt(-2 * Math.log(uniform() + 1e-12)) * Math.cos(2 * Math.PI * uniform());
  return { uniform, gaussian };
}
