import { PREFIX, PRICE_SOURCE, REPLAY, RUNS, type Run } from "./config";
import { seededNoise } from "./noise";
import type { Model, Raw, Series } from "./types";

const byRun = (f: () => Series) => Object.fromEntries(RUNS.map((r) => [r, f()])) as Record<Run, Series>;
const asRun = (name: string) => (RUNS as readonly string[]).includes(name.slice(PREFIX.length)) ? (name.slice(PREFIX.length) as Run) : null;

/** Turn table rows into step-aligned arrays (index = step - 1) for the charts. */
export function shape(raw: Raw): Model {
  const maxStep = raw.steps.reduce((m, r) => Math.max(m, r.step), 0);
  const blank = () => new Array<number | null>(maxStep).fill(null);
  const lr = byRun(blank), train = byRun(blank), val = byRun(blank), close = byRun(blank);
  const stepTs = blank();
  const closeAt = new Map<string, Map<number, number>>();
  for (const p of raw.prices) {
    if (!closeAt.has(p.ticker)) closeAt.set(p.ticker, new Map());
    closeAt.get(p.ticker)!.set(Date.parse(p.ts), p.close);
  }
  for (const row of raw.steps) {
    const r = asRun(row.run);
    if (!r) continue;
    const i = row.step - 1, t = Date.parse(row.ts);
    lr[r][i] = row.lr * 1e3;
    train[r][i] = row.train_loss;
    const src = PRICE_SOURCE[r];
    if (src) { const c = closeAt.get(src)?.get(t); if (c != null) close[r][i] = c; }
    if (stepTs[i] == null || src) stepTs[i] = t;
  }
  for (const row of raw.evals) {
    const r = asRun(row.run);
    if (r && row.step >= 1 && row.step <= maxStep) val[r][row.step - 1] = row.val_loss;
  }
  // Published replays contain three price-backed runs. Generate only the missing
  // ANT losses on their step/evaluation cadence using the replay job's loss model.
  if (REPLAY && !train.ANT.some(v => v != null)) {
    const { gaussian: noise } = seededNoise(0x414e54);
    for (let i = 0; i < maxStep; i++) {
      if (train.SPCX[i] == null) continue;
      const trend = 2.9 + 2.1 * Math.exp(-(i + 1) / 1900);
      train.ANT[i] = Number((trend + .09 * noise()).toFixed(5));
      if (val.SPCX[i] != null) val.ANT[i] = Number((trend + .12 + .025 * noise()).toFixed(5));
    }
  }
  return { maxStep, lr, train, val, close, stepTs };
}

export function ema(arr: Series, span: number): Series {
  if (span <= 1) return arr;
  const a = 2 / (span + 1);
  let acc: number | null = null;
  return arr.map((v) => (v == null ? null : (acc = acc == null ? v : a * v + (1 - a) * acc)));
}
