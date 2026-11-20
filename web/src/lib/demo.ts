// Synthetic rows with exactly the shapes Supabase returns, so the UI can be built and previewed offline (?demo).
import { seededNoise } from "./noise";
import { ETA0, PREFIX, PRICE_SOURCE, RUNS, wsd } from "./config";
import type { EvalRow, JobRow, PriceRow, Raw, StepRow } from "./types";

export function demoData(): Raw {
  const { gaussian: gauss, uniform: rand } = seededNoise(7);
  const start: Record<string, number> = { ANT: 96, SPCX: 138, SPY: 716, NVDA: 146 };
  const vol: Record<string, number> = { ANT: 0.0016, SPCX: 0.0011, SPY: 0.0004, NVDA: 0.0009 };
  const tickers = Object.keys(start);

  const windows: [number, number][] = [];
  for (const d of [Date.UTC(2026, 10, 16), Date.UTC(2026, 10, 17), Date.UTC(2026, 10, 18)]) {
    const open = d + 14.5 * 3600e3; // 9:30 ET (UTC-5 in November)
    windows.push([open, open + 30 * 60e3]);
    for (let h = 0; h < 6; h++) windows.push([open + (30 + 60 * h) * 60e3, open + (90 + 60 * h) * 60e3]);
  }
  const bars: Record<string, PriceRow[]> = Object.fromEntries(tickers.map((t) => [t, []]));
  const px = { ...start };
  for (const [a, b] of windows) for (let m = a; m < b; m += 60e3) for (const t of tickers) {
    px[t] *= Math.exp(vol[t] * gauss() + (t === "NVDA" ? -0.00006 : 0));
    bars[t].push({ ticker: t, ts: new Date(m).toISOString(), open: px[t], high: px[t], low: px[t], close: px[t], volume: 1000 });
  }
  // same formula as stocklr/mappers.py Detrended(k=15, span=390, smooth=10)
  const ewm = (x: number[], span: number) => { const a = 2 / (span + 1); let acc = x[0]; return x.map((v) => (acc = a * v + (1 - a) * acc)); };
  const mult: Record<string, number[]> = {};
  for (const t of tickers) {
    const lp = ewm(bars[t].map((b) => b.close), 10).map(Math.log), base = ewm(lp, 390), cap = Math.log(2);
    mult[t] = lp.map((v, i) => Math.exp(cap * Math.tanh((15 * (v - base[i])) / cap)));
  }
  const offset: Record<string, number> = { ANT: 0.018, SPCX: 0.006, SP500: -0.004, NVDA: 0.01 };
  const steps: StepRow[] = [], evals: EvalRow[] = [], jobs: JobRow[] = [];
  let barOff = 0, step = 0;
  windows.forEach(([a, b], j) => {
    const n = (b - a) / 60e3, ranEval = (new Date(b).getUTCHours() - 5) % 2 === 0;
    for (let i = 0; i < 240; i++) {
      step++;
      const bi = barOff + Math.floor((i * n) / 240);
      const loss = 2.95 + 2.0 * Math.exp(-step / 900) + 0.4 * Math.exp(-step / 70);
      for (const r of RUNS) {
        const src = PRICE_SOURCE[r];
        const m = src ? mult[src][bi] : 1;
        const ts = src ? bars[src][bi].ts : new Date(a + ((b - a) * i) / 240).toISOString();
        const drift = 0.004 * Math.sin(step / 700 + offset[r] * 300);
        steps.push({ run: PREFIX + r, step, ts, lr: ETA0 * wsd(step) * m, lr_mult: m, train_loss: loss + offset[r] + drift + 0.05 * gauss(), job_id: j + 1 });
        if (ranEval && i % 60 === 59) evals.push({ run: PREFIX + r, step, ts: new Date(b + 17 * 60e3).toISOString(), val_loss: loss + offset[r] + drift + 0.012 });
      }
    }
    jobs.push({ id: j + 1, window_start: new Date(a).toISOString(), window_end: new Date(b).toISOString(), status: "ok",
      ran_eval: ranEval, n_bars: { ANT: n, SPCX: n, SPY: n, NVDA: n }, duration_s: 190 + 10 * rand(),
      started_at: new Date(b + 17 * 60e3).toISOString(), finished_at: new Date(b + 20 * 60e3).toISOString(),
      mappers: { ANT: { name: "detrended", k: 15, span: 390, smooth: 10 } } });
    barOff += n;
  });
  const prices = Object.values(bars).flat().sort((x, y) => (x.ts < y.ts ? -1 : 1));
  return { steps, evals, jobs: jobs.reverse(), prices };
}
