// Mirrors stocklr/config.py + stocklr/lr.py. Keep in sync when runs or the schedule change.
export const RUNS = ["ANT", "SPCX", "SP500", "NVDA"] as const;
export type Run = (typeof RUNS)[number];

export const PRICE_SOURCE: Record<Run, string | null> = {
  ANT: "ANT", SPCX: "SPCX", SP500: "SPY", NVDA: "NVDA",
};
export const COLOR_VAR: Record<Run, string> = {
  ANT: "--s-ant", SPCX: "--s-spcx", SP500: "--s-sp500", NVDA: "--s-nvda",
};
const qs = new URLSearchParams(location.search);
// ?backup=<tag> loads web/public/backups/<tag>.json (made by research/backup_to_dashboard.py), no network
export const BACKUP: string = /^[\w.-]+$/.test(qs.get("backup") ?? "") ? qs.get("backup")! : "";
export const REPLAY: string = qs.get("replay") ?? (BACKUP || qs.has("test") || qs.has("demo") ? "" : import.meta.env.VITE_REPLAY ?? "");
export const LOSS_RUNS = RUNS.filter(r => PRICE_SOURCE[r]);
export const PRICED_RUNS = LOSS_RUNS.filter(r => !REPLAY || r !== "ANT");

export const TOTAL_STEPS = 9375;
export const TOKENS_PER_STEP = 64 * 1024;   // BATCH_SIZE x SEQ_LEN in stocklr/train.py
export const ETA0 = 3e-3;
export const WARMUP = 94;
export const DECAY_START = 7500;
export const REFRESH_MS = 5 * 60 * 1000;

export const wsd = (step: number) =>
  step <= WARMUP ? step / WARMUP : step <= DECAY_START ? 1 : Math.max(0, (TOTAL_STEPS - step) / (TOTAL_STEPS - DECAY_START));

export const SUPABASE_URL = (import.meta.env.VITE_SUPABASE_URL as string | undefined) ?? "";
export const SUPABASE_ANON_KEY = (import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined) ?? "";
export const DEMO = qs.has("demo");
export const PREFIX = !REPLAY && qs.has("test") ? "test-" : "";   // ?test shows runs kept by `modal run modal_app.py --keep`
