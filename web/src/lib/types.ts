import type { Run } from "./config";

export interface StepRow { run: string; step: number; ts: string; lr: number; lr_mult: number; train_loss: number; job_id: number }
export interface EvalRow { run: string; step: number; ts: string; val_loss: number }
export interface PriceRow { ticker: string; ts: string; open: number; high: number; low: number; close: number; volume: number; job_id?: number }
export interface JobRow {
  id: number; started_at: string; finished_at?: string | null; window_start: string; window_end: string;
  status: "running" | "ok" | "skipped" | "failed"; ran_eval: boolean; n_bars: Record<string, number> | null;
  duration_s: number | null; mappers: Record<string, Record<string, unknown>> | null; error?: string | null;
}
export interface Raw { steps: StepRow[]; evals: EvalRow[]; jobs: JobRow[]; prices: PriceRow[] }

export type Series = (number | null)[];
export interface Model {
  maxStep: number;
  lr: Record<Run, Series>;          // ×1e3
  train: Record<Run, Series>;
  val: Record<Run, Series>;
  close: Record<Run, Series>;       // closing price of each run's ticker at that step
  stepTs: (number | null)[];        // ms, per step index
}
