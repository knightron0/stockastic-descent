import { createClient } from "@supabase/supabase-js";
import { PREFIX, SUPABASE_ANON_KEY, SUPABASE_URL } from "./config";
import type { EvalRow, JobRow, PriceRow, Raw, StepRow } from "./types";

const PAGE = 1000;     // Supabase's default max rows per request
const PARALLEL = 6;

/** Fetch every row of a query, PAGE rows at a time, PARALLEL pages per round trip. */
async function all<T>(query: (from: number, to: number) => PromiseLike<{ data: T[] | null; error: { message: string } | null }>) {
  const out: T[] = [];
  for (let wave = 0; ; wave++) {
    const pages = await Promise.all(Array.from({ length: PARALLEL }, (_, i) => {
      const from = (wave * PARALLEL + i) * PAGE;
      return query(from, from + PAGE - 1);
    }));
    for (const { data, error } of pages) {
      if (error) throw new Error(error.message);
      out.push(...(data ?? []));
    }
    if (pages.some((p) => (p.data?.length ?? 0) < PAGE)) return out;
  }
}

export async function loadSupabase(): Promise<Raw> {
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) throw new Error("Supabase connection settings missing");
  const sb = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, { auth: { persistSession: false } });
  // real runs vs test-* runs (kept by `modal run modal_app.py --keep`)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const runs = (q: any) => (PREFIX ? q.like("run", "test-%") : q.not("run", "like", "test-%"));
  const [steps, evals, jobs, prices] = await Promise.all([
    all<StepRow>((a, b) => runs(sb.from("steps").select("run,step,ts,lr,lr_mult,train_loss,job_id")).order("step").order("run").range(a, b)),
    all<EvalRow>((a, b) => runs(sb.from("evals").select("run,step,ts,val_loss")).order("step").range(a, b)),
    all<JobRow>((a, b) => sb.from("jobs").select("id,started_at,finished_at,window_start,window_end,status,ran_eval,n_bars,duration_s,mappers,error")
      .eq("is_test", !!PREFIX).order("window_end", { ascending: false }).range(a, b)),
    all<PriceRow>((a, b) => sb.from("prices").select("ticker,ts,open,high,low,close,volume,job_id").order("ts").order("ticker").range(a, b)),
  ]);
  return { steps, evals, jobs, prices };
}
