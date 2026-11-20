-- stock-lr central store. Writer: Modal hourly job (service role / direct Postgres). Reader: Vercel dashboard (anon, read-only).

create table if not exists jobs (
  id            bigserial primary key,
  started_at    timestamptz not null default now(),
  finished_at   timestamptz,
  window_start  timestamptz not null,          -- bars trained on: [window_start, window_end)
  window_end    timestamptz not null,
  status        text not null default 'running' check (status in ('running', 'ok', 'skipped', 'failed')),
  ran_eval      boolean not null default false,
  n_bars        jsonb,                         -- {"NVDA": 60, ...}
  duration_s    double precision,
  error         text
);

-- every 1-minute SIP bar the job pulled (regular session only); same data as the raw CSV
create table if not exists prices (
  ticker  text not null,
  ts      timestamptz not null,                -- bar start time
  open    double precision, high double precision, low double precision,
  close   double precision not null,
  volume  double precision,
  job_id  bigint references jobs(id),
  primary key (ticker, ts)
);

-- one row per optimizer step per model
create table if not exists steps (
  run         text not null,                   -- 'ANT' | 'SPCX' | 'GOOGL' | 'NVDA' | 'BASELINE'
  step        integer not null,                -- global step, 1-based
  ts          timestamptz not null,            -- the price bar this step's LR came from
  lr          double precision not null,       -- LR actually applied (WSD x price multiplier)
  lr_mult     double precision not null,       -- price multiplier alone (1.0 for BASELINE)
  train_loss  double precision not null,
  job_id      bigint references jobs(id),
  primary key (run, step)
);

create table if not exists evals (
  run       text not null,
  step      integer not null,
  ts        timestamptz not null default now(),
  val_loss  double precision not null,
  job_id    bigint references jobs(id),
  primary key (run, step)
);

create index if not exists steps_ts_idx  on steps (ts);
create index if not exists prices_ts_idx on prices (ts);

-- dashboard reads with the public anon key: allow SELECT only, nothing else
alter table jobs   enable row level security;
alter table prices enable row level security;
alter table steps  enable row level security;
alter table evals  enable row level security;
do $$ begin
  create policy "public read" on jobs   for select to anon using (true);
  create policy "public read" on prices for select to anon using (true);
  create policy "public read" on steps  for select to anon using (true);
  create policy "public read" on evals  for select to anon using (true);
exception when duplicate_object then null; end $$;
alter table jobs add column if not exists is_test boolean not null default false;
alter table jobs add column if not exists mappers jsonb;   -- mapper spec per run used by this job
