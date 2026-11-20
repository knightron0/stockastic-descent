<script lang="ts">
  import { onMount } from "svelte";
  import Chart from "./lib/Chart.svelte";
  import type { ChartSpec } from "./lib/chartOptions";
  import { BACKUP, COLOR_VAR, DEMO, LOSS_RUNS, PRICE_SOURCE, PRICED_RUNS, REPLAY, REFRESH_MS, TOKENS_PER_STEP, TOTAL_STEPS, type Run } from "./lib/config";
  import { loadReplay } from "./lib/replay";
  import { loadBackup } from "./lib/backup";
  import { loadSupabase } from "./lib/data";
  import { demoData } from "./lib/demo";
  import { fmtET, fmtTokens } from "./lib/format";
  import { ema, shape } from "./lib/shape";
  import type { Raw } from "./lib/types";

  let raw = $state<Raw | null>(null);
  let error = $state<string | null>(null);
  let ticker = $state<Run>(PRICED_RUNS[0]);
  let hovered = $state<Run | null>(null);   // run under the pointer, highlighted in every panel
  const params = new URLSearchParams(location.search);
  const initialSmooth = Number(params.get("smooth") ?? 100);
  let view = $state<"zoom" | "full">(params.get("view") === "full" ? "full" : "zoom");
  let smooth = $state([1, 20, 100, 400].includes(initialSmooth) ? initialSmooth : 100);

  const model = $derived(raw ? shape(raw) : null);
  const dataThrough = $derived(raw?.steps.reduce((latest, row) => {
    const ts = Date.parse(row.ts);
    return ts > latest ? ts : latest;
  }, 0));
  const hasVal = $derived(!!model && LOSS_RUNS.some(run => model.val[run].some(v => v != null)));

  const standings = $derived.by(() => {
    if (!model) return null;
    // latest step where every run has a validation loss, compared with the standings just before the most
    // recent change in order: the arrows stay up until the order changes again
    type Rows = { run: Run; loss: number }[];
    const ranked = (i: number): Rows => LOSS_RUNS.map(run => ({ run, loss: model.val[run][i]! })).sort((a, b) => a.loss - b.loss);
    const full = (i: number) => LOSS_RUNS.every(run => Number.isFinite(model.val[run][i]));
    const rankOf = (rows: Rows, run: Run) => {
      const loss = rows.find(r => r.run === run)!.loss;
      return rows.findIndex(r => r.loss === loss) + 1;     // ties share a rank
    };
    const order = (rows: Rows) => LOSS_RUNS.map(run => rankOf(rows, run)).join(",");
    let curI = model.maxStep - 1;
    while (curI >= 0 && !full(curI)) curI--;
    if (curI < 0) return null;
    const cur = { step: curI + 1, rows: ranked(curI) };
    const key = order(cur.rows);
    let prev: Rows | null = null, since = cur.step;      // since = first eval with today's order
    for (let i = curI - 1; i >= 0; i--) {
      if (!full(i)) continue;
      const rows = ranked(i);
      if (order(rows) !== key) { prev = rows; break; }
      since = i + 1;
    }
    // positions gained at the last change (positive = moved up); no arrows if the order never changed
    const moved = Object.fromEntries(cur.rows.map(r => [r.run, prev ? rankOf(prev, r.run) - rankOf(cur.rows, r.run) : 0])) as Record<Run, number>;
    return { ...cur, moved, since: prev ? since : null };
  });

  const specs = $derived.by(() => {
    if (!model) return null;
    const lossFmt = (v: number) => v.toFixed(view === "zoom" ? 3 : 2).replace(/\.?0+$/, "");
    // zoomed: the most recent ~70% of observed steps (after the steep early drop), y fitted to that window
    const xRange: [number, number] = view === "zoom" && model.maxStep > 50
      ? [Math.max(1, Math.floor(model.maxStep * 0.3)), model.maxStep]
      : [1, TOTAL_STEPS];
    const train: ChartSpec = { series: LOSS_RUNS.map(run => ({ run, data: ema(model.train[run], smooth) })), yFmt: lossFmt, focus: view === "zoom", xRange };
    const val: ChartSpec = { series: LOSS_RUNS.map(run => ({ run, data: model.val[run] })), yFmt: lossFmt, symbol: "none", focus: view === "zoom", xRange };
    const market: ChartSpec = {
      series: [
        { run: ticker, name: "Closing price", data: model.close[ticker], area: true },
        { run: ticker, name: "Learning rate (×10⁻³)", data: model.lr[ticker], rightAxis: true },
      ],
      yFmt: v => "$" + Number(v.toFixed(2)),
      rightFmt: v => String(Number(v.toFixed(2))),
      xRange,
    };
    return { train, val, market };
  });

  let refreshing = false;
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      raw = BACKUP ? await loadBackup() : REPLAY ? await loadReplay() : DEMO ? demoData() : await loadSupabase();
      error = null;
    } catch (e) {
      console.error(e);
      error = e instanceof Error ? e.message : String(e);
    } finally {
      refreshing = false;
    }
  }

  onMount(() => {
    refresh();
    const poll = !BACKUP && (REPLAY || !DEMO) ? setInterval(refresh, REPLAY ? 3000 : REFRESH_MS) : undefined;
    return () => clearInterval(poll);
  });
</script>

<main class="page">
  <header>
    <div>
      <h1>Stockastic Descent</h1>
      <p class="tagline">Four identical LLM training runs, each with a learning rate that follows a stock.</p>
    </div>
    <div class="freshness" aria-live="polite">
      {#if error}<span class="err" title={error}>{raw ? "Refresh failed" : error}</span>{/if}
      {#if dataThrough && model}
        <span class="data-time">
          {#if !error}<i aria-hidden="true"></i>{/if}
          <span>Data through <time datetime={new Date(dataThrough).toISOString()}>{fmtET(dataThrough)}</time></span>
        </span>
      {:else if !raw && !error}<span>Loading…</span>{/if}
    </div>
  </header>

  <div class="progress-line" title="Training progress">
    <span>{fmtTokens((model?.maxStep ?? 0) * TOKENS_PER_STEP)} of {fmtTokens(TOTAL_STEPS * TOKENS_PER_STEP)} tokens</span>
    <span class="bar"><span class="fill" style:width={`${Math.min(100, (100 * (model?.maxStep ?? 0)) / TOTAL_STEPS)}%`}></span></span>
    <span>{Math.floor((100 * (model?.maxStep ?? 0)) / TOTAL_STEPS)}%</span>
  </div>

    {#snippet plot(key: "market" | "train" | "val")}
      <div class="plot-frame">
        {#if model && specs && model.maxStep > 0 && (key !== "val" || hasVal)}
          <Chart spec={specs[key]} {model} {hovered} onhover={run => (hovered = run)} />
        {:else}
          <div class="plot placeholder">{error ? "Couldn’t load data." : !raw ? "Loading…" : key === "val" ? "Awaiting first validation." : "No measurements yet."}</div>
        {/if}
      </div>
    {/snippet}

    <div class="grid-top">
      <section class="panel">
        <div class="panel-head">
          <h2>Validation loss</h2>
          <div class="seg" role="group" aria-label="Chart range">
            <button type="button" aria-pressed={view === "zoom"} onclick={() => (view = "zoom")} title="Recent steps, y-axis fitted to the data">Zoomed</button>
            <button type="button" aria-pressed={view === "full"} onclick={() => (view = "full")} title="Whole training schedule">Full run</button>
          </div>
        </div>
        {@render plot("val")}
      </section>
      <aside class="panel standings" aria-label="Validation loss leaderboard">
        <div class="panel-head">
          <h2>Leaderboard{#if standings}<span class="sub">step {standings.step.toLocaleString()}</span>{/if}</h2>
        </div>
        <ol>
          {#if standings}
            {#each standings.rows as row (row.run)}
              {@const rank = standings.rows.findIndex(other => other.loss === row.loss) + 1}
              <li class:leader={rank === 1} class:hot={hovered === row.run} class:dim={hovered != null && hovered !== row.run}
                  style:--c={`var(${COLOR_VAR[row.run]})`} title={"Validation loss: " + row.loss}
                  onmouseenter={() => (hovered = row.run)} onmouseleave={() => (hovered = null)}>
                <span class="rank" aria-label={`Rank ${rank}`}>{rank}</span>
                <span class="swatch" aria-hidden="true"></span>
                <span class="name">{PRICE_SOURCE[row.run]}{#if standings.moved[row.run] > 0}<span class="move up" title="Up {standings.moved[row.run]} since step {standings.since?.toLocaleString()}">▲{standings.moved[row.run]}</span>{:else if standings.moved[row.run] < 0}<span class="move down" title="Down {-standings.moved[row.run]} since step {standings.since?.toLocaleString()}">▼{-standings.moved[row.run]}</span>{/if}</span>
                <span class="score">{row.loss.toFixed(3)}</span>
              </li>
            {/each}
          {:else}
            {#each LOSS_RUNS as run (run)}
              <li style:--c={`var(${COLOR_VAR[run]})`}>
                <span class="rank">–</span><span class="swatch" aria-hidden="true"></span>
                <span class="name">{PRICE_SOURCE[run]}</span><span class="score">—</span>
              </li>
            {/each}
          {/if}
        </ol>
        <p class="note">Each run trains the same model; its learning rate follows the named ticker.</p>
      </aside>
    </div>

    <div class="grid-bottom">
      <section class="panel">
        <div class="panel-head">
          <h2>Training loss</h2>
          <div class="control">
            <span class="seg-label">Smoothing</span>
            <div class="seg" role="group" aria-label="Training loss smoothing">
              {#each [[1, "Raw"], [20, "20"], [100, "100"], [400, "400"]] as [n, label] (n)}
                <button type="button" aria-pressed={smooth === n} onclick={() => (smooth = n as number)}>{label}</button>
              {/each}
            </div>
          </div>
        </div>
        {@render plot("train")}
      </section>
      <section class="panel">
        <div class="panel-head">
          <div class="axis-key">
            <span style:--series-color={`var(${COLOR_VAR[ticker]})`}><strong>{PRICE_SOURCE[ticker]}</strong>&nbsp;price</span>
            <span class="dashed">Learning rate (×10⁻³)</span>
          </div>
          <div class="seg" role="group" aria-label="Price and learning rate ticker">
            {#each PRICED_RUNS as run}
              <button type="button" aria-pressed={ticker === run} onclick={() => ticker = run}>{PRICE_SOURCE[run]}</button>
            {/each}
          </div>
        </div>
        {@render plot("market")}
      </section>
    </div>

</main>
