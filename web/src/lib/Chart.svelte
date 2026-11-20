<script lang="ts">
  import { onMount } from "svelte";
  import { echarts, GROUP } from "./echarts";
  import { buildOption, type ChartSpec } from "./chartOptions";
  import type { Run } from "./config";
  import type { Model } from "./types";

  let { spec, model, hovered = null, onhover }: {
    spec: ChartSpec; model: Model; hovered?: Run | null; onhover?: (run: Run | null) => void;
  } = $props();

  let el: HTMLDivElement;
  let chart = $state<ReturnType<typeof echarts.init>>();
  let fontsReady = $state(false);
  document.fonts?.ready.then(() => (fontsReady = true));

  onMount(() => {
    const c = echarts.init(el, null, { renderer: "canvas" });
    c.group = GROUP;
    echarts.connect(GROUP);                          // shared hover across every chart
    // report which run the pointer is on; the parent feeds it back as `hovered` to every chart
    c.on("mouseover", (p: { componentType: string; seriesIndex?: number }) => {
      if (p.componentType === "series" && p.seriesIndex != null) onhover?.(spec.series[p.seriesIndex]?.run ?? null);
    });
    c.on("mouseout", () => onhover?.(null));
    c.on("globalout", () => onhover?.(null));
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el);
    chart = c;
    return () => { ro.disconnect(); c.dispose(); chart = undefined; };
  });

  $effect(() => { void fontsReady; chart?.setOption(buildOption(spec, model), { notMerge: true }); applyHover(); });
  $effect(() => { void hovered; applyHover(); });

  // silent: connected charts re-broadcast action events by seriesIndex, which would light up the wrong line in
  // other panels; each chart applies the shared `hovered` itself instead.
  function applyHover() {
    if (!chart) return;
    const all = spec.series.map((_, i) => i);
    const hit = all.filter(i => spec.series[i].run === hovered);
    chart.dispatchAction({ type: "downplay", seriesIndex: all }, { silent: true });
    if (hit.length) chart.dispatchAction({ type: "highlight", seriesIndex: hit }, { silent: true });
  }
</script>

<div class="plot" bind:this={el}></div>
