<script lang="ts">
  import { onMount } from "svelte";
  import { echarts, GROUP } from "./echarts";
  import { buildOption, type ChartSpec } from "./chartOptions";
  import type { Model } from "./types";

  let { spec, model }: { spec: ChartSpec; model: Model } = $props();

  let el: HTMLDivElement;
  let chart = $state<ReturnType<typeof echarts.init>>();

  onMount(() => {
    const c = echarts.init(el, null, { renderer: "canvas" });
    c.group = GROUP;
    echarts.connect(GROUP);                          // shared hover across every chart
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el);
    chart = c;
    return () => { ro.disconnect(); c.dispose(); chart = undefined; };
  });

  $effect(() => { chart?.setOption(buildOption(spec, model), { notMerge: true }); });
</script>

<div class="plot" bind:this={el}></div>
