import { COLOR_VAR, TOTAL_STEPS, type Run } from "./config";
import { fmtET } from "./format";
import type { Model, Series } from "./types";

export interface ChartSpec {
  series: { run: Run; data: Series; name?: string; rightAxis?: boolean; area?: boolean }[];
  yFmt: (v: number) => string;
  rightFmt?: (v: number) => string;
  symbol?: "none";
  /** fit the left y-axis to the data inside the x window (so early high losses don't flatten the curves) */
  focus?: boolean;
  /** visible step range; defaults to the full schedule */
  xRange?: [number, number];
}

/** #rrggbb + alpha -> rgba() (ECharts draws on canvas, so CSS color-mix is not available) */
function alpha(hex: string, a: number) {
  const n = parseInt(hex.replace("#", ""), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/** Padded [min, max] of the series values whose step lies in [lo, hi]. */
function fitY(spec: ChartSpec, lo: number, hi: number, rightAxis = false): [number, number] | null {
  let min = Infinity, max = -Infinity;
  for (const s of spec.series) {
    if (Boolean(s.rightAxis) !== rightAxis) continue;
    for (let j = Math.max(0, lo - 1); j < Math.min(s.data.length, hi); j++) {
      const v = s.data[j];
      if (v == null) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (!Number.isFinite(min)) return null;
  const pad = Math.max((max - min) * 0.08, 1e-3);
  return [min - pad, max + pad];
}

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const escape = (s: string) => s.replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]!);

export function buildOption(spec: ChartSpec, model: Model) {
  const c = { ink: css("--ink"), ink3: css("--ink-3"), rule: css("--rule"), panel: css("--panel") };
  const font = css("--f-body") || "sans-serif", mono = css("--f-mono") || "monospace";
  const { yFmt } = spec;
  const axisLine = { show: true, onZero: false, lineStyle: { color: c.rule, width: 1 } };
  const [xMin, xMax] = spec.xRange ?? [1, TOTAL_STEPS];
  const yFit = spec.focus ? fitY(spec, xMin, xMax) : null;

  const rightFit = spec.rightFmt ? fitY(spec, xMin, xMax, true) : null;

  return {
    animation: false,
    textStyle: { fontFamily: font },
    grid: { left: 4, right: spec.rightFmt ? 8 : 22, top: 12, bottom: 4, containLabel: true },
    tooltip: {
      trigger: "axis", confine: true, order: "valueDesc",
      backgroundColor: c.panel, borderColor: c.rule, borderWidth: 1, padding: [8, 10],
      textStyle: { color: c.ink, fontFamily: font, fontSize: 12 },
      extraCssText: "box-shadow: 0 8px 28px rgba(16,24,40,.14); border-radius: 10px;",
      axisPointer: { type: "line", lineStyle: { color: c.ink3, width: 1, type: [3, 3], opacity: 0.7 } },
      formatter: (params: { axisValue: number; color: string; seriesName: string; seriesIndex: number; value: [number, number] }[]) => {
        if (!params.length) return "";
        const step = Math.round(params[0].axisValue);
        const rows = params.filter((p) => p.value?.[1] != null).map((p) =>
          `<div class="tt-row"><span class="tt-dot" style="background:${p.color}"></span>` +
          `<span class="tt-name">${escape(p.seriesName)}</span><span class="tt-val">${(spec.series[p.seriesIndex]?.rightAxis && spec.rightFmt || yFmt)(p.value[1])}</span></div>`).join("");
        return `<div class="tt-head">Step ${step.toLocaleString()}<span>${fmtET(model.stepTs[step - 1])}</span></div>` +
          (rows || `<div class="tt-none">no data at this step</div>`);
      },
    },
    xAxis: {
      type: "value", splitNumber: 5, min: xMin, max: xMax,
      axisLine, axisTick: { show: false }, splitLine: { show: false },
      axisLabel: { color: c.ink3, fontFamily: mono, fontSize: 11, margin: 10, hideOverlap: true, showMinLabel: false,
        showMaxLabel: xMax === TOTAL_STEPS,            // zoomed, the max is the live step (1,260): end on a round tick
        formatter: (v: number) => v >= 1000 && v % 100 === 0 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + "k" : Math.round(v).toLocaleString() },
    },
    yAxis: [{
      type: "value", splitNumber: 4, scale: true, boundaryGap: ["4%", "4%"],
      ...(yFit ? { min: yFit[0], max: yFit[1] } : {}),
      splitLine: { show: true, lineStyle: { color: c.rule, width: 1, type: [3, 4] } },
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: c.ink3, fontFamily: mono, fontSize: 11, margin: 10, hideOverlap: true, showMinLabel: !yFit, showMaxLabel: !yFit, formatter: yFmt },

    }, ...(spec.rightFmt ? [{
      type: "value", position: "right", scale: true, splitNumber: 4, boundaryGap: ["4%", "4%"],
      ...(rightFit ? { min: Math.max(0, rightFit[0]), max: rightFit[1] } : {}),
      splitLine: { show: false }, axisLine, axisTick: { show: false },
      axisLabel: { color: c.ink3, fontFamily: mono, fontSize: 11, margin: 10, hideOverlap: true, showMinLabel: rightFit ? rightFit[0] <= 0 : true, showMaxLabel: !rightFit, formatter: spec.rightFmt },
    }] : [])],
    series: spec.series.map(({ run, data, name, rightAxis, area }) => {
      const color = rightAxis ? c.ink3 : css(COLOR_VAR[run]);
      const pts: [number, number][] = [];
      data.forEach((v, j) => { if (v != null) pts.push([j + 1, v]); });   // only real points: sparse series stay visible
      return {
        name: name ?? run, yAxisIndex: rightAxis ? 1 : 0, type: "line", data: pts, connectNulls: true,
        sampling: spec.symbol === "none" ? undefined : "lttb",
        showSymbol: false, symbol: spec.symbol ?? "circle", symbolSize: 4,
        triggerLineEvent: true,   // hovering the line itself (not just a point) reports the run for cross-panel highlight
        lineStyle: { color, width: rightAxis ? 1.4 : 2, type: rightAxis ? [5, 4] : "solid", cap: "round", join: "round" },
        itemStyle: { color },
        ...(area ? { areaStyle: { origin: "start", color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [{ offset: 0, color: alpha(color, 0.18) }, { offset: 1, color: alpha(color, 0) }] } } } : {}),
        emphasis: { focus: "series", lineStyle: { width: 2.8 } },
        blur: { lineStyle: { opacity: 0.25 }, itemStyle: { opacity: 0.25 } },
        z: rightAxis ? 3 : 2,
      };
    }),
  };
}
