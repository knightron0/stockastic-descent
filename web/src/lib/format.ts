const tz = "America/New_York";
const etFull = new Intl.DateTimeFormat("en-US", { timeZone: tz, weekday: "short", month: "numeric", day: "numeric", hour: "numeric", minute: "2-digit" });

/** 82.6M, 614M, 1.2B */
export const fmtTokens = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(1).replace(/\.0$/, "")}B` : n >= 1e8 ? `${Math.round(n / 1e6)}M` : n >= 1e6 ? `${(n / 1e6).toFixed(1).replace(/\.0$/, "")}M` : n >= 1e3 ? `${Math.round(n / 1e3)}K` : `${n}`;

export const fmtET = (ms: number | null | undefined) => (ms == null ? "—" : `${etFull.format(ms)} ET`);
