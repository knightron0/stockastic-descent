const tz = "America/New_York";
const etFull = new Intl.DateTimeFormat("en-US", { timeZone: tz, weekday: "short", month: "numeric", day: "numeric", hour: "numeric", minute: "2-digit" });

export const fmtET = (ms: number | null | undefined) => (ms == null ? "—" : `${etFull.format(ms)} ET`);
