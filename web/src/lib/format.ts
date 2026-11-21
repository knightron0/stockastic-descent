const tz = "America/New_York";
const etFull = new Intl.DateTimeFormat("en-US", { timeZone: tz, weekday: "short", month: "numeric", day: "numeric", hour: "numeric", minute: "2-digit" });

/** 82.6M, 614M, 1.2B */
export const fmtTokens = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(1).replace(/\.0$/, "")}B` : n >= 1e8 ? `${Math.round(n / 1e6)}M` : n >= 1e6 ? `${(n / 1e6).toFixed(1).replace(/\.0$/, "")}M` : n >= 1e3 ? `${Math.round(n / 1e3)}K` : `${n}`;

export const fmtET = (ms: number | null | undefined) => (ms == null ? "—" : `${etFull.format(ms)} ET`);

// Market session (regular hours, weekdays; exchange holidays not modeled). Mirrors stocklr/market.py and
// the cron in modal_app.py, which first fires at 10:02 ET (trains 9:30-9:45 once its delayed bars exist).
const etParts = new Intl.DateTimeFormat("en-US", { timeZone: tz, weekday: "short", hour: "numeric", minute: "numeric", hourCycle: "h23" });
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const etClock = (ms: number) => {
  const p = Object.fromEntries(etParts.formatToParts(ms).map(x => [x.type, x.value]));
  return { day: WEEKDAYS.indexOf(p.weekday), mins: Number(p.hour) * 60 + Number(p.minute) };
};
const FIRST_UPDATE = 10 * 60 + 2;

export const marketOpen = (ms: number) => {
  const { day, mins } = etClock(ms);
  return day >= 1 && day <= 5 && mins >= 9 * 60 + 30 && mins < 16 * 60;
};

/** "today at 10:02 AM ET", "Mon at 10:02 AM ET": the first cron run of the next session */
export const nextUpdate = (ms: number) => {
  const { day, mins } = etClock(ms);
  if (day >= 1 && day <= 5 && mins < FIRST_UPDATE) return "today at 10:02 AM ET";
  let d = day;
  do d = (d + 1) % 7; while (d === 0 || d === 6);
  return `${d === (day + 1) % 7 ? "tomorrow" : WEEKDAYS[d]} at 10:02 AM ET`;
};
