# stockastic descent: dashboard

Vite + Svelte 5 + TypeScript, charts with Apache ECharts, data from Supabase via supabase-js.
The browser reads the tables with the **anon / publishable** key, which row-level security limits to `SELECT`.

## Run locally
```bash
cd web
npm install
cp .env.example .env.local     # fill in VITE_SUPABASE_URL + VITE_SUPABASE_ANON_KEY (never the service key)
npm run dev                    # http://localhost:5173  (add ?demo for synthetic data)
```

URL flags:

| flag | effect |
| --- | --- |
| `?demo` | synthetic data, no network |
| `?replay=<project-ref>/<replay-id>` | open a published replay |
| `?backup=<tag>` | load `public/backups/<tag>.json` (from `research/backup_to_dashboard.py`) |
| `?test` | show `test-*` runs kept by `uv run modal run modal_app.py --keep` |
| `?view=full` | start on the full-schedule view instead of the zoomed one |
| `?smooth=1\|20\|100\|400` | initial training-loss EMA span |

Other scripts: `npm run check` (svelte-check / types), `npm run build` (→ `dist/`), `npm run preview`.

## Deploy (GitHub Pages)
`.github/workflows/deploy-web.yml` builds `web/` and publishes it on every push to `main` that touches `web/`.
One-time setup in the repo:
1. **Settings → Pages → Source: GitHub Actions.**
2. **Settings → Secrets and variables → Actions:** add `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`.

`vite.config.ts` uses `base: "./"`, so the build works from the `/<repo>/` sub-path. The site is a static
bundle, so Vercel/Netlify also work (root directory `web`, build `npm run build`, output `dist`, same env vars).

## Layout
```
src/App.svelte            the page: header, progress, validation loss + leaderboard, training loss, price/LR
src/app.css               all styles; the page fits the viewport (no scroll) and clamps chart aspect ratios
src/lib/config.ts         runs, price sources, colours, schedule, URL flags (mirror of stocklr/config.py + lr.py)
src/lib/data.ts           Supabase loading (paginated, 1000 rows/page)
src/lib/replay.ts         published-replay loading;  backup.ts: local backup files;  demo.ts: synthetic rows
src/lib/shape.ts          rows -> step-aligned series, EMA
src/lib/chartOptions.ts   ECharts option builder (tooltip, zoomed y-fit)
src/lib/Chart.svelte      ECharts wrapper; all charts share hover
```
