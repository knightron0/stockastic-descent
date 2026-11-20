import { REPLAY, TOTAL_STEPS } from "./config";
import type { Raw } from "./types";

let loadedChunks = 0;
let snapshot: Raw = { jobs: [], prices: [], steps: [], evals: [] };
let complete = false;

export async function loadReplay(): Promise<Raw> {
  if (complete) return snapshot;
  const parts = /^([a-z0-9]+)\/([a-f0-9]{32})$/.exec(REPLAY);
  if (!parts) throw new Error("Invalid replay identifier");
  const base = `https://${parts[1]}.supabase.co/storage/v1/object/public/stock-lr-replays/${parts[2]}/`;
  async function read<T>(file: string): Promise<T> {
    const response = await fetch(base + file);
    if (!response.ok) throw new Error(`Replay file unavailable (${response.status})`);
    return response.json();
  }

  const { chunks } = await read<{ chunks: number[] }>("manifest.json?t=" + Date.now());
  while (loadedChunks < chunks.length) {
    const pending = chunks.slice(loadedChunks, loadedChunks + 6);
    const batches = await Promise.all(pending.map(step => read<Raw>(step + ".json")));
    snapshot = {
      steps: snapshot.steps.concat(batches.flatMap(batch => batch.steps)),
      prices: snapshot.prices.concat(batches.flatMap(batch => batch.prices)),
      evals: snapshot.evals.concat(batches.flatMap(batch => batch.evals)),
      jobs: batches.flatMap(batch => batch.jobs).reverse().concat(snapshot.jobs),
    };
    loadedChunks += batches.length;
  }
  complete = chunks[chunks.length - 1] === TOTAL_STEPS;
  return snapshot;
}
