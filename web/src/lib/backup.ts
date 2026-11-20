import { BACKUP } from "./config";
import type { Raw } from "./types";

/** A saved run exported by research/backup_to_dashboard.py, served from web/public/backups/. */
export async function loadBackup(): Promise<Raw> {
  const r = await fetch(`${import.meta.env.BASE_URL}backups/${BACKUP}.json`);
  if (!r.ok) throw new Error(`backup "${BACKUP}" not found (${r.status}): run research/backup_to_dashboard.py ${BACKUP}`);
  return r.json();
}
