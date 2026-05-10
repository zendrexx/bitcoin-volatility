import fs from "fs";
import path from "path";

export type DatasetRow = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  log_return?: number;
  volatility?: number;
  state?: string;
};

export function loadDataset(tf: string): DatasetRow[] {
  const file = path.join(
    process.cwd(),
    "data/processed",
    `hmm_${tf}.json`
  );

  const raw = fs.readFileSync(file, "utf-8");
  return JSON.parse(raw);
}