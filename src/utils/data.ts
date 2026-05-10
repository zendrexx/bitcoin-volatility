
import fs from "fs";
import path from "path";
export type Timeframe = "5m" | "15m" | "30m";

export type PricePoint = { timestamp: string; price: number };
export type VolPoint = { timestamp: string; log_return: number; rolling_volatility: number };

export type HmmState = "LOW" | "MEDIUM" | "HIGH";

export type HmmStateResponse = {
  current_state: HmmState;
  probabilities: { low: number; medium: number; high: number };
  transition_matrix: number[][];
};

export type VarResponse = {
  var95: number;
  var99: number;
  interpretation95: string;
  interpretation99: string;
};



function round(n: number, dp = 4) {
  const p = 10 ** dp;
  return Math.round(n * p) / p;
}


export function parseTimeframe(input: unknown): Timeframe {
  const v = String(input ?? "").toLowerCase();
  if (v === "30m" || v === "30") return "30m";
  if (v === "15m" || v === "15") return "15m";
  return "5m";
}


export function loadRawBTC(tf: Timeframe) {
  const file = path.join(
    process.cwd(),
    "data/raw",
    `btc_${tf}_1year.json`
  );

  const raw = JSON.parse(fs.readFileSync(file, "utf-8"));

  return raw.map((d: any) => ({
    timestamp: d.timestamp,
    open: Number(d.open),
    high: Number(d.high),
    low: Number(d.low),
    close: Number(d.close),
    volume: Number(d.volume),
  }));
}
export function computeVolatilityFromRaw(
  prices: { timestamp: string; close: number }[],
  window = 20
): VolPoint[] {
  const returns: number[] = [];

  for (let i = 1; i < prices.length; i++) {
    returns.push(Math.log(prices[i].close / prices[i - 1].close));
  }

  const vol: number[] = [];

  for (let i = 0; i < returns.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = returns.slice(start, i + 1);

    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;

    const variance =
      slice.reduce((acc, r) => acc + (r - mean) ** 2, 0) /
      Math.max(1, slice.length - 1);

    vol.push(Math.sqrt(variance));
  }

  return returns.map((r, i) => ({
    timestamp: prices[i + 1].timestamp,
    log_return: r,
    rolling_volatility: vol[i],
  }));
}

export function loadHmm(tf: Timeframe) {
  const file = path.join(
    process.cwd(),
    "data/processed",
    `hmm_${tf}.json`
  );

  const raw = JSON.parse(fs.readFileSync(file, "utf-8"));

  return raw.map((d: any) => ({
    timestamp: d.timestamp,
    state: d.state, // LOW / MEDIUM / HIGH
    probabilities: d.probabilities,
  }));
}



