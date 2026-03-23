import { hashSeed, mulberry32 } from "@/utils/prng";

export type Timeframe = "5m" | "10m" | "15m";

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

function clamp(n: number, min: number, max: number) {
  return Math.min(max, Math.max(min, n));
}

function round(n: number, dp = 4) {
  const p = 10 ** dp;
  return Math.round(n * p) / p;
}

function normalize3(a: number, b: number, c: number) {
  const s = a + b + c || 1;
  return [a / s, b / s, c / s] as const;
}

export function parseTimeframe(input: unknown): Timeframe {
  const v = String(input ?? "").toLowerCase();
  if (v === "15m" || v === "15") return "15m";
  if (v === "10m" || v === "10") return "10m";
  return "5m";
}

function timeframeMs(tf: Timeframe) {
  if (tf === "15m") return 15 * 60 * 1000;
  if (tf === "10m") return 10 * 60 * 1000;
  return 5 * 60 * 1000;
}

function gaussian(rng: () => number) {
  // Box–Muller
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function generatePriceHistory(tf: Timeframe, n = 240): PricePoint[] {
  const seed = hashSeed(`price:${tf}`);
  const rng = mulberry32(seed);

  const now = Date.now();
  const step = timeframeMs(tf);
  const start = now - (n - 1) * step;

  // Start near a plausible BTC level, plus small deterministic variation
  let price = 65000 + Math.round((rng() - 0.5) * 8000);
  const driftPerStep = (rng() - 0.5) * 8; // mild drift
  const vol = tf === "5m" ? 0.0022 : tf === "10m" ? 0.0029 : 0.0034;

  const points: PricePoint[] = [];
  for (let i = 0; i < n; i++) {
    const t = start + i * step;
    const shock = gaussian(rng) * vol;
    price = Math.max(5000, price * (1 + driftPerStep / 10000 + shock));
    points.push({
      timestamp: new Date(t).toISOString(),
      price: Math.round(price * 100) / 100,
    });
  }
  return points;
}

export function generateVolatility(tf: Timeframe, prices: PricePoint[], window = 20): VolPoint[] {
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    const p0 = prices[i - 1]!.price;
    const p1 = prices[i]!.price;
    returns.push(Math.log(p1 / p0));
  }

  const vol: number[] = [];
  for (let i = 0; i < returns.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = returns.slice(start, i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance =
      slice.reduce((acc, r) => acc + (r - mean) ** 2, 0) / Math.max(1, slice.length - 1);
    vol.push(Math.sqrt(variance));
  }

  // Align to timestamps (skip first price timestamp)
  return returns.map((r, idx) => ({
    timestamp: prices[idx + 1]!.timestamp,
    log_return: round(r, 5),
    rolling_volatility: round(vol[idx]!, 5),
  }));
}

export function classifyHmmState(volLatest: number) {
  // Heuristic thresholds for demo; not "prediction", just regime labeling.
  if (volLatest < 0.0016) return "LOW" as const;
  if (volLatest < 0.0032) return "MEDIUM" as const;
  return "HIGH" as const;
}

export function generateHmmState(tf: Timeframe, volLatest: number): HmmStateResponse {
  const seed = hashSeed(`hmm:${tf}`);
  const rng = mulberry32(seed);

  const state = classifyHmmState(volLatest);
  const base =
    state === "LOW"
      ? normalize3(0.74, 0.2, 0.06)
      : state === "MEDIUM"
        ? normalize3(0.18, 0.62, 0.2)
        : normalize3(0.08, 0.28, 0.64);

  const jitter = () => (rng() - 0.5) * 0.06;
  const [l, m, h] = normalize3(
    clamp(base[0] + jitter(), 0.01, 0.98),
    clamp(base[1] + jitter(), 0.01, 0.98),
    clamp(base[2] + jitter(), 0.01, 0.98),
  );

  // Transition matrix: diagonals dominant, slightly adjusted by regime
  const mat =
    state === "LOW"
      ? [
          [0.82, 0.14, 0.04],
          [0.21, 0.63, 0.16],
          [0.1, 0.28, 0.62],
        ]
      : state === "MEDIUM"
        ? [
            [0.74, 0.2, 0.06],
            [0.18, 0.66, 0.16],
            [0.08, 0.3, 0.62],
          ]
        : [
            [0.7, 0.22, 0.08],
            [0.15, 0.6, 0.25],
            [0.1, 0.28, 0.62],
          ];

  return {
    current_state: state,
    probabilities: { low: round(l, 2), medium: round(m, 2), high: round(h, 2) },
    transition_matrix: mat,
  };
}

export function generateVar(tf: Timeframe, volLatest: number): VarResponse {
  // Simplified VaR approximation: VaR ≈ -z * sigma (for log returns)
  const z95 = 1.65;
  const z99 = 2.33;

  // Scale volatility slightly with timeframe for UI readability
  const tfScale = tf === "5m" ? 1 : tf === "10m" ? 1.15 : 1.3;
  const sigma = Math.max(0.0004, volLatest * tfScale);

  const var95 = -z95 * sigma;
  const var99 = -z99 * sigma;

  const p95 = Math.abs(var95) * 100;
  const p99 = Math.abs(var99) * 100;

  return {
    var95: round(var95, 4),
    var99: round(var99, 4),
    interpretation95: `This means there is a 5% chance the loss will exceed ${p95.toFixed(
      2,
    )}% within the next interval.`,
    interpretation99: `This means there is a 1% chance the loss will exceed ${p99.toFixed(
      2,
    )}% within the next interval.`,
  };
}

