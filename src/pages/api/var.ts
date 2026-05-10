import type { NextApiRequest, NextApiResponse } from "next";
import { parseTimeframe } from "@/utils/data";
import { loadDataset } from "@/utils/dataset";

function computeVar(vol: number, tf: string) {
  const z95 = 1.65;
  const z99 = 2.33;

  const tfScale = tf === "5m" ? 1 : tf === "10m" ? 1.15 : 1.3;
  const sigma = Math.max(0.0004, vol * tfScale);

  const var95 = -z95 * sigma;
  const var99 = -z99 * sigma;

  return {
    var95,
    var99,
    interpretation95: `5% chance loss exceeds ${(Math.abs(var95) * 100).toFixed(2)}%`,
    interpretation99: `1% chance loss exceeds ${(Math.abs(var99) * 100).toFixed(2)}%`,
  };
}

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const tf = parseTimeframe(req.query.timeframe);
  const data = loadDataset(tf);

  const latestVol = data.at(-1)?.volatility ?? 0.0015;

  const result = computeVar(latestVol, tf);

  res.status(200).json(result);
}