import type { NextApiRequest, NextApiResponse } from "next";
import {
  generateHmmState,
  generatePriceHistory,
  generateVolatility,
  parseTimeframe,
} from "@/utils/mockData";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const tf = parseTimeframe(req.query.timeframe);
  const prices = generatePriceHistory(tf, 240);
  const vol = generateVolatility(tf, prices, 20);
  const latest = vol.at(-1)?.rolling_volatility ?? 0.0015;
  const payload = generateHmmState(tf, latest);
  res.status(200).json(payload);
}

