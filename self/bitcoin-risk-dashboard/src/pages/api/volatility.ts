import type { NextApiRequest, NextApiResponse } from "next";
import {
  generatePriceHistory,
  generateVolatility,
  parseTimeframe,
} from "@/utils/mockData";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const tf = parseTimeframe(req.query.timeframe);
  const prices = generatePriceHistory(tf, 240);
  const items = generateVolatility(tf, prices, 20);
  res.status(200).json(items);
}

