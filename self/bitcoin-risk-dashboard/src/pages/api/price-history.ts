import type { NextApiRequest, NextApiResponse } from "next";
import { generatePriceHistory, parseTimeframe } from "@/utils/mockData";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const tf = parseTimeframe(req.query.timeframe);
  const items = generatePriceHistory(tf, 240);
  res.status(200).json(items);
}

