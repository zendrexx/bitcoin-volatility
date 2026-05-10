import type { NextApiRequest, NextApiResponse } from "next";
import { parseTimeframe } from "@/utils/data";
import { loadDataset } from "@/utils/dataset";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const tf = parseTimeframe(req.query.timeframe);
  const data = loadDataset(tf);

  res.status(200).json(
    data
      .filter(d => d.volatility !== undefined)
      .map(d => ({
        timestamp: d.timestamp,
        log_return: d.log_return,
        rolling_volatility: d.volatility,
      }))
  );
}