import type { NextApiRequest, NextApiResponse } from "next";
import { parseTimeframe } from "@/utils/data";
import { loadDataset } from "@/utils/dataset";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const tf = parseTimeframe(req.query.timeframe);

  const data = loadDataset(tf);

  res.status(200).json(
    data.map(d => ({
      timestamp: d.timestamp,
      price: d.close
    }))
  );
}