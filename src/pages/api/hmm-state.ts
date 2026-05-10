import type { NextApiRequest, NextApiResponse } from "next";
import path from "path";
import fs from "fs";
import { parseTimeframe } from "@/utils/data";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    const tf = parseTimeframe(req.query.timeframe);

    const file = path.join(
      process.cwd(),
      "data/processed",
      `hmm_${tf}.json`
    );

    const raw = fs.readFileSync(file, "utf-8");
    const data = JSON.parse(raw);

    // optional: return ONLY latest state OR full series
    const latest = data.at(-1);

    res.status(200).json({
      current_state: latest.state,
      probabilities: latest.probabilities,
      transition_matrix: latest.transition_matrix ?? [],
      series: data, // optional if your UI needs history
    });
  } catch (err) {
    res.status(500).json({
      error: "Failed to load HMM data",
    });
  }
}