import type { NextApiRequest, NextApiResponse } from "next";
import fs from "fs";
import path from "path";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    const timeframe = String(req.query.timeframe ?? "5m");

    // ── 1. Raw OHLCV (array-of-arrays format) ──────────────────────────────
    const rawFile = path.join(process.cwd(), "data", "raw", `btc_${timeframe}_1year.json`);
    if (!fs.existsSync(rawFile)) {
      console.error("Missing raw file:", rawFile);
      return res.status(500).json({ error: `Raw file not found: btc_${timeframe}_1year.json` });
    }
    const raw: any[][] = JSON.parse(fs.readFileSync(rawFile, "utf-8"));

    // ── 2. Processed (object format with state/volatility) ─────────────────
    // Try common naming conventions
    const procCandidates = [
      path.join(process.cwd(), "data", "processed", `btc_${timeframe}_1year.json`),
      path.join(process.cwd(), "data", "processed", `btc_${timeframe}.json`),
      path.join(process.cwd(), "data", "processed", `hmm_${timeframe}.json`),
    ];
    const procFile = procCandidates.find(fs.existsSync);
    if (!procFile) {
      console.error("Missing processed file, tried:", procCandidates);
      return res.status(500).json({
        error: `Processed file not found for timeframe: ${timeframe}`,
        tried: procCandidates,
      });
    }
    const processed: any[] = JSON.parse(fs.readFileSync(procFile, "utf-8"));

    // ── 3. Build state lookup: ISO timestamp → row ─────────────────────────
    const stateMap = new Map<string, any>();
    for (const row of processed) {
      // Processed uses "time" key based on your sample
      const raw_ts = row.time ?? row.timestamp ?? "";
      if (!raw_ts) continue;
      // Normalize: strip trailing Z issues, always compare as ISO
      const key = new Date(raw_ts).toISOString();
      stateMap.set(key, row);
    }

    console.log(`[dataset] raw rows: ${raw.length}, processed rows: ${processed.length}, stateMap size: ${stateMap.size}`);

    // ── 4. Merge ───────────────────────────────────────────────────────────
    const candles = raw
      .slice(-500)
      .map((d) => {
        const ts = new Date(d[0]).toISOString();
        const proc = stateMap.get(ts);
        return {
          timestamp:  ts,
          open:       Number(d[1]),
          high:       Number(d[2]),
          low:        Number(d[3]),
          close:      Number(d[4]),
          volume:     Number(d[5]),
          state:          proc?.state       ?? null,
          log_return:     proc?.log_return  ?? null,
          volatility:     proc?.volatility  ?? null,
        };
      })
      .filter((d) => d.state !== null); // only rows with matched state

    console.log(`[dataset] merged candles: ${candles.length}`);

    // ── 5. Map raw state numbers → sorted volatility rank ─────────────────
    // HMM state labels (0,1,2) are arbitrary — sort by mean volatility
    // so that rank 0 = lowest vol, 1 = medium, 2 = highest vol
    const volByState: Record<number, number[]> = {};
    for (const c of candles) {
      if (c.state === null || c.volatility === null) continue;
      if (!volByState[c.state]) volByState[c.state] = [];
      volByState[c.state].push(c.volatility);
    }

    // mean volatility per raw state
    const meanVol = Object.entries(volByState).map(([s, vals]) => ({
      rawState: Number(s),
      mean: vals.reduce((a, b) => a + b, 0) / vals.length,
    }));

    // sort ascending → index in sorted array = volatility rank (0=low, 2=high)
    meanVol.sort((a, b) => a.mean - b.mean);
    const stateRankMap: Record<number, number> = {};
    meanVol.forEach(({ rawState }, rank) => { stateRankMap[rawState] = rank; });

    console.log("[dataset] state→rank map:", stateRankMap, "mean vols:", meanVol);

    // Remap each candle's state to its volatility rank
    const ranked = candles.map((c) => ({
      ...c,
      state: c.state !== null ? (stateRankMap[c.state] ?? c.state) : null,
    }));

    if (candles.length === 0) {
      // Timestamps didn't match — log samples to debug
      const rawSample = new Date(raw[0][0]).toISOString();
      const procSample = new Date(processed[0].time ?? processed[0].timestamp).toISOString();
      console.error("Timestamp mismatch! raw sample:", rawSample, "| processed sample:", procSample);
      return res.status(500).json({
        error: "Timestamps did not match between raw and processed files",
        raw_sample: rawSample,
        processed_sample: procSample,
      });
    }

    res.status(200).json(ranked);
  } catch (err) {
    console.error("Dataset handler error:", err);
    res.status(500).json({ error: "Failed to load dataset", detail: String(err) });
  }
}