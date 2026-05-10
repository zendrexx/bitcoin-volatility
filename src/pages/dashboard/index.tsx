import { useEffect, useMemo, useState } from "react";
import { DashboardLayout } from "@/components/dashboard/DashboardLayout";
import { Card } from "@/components/ui/Card";
import { TimeframeSelector } from "@/components/dashboard/TimeframeSelector";
import { PriceChart } from "@/charts/PriceChart";
import { VolatilityChart } from "@/charts/VolatilityChart";
import { HMMStatePanel } from "@/components/dashboard/HMMStatePanel";
import { TransitionMatrixTable } from "@/components/dashboard/TransitionMatrixTable";
import { RiskCard } from "@/components/dashboard/RiskCard";
import { apiGet } from "@/utils/api";
import type {
  HmmStateResponse,
  PricePoint,
  Timeframe,
  VarResponse,
  VolPoint,
} from "@/utils/data";

const TF_OPTIONS: Timeframe[] = ["5m", "15m", "30m"];
type Candle = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  state: number | null;
  log_return?: number | null;   
  volatility?: number | null;   
};
export default function DashboardPage() {
  const [timeframe, setTimeframe] = useState<Timeframe>("5m");
  const [prices, setPrices] = useState<Candle[]>([]);
  const [vol, setVol] = useState<VolPoint[]>([]);
  const [hmm, setHmm] = useState<HmmStateResponse | null>(null);
  const [varData, setVarData] = useState<VarResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
const DISPLAY_LIMIT = 70;

const visibleData = prices.slice(-DISPLAY_LIMIT);
const currentPrice = prices.length
  ? prices[prices.length - 1].close
  : null;
useEffect(() => {
  let cancelled = false;

  async function load() {
    setLoading(true);
    setError(null);

    try {
      const data = await apiGet<any[]>("/api/dataset", { timeframe });

      if (cancelled) return;

      if (!data || data.length === 0) {
        throw new Error("Empty dataset received");
      }

      setPrices(
          data.map((d) => ({
            timestamp: d.timestamp,
            open:      d.open,
            high:      d.high,
            low:       d.low,
            close:     d.close, 
            state:     d.state ?? null,
            log_return: d.log_return ?? null,
            volatility: d.volatility ?? null,
          }))
        );

        // ✅ Keep volatility fields for VolatilityChart
        setVol(
          data.map((d) => ({
            timestamp:          d.timestamp,
            log_return:         d.log_return      ?? null,
            rolling_volatility: d.volatility      ?? null,
          }))
        );

      const last = data.at(-1);

      if (!last) throw new Error("No latest data point");

      setHmm({
        current_state: last.state,
        probabilities: last.probabilities ?? {
          low: 0,
          medium: 0,
          high: 0,
        },
        transition_matrix: [],
      });

    } catch (err: any) {
      console.error("Dataset load error:", err);
      setError(err.message ?? "Failed to load data");
    } finally {
      if (!cancelled) setLoading(false);
    }
  }

  load();

  return () => {
    cancelled = true;
  };
}, [timeframe]);
  return (
    <DashboardLayout title="Dashboard" currentPrice={currentPrice}>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="lg:col-span-12">
          <Card
            title="Timeframe Selector"
            subtitle="Changing timeframe updates all panels."
            right={
              <div className="hidden text-xs text-zinc-500 sm:block">
                {TF_OPTIONS.map((t) => (
                  <span
                    key={t}
                    className={[
                      "mr-2 inline-flex rounded-lg px-2 py-1",
                      t === timeframe ? "bg-white/10 text-zinc-100" : "bg-black/10 text-zinc-400",
                    ].join(" ")}
                  >
                    {t}
                  </span>
                ))}
              </div>
            }
          >
            <TimeframeSelector value={timeframe} onChange={setTimeframe} />
          </Card>
        </div>

        {error ? (
          <div className="lg:col-span-12">
            <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 px-5 py-4 text-sm text-rose-200">
              {error}
            </div>
          </div>
        ) : null}

        <div className="lg:col-span-8">
          <Card
            title="Bitcoin Price"
            subtitle="Historical BTC price (line chart)."
            right={
              <div className="text-xs text-zinc-500">
                {loading ? "Loading…" : "Actual data"}
              </div>
            }
          >
         <PriceChart data={visibleData} />
          </Card>
        </div>

        <div className="lg:col-span-4">
          <Card
            title="Volatility Level"
            subtitle="Overall market volatility classification."
            right={
              <div className="text-xs text-zinc-500">
                Volatility-based
              </div>
            }
          >
            <RiskCard state={hmm?.current_state ?? "MEDIUM"} />
          </Card>
        </div>

        <div className="lg:col-span-5">
          <Card
            title="Volatility Regime (HMM)"
            subtitle="Current regime + state probabilities."
            right={<div className="text-xs text-zinc-500">HMM output</div>}
          >
            {hmm ? <HMMStatePanel data={hmm} /> : <div className="text-sm text-zinc-400">Loading…</div>}
          </Card>
        </div>

        <div className="lg:col-span-7">
          <Card
            title="Volatility & Returns"
            subtitle="Log returns (bars) and rolling volatility (line)."
            right={<div className="text-xs text-zinc-500">Clustering view</div>}
          >
            <VolatilityChart data={vol} />
          </Card>
        </div>

      

        <div className="lg:col-span-5">
          <Card
            title="Regime Transition Probabilities"
            subtitle="HMM transition matrix (from → to)."
            right={<div className="text-xs text-zinc-500">Markov</div>}
          >
            {hmm ? (
              <TransitionMatrixTable data={hmm} />
            ) : (
              <div className="text-sm text-zinc-400">Loading…</div>
            )}
          </Card>
        </div>
      </div>
    </DashboardLayout>
  );
}

