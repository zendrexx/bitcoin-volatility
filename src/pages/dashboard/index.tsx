import { useEffect, useMemo, useState } from "react";
import { DashboardLayout } from "@/components/dashboard/DashboardLayout";
import { Card } from "@/components/ui/Card";
import { TimeframeSelector } from "@/components/dashboard/TimeframeSelector";
import { PriceChart } from "@/charts/PriceChart";
import { VolatilityChart } from "@/charts/VolatilityChart";
import { HMMStatePanel } from "@/components/dashboard/HMMStatePanel";
import { VaRPanel } from "@/components/dashboard/VaRPanel";
import { TransitionMatrixTable } from "@/components/dashboard/TransitionMatrixTable";
import { RiskCard } from "@/components/dashboard/RiskCard";
import { apiGet } from "@/utils/api";
import type {
  HmmStateResponse,
  PricePoint,
  Timeframe,
  VarResponse,
  VolPoint,
} from "@/utils/mockData";

const TF_OPTIONS: Timeframe[] = ["5m", "10m", "15m"];

export default function DashboardPage() {
  const [timeframe, setTimeframe] = useState<Timeframe>("5m");
  const [prices, setPrices] = useState<PricePoint[]>([]);
  const [vol, setVol] = useState<VolPoint[]>([]);
  const [hmm, setHmm] = useState<HmmStateResponse | null>(null);
  const [varData, setVarData] = useState<VarResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const currentPrice = useMemo(() => prices.at(-1)?.price ?? null, [prices]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const params = { timeframe };
        const [p, v, h, r] = await Promise.all([
          apiGet<PricePoint[]>("/api/price-history", params),
          apiGet<VolPoint[]>("/api/volatility", params),
          apiGet<HmmStateResponse>("/api/hmm-state", params),
          apiGet<VarResponse>("/api/var", params),
        ]);
        if (cancelled) return;
        setPrices(p);
        setVol(v);
        setHmm(h);
        setVarData(r);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Failed to load data.");
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
                {loading ? "Loading…" : "Mock data"}
              </div>
            }
          >
            <PriceChart data={prices} />
          </Card>
        </div>

        <div className="lg:col-span-4">
          <Card
            title="Risk Level"
            subtitle="Overall market risk classification."
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

        <div className="lg:col-span-7">
          <Card
            title="Value-at-Risk (VaR)"
            subtitle="Downside risk measurement at 95% and 99% confidence."
            right={<div className="text-xs text-zinc-500">No price prediction</div>}
          >
            {varData ? <VaRPanel data={varData} /> : <div className="text-sm text-zinc-400">Loading…</div>}
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

