import type { HmmStateResponse } from "@/utils/mockData";
import { InfoTooltip } from "@/components/ui/InfoTooltip";

function colorForState(state: "LOW" | "MEDIUM" | "HIGH") {
  if (state === "LOW") return "bg-emerald-400";
  if (state === "MEDIUM") return "bg-amber-400";
  return "bg-rose-400";
}

function labelForState(state: "LOW" | "MEDIUM" | "HIGH") {
  if (state === "LOW") return "LOW VOLATILITY";
  if (state === "MEDIUM") return "MEDIUM VOLATILITY";
  return "HIGH VOLATILITY";
}

function ProgressRow({
  label,
  value,
  barClass,
}: {
  label: string;
  value: number;
  barClass: string;
}) {
  const pct = Math.round(value * 100);
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs">
        <div className="text-zinc-300">{label}</div>
        <div className="font-semibold text-zinc-50">{pct}%</div>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-white/10">
        <div className={["h-full rounded-full", barClass].join(" ")} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function HMMStatePanel({ data }: { data: HmmStateResponse }) {
  const state = data.current_state;
  return (
    <div className="space-y-5">
      <div className="flex items-start gap-3">
        <div className={["mt-1 h-3 w-3 rounded-full", colorForState(state)].join(" ")} />
        <div className="min-w-0">
          <div className="flex items-center gap-1 text-xs font-medium tracking-widest text-zinc-400">
            <span>CURRENT STATE</span>
            <InfoTooltip label="Hidden Markov Model (HMM)">
              The HMM tracks volatility regimes that are not directly observable. Each regime
              corresponds to a different volatility level, and the model infers which regime is
              most likely given recent return behaviour.
            </InfoTooltip>
          </div>
          <div className="mt-1 text-lg font-semibold text-zinc-50">
            {labelForState(state)}
          </div>
          <div className="mt-1 text-sm text-zinc-400">
            Regime classification derived from volatility dynamics (HMM).
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex items-center gap-1 text-xs font-medium tracking-widest text-zinc-400">
          <span>STATE PROBABILITIES</span>
          <InfoTooltip label="Regime probabilities">
            These probabilities show how likely it is that the market is currently in each volatility
            regime (low, medium, high). Values sum to 1 and are updated as new returns are observed.
          </InfoTooltip>
        </div>
        <ProgressRow
          label="Low"
          value={data.probabilities.low}
          barClass="bg-emerald-400/90"
        />
        <ProgressRow
          label="Medium"
          value={data.probabilities.medium}
          barClass="bg-amber-400/90"
        />
        <ProgressRow
          label="High"
          value={data.probabilities.high}
          barClass="bg-rose-400/90"
        />
      </div>
    </div>
  );
}

