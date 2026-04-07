import type { HmmState } from "@/utils/mockData";

function riskFromState(state: HmmState) {
  if (state === "LOW") return "LOW VOLATILITY" as const;
  if (state === "MEDIUM") return "MEDIUM VOLATILITY" as const;
  return "HIGH VOLATILITY" as const;
}

function color(state: HmmState) {
  if (state === "LOW") return { ring: "ring-emerald-400/25", bg: "bg-emerald-400/10", dot: "bg-emerald-400" };
  if (state === "MEDIUM") return { ring: "ring-amber-400/25", bg: "bg-amber-400/10", dot: "bg-amber-400" };
  return { ring: "ring-rose-400/25", bg: "bg-rose-400/10", dot: "bg-rose-400" };
}

function message(state: HmmState) {
  if (state === "LOW") {
    return "Current market conditions indicate stable volatility. Short-term price movements are relatively orderly.";
  }
  if (state === "MEDIUM") {
    return "Volatility is rising. Short-term moves may be choppy; risk controls and position sizing matter more.";
  }
  return "Current market conditions indicate elevated volatility. Short-term price movements may be unstable.";
}

export function RiskCard({ state }: { state: HmmState }) {
  const level = riskFromState(state);
  const c = color(state);
  return (
    <div className={["rounded-2xl p-5 ring-1", c.ring, c.bg].join(" ")}>
      <div className="flex items-start gap-3">
        <div className={["mt-1 h-3 w-3 rounded-full", c.dot].join(" ")} />
        <div className="min-w-0">
          <div className="text-xs font-medium tracking-widest text-zinc-400">
            OVERALL CLASSIFICATION
          </div>
          <div className="mt-1 text-2xl font-semibold text-zinc-50">{level}</div>
          <div className="mt-2 text-sm text-zinc-300">{message(state)}</div>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-white/10 bg-black/20 p-4 text-xs text-zinc-400">
        This dashboard classifies volatility based on regimes and VaR. It does not forecast future prices.
      </div>
    </div>
  );
}

