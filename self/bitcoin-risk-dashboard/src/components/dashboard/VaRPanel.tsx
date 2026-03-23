import type { VarResponse } from "@/utils/mockData";
import { formatPct } from "@/utils/format";
import { InfoTooltip } from "@/components/ui/InfoTooltip";

export function VaRPanel({ data }: { data: VarResponse }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between text-xs font-medium tracking-widest text-zinc-400">
        <span>VALUE-AT-RISK (VAR)</span>
        <InfoTooltip label="VaR interpretation">
          Value-at-Risk summarizes how much loss is expected not to be exceeded with a given
          confidence level over the next interval. For example, 95% VaR focuses on extreme
          downside moves and does not say anything about typical or average returns.
        </InfoTooltip>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded-xl border border-white/10 bg-black/20 p-4">
          <div className="text-xs font-medium tracking-widest text-zinc-400">
            VAR (95%)
          </div>
          <div className="mt-2 text-2xl font-semibold text-zinc-50">
            {formatPct(data.var95)}
          </div>
          <div className="mt-2 text-sm text-zinc-400">{data.interpretation95}</div>
        </div>

        <div className="rounded-xl border border-white/10 bg-black/20 p-4">
          <div className="text-xs font-medium tracking-widest text-zinc-400">
            VAR (99%)
          </div>
          <div className="mt-2 text-2xl font-semibold text-zinc-50">
            {formatPct(data.var99)}
          </div>
          <div className="mt-2 text-sm text-zinc-400">{data.interpretation99}</div>
        </div>
      </div>

      <div className="rounded-xl border border-white/10 bg-white/5 p-4 text-sm text-zinc-300">
        VaR summarizes downside risk for the next interval. It does{" "}
        <span className="font-semibold text-zinc-100">not predict price direction</span>{" "}
        or future returns.
      </div>
    </div>
  );
}

