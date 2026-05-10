import type { HmmStateResponse } from "@/utils/data";
import { InfoTooltip } from "@/components/ui/InfoTooltip";

function fmt(n: number) {
  return n.toFixed(2);
}

export function TransitionMatrixTable({ data }: { data: HmmStateResponse }) {
  const m = data.transition_matrix;
  const labels = ["Low", "Medium", "High"];
  return (
    <div className="overflow-hidden rounded-xl border border-white/10">
      <table className="w-full border-collapse text-sm">
        <thead className="bg-white/5">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-medium tracking-widest text-zinc-400">
              <div className="flex items-center gap-1">
                <span>FROM \ TO</span>
                <InfoTooltip label="Regime transitions">
                  Each cell gives the probability of moving from one volatility regime to another in
                  the next step. Large diagonal values mean regimes are persistent, while
                  off-diagonal values capture how often the market switches between volatility
                  states.
                </InfoTooltip>
              </div>
            </th>
            {labels.map((l) => (
              <th
                key={l}
                className="px-4 py-3 text-left text-xs font-medium tracking-widest text-zinc-400"
              >
                {l.toUpperCase()}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {m.map((row, i) => (
            <tr key={labels[i]} className="border-t border-white/10">
              <td className="px-4 py-3 font-semibold text-zinc-200">
                {labels[i]}
              </td>
              {row.map((v, j) => {
                const strong = i === j;
                return (
                  <td key={j} className="px-4 py-3 text-zinc-200">
                    <span
                      className={[
                        "inline-flex rounded-lg px-2 py-1",
                        strong ? "bg-white/10 text-zinc-50" : "bg-black/10",
                      ].join(" ")}
                    >
                      {fmt(v)}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

