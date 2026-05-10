import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Bar,
} from "recharts";
import type { VolPoint } from "@/utils/data";
import { formatTime } from "@/utils/format";

function TooltipContent({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { name?: string; value?: number }[];
  label?: string;
}) {
  if (!active || !payload?.length || !label) return null;
  const lr = payload.find((p) => p.name === "log_return")?.value;
  const rv = payload.find((p) => p.name === "rolling_volatility")?.value;
  return (
    <div className="rounded-xl border border-white/10 bg-black/70 px-3 py-2 text-xs text-zinc-200 backdrop-blur">
      <div className="text-zinc-400">{formatTime(label)}</div>
      <div className="mt-1 grid gap-1">
        <div className="flex items-center justify-between gap-6">
          <span className="text-zinc-300">Log return</span>
          <span className="font-semibold text-zinc-50">
            {typeof lr === "number" ? lr.toFixed(5) : "—"}
          </span>
        </div>
        <div className="flex items-center justify-between gap-6">
          <span className="text-zinc-300">Rolling vol</span>
          <span className="font-semibold text-zinc-50">
            {typeof rv === "number" ? rv.toFixed(5) : "—"}
          </span>
        </div>
      </div>
    </div>
  );
}

export function VolatilityChart({ data }: { data: VolPoint[] }) {
  return (
    <div className="h-[320px] w-full min-w-0">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
          <XAxis
            dataKey="timestamp"
            tickFormatter={(v) => formatTime(String(v))}
            tick={{ fill: "rgba(228,228,231,0.7)", fontSize: 11 }}
            axisLine={{ stroke: "rgba(255,255,255,0.10)" }}
            tickLine={false}
            minTickGap={24}
          />
          <YAxis
            yAxisId="left"
            width={62}
            tick={{ fill: "rgba(228,228,231,0.7)", fontSize: 11 }}
            axisLine={{ stroke: "rgba(255,255,255,0.10)" }}
            tickLine={false}
            tickFormatter={(v) => Number(v).toFixed(3)}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            width={58}
            tick={{ fill: "rgba(228,228,231,0.7)", fontSize: 11 }}
            axisLine={{ stroke: "rgba(255,255,255,0.10)" }}
            tickLine={false}
            tickFormatter={(v) => Number(v).toFixed(3)}
          />
          <Tooltip
            content={<TooltipContent />}
            cursor={{ stroke: "rgba(34,211,238,0.25)" }}
          />

          <Bar
            yAxisId="left"
            dataKey="log_return"
            name="log_return"
            fill="rgba(148,163,184,0.35)"
            stroke="rgba(148,163,184,0.2)"
            barSize={10}
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="rolling_volatility"
            name="rolling_volatility"
            stroke="rgba(99,102,241,0.95)"
            strokeWidth={2}
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

