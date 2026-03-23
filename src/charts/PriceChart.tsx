import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PricePoint } from "@/utils/mockData";
import { formatTime, formatUSD } from "@/utils/format";

function TooltipContent({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { value?: number }[];
  label?: string;
}) {
  if (!active || !payload?.length || !label) return null;
  const price = payload[0]?.value;
  return (
    <div className="rounded-xl border border-white/10 bg-black/70 px-3 py-2 text-xs text-zinc-200 backdrop-blur">
      <div className="text-zinc-400">{formatTime(label)}</div>
      <div className="mt-1 text-sm font-semibold text-zinc-50">
        {typeof price === "number" ? formatUSD(price) : "—"}
      </div>
    </div>
  );
}

export function PriceChart({ data }: { data: PricePoint[] }) {
  return (
    <div className="h-[320px] w-full min-w-0">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
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
            width={62}
            tick={{ fill: "rgba(228,228,231,0.7)", fontSize: 11 }}
            axisLine={{ stroke: "rgba(255,255,255,0.10)" }}
            tickLine={false}
            tickFormatter={(v) => `$${Number(v).toFixed(0)}`}
          />
          <Tooltip
            content={<TooltipContent />}
            cursor={{ stroke: "rgba(34,211,238,0.25)" }}
          />
          <Line
            type="monotone"
            dataKey="price"
            stroke="rgba(34,211,238,0.95)"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3, fill: "rgba(34,211,238,1)" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

