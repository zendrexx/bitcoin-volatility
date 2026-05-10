import {
  ResponsiveContainer,
  ComposedChart,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Bar,
  ReferenceLine,
} from "recharts";
import { formatTime, formatUSD } from "@/utils/format";

type Candle = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  state: number | null;
  log_return?: number | null;
};

const STATE_COLOR: Record<number, string> = {
  0: "#22c55e",
  1: "#f59e0b",
  2: "#ef4444",
};
const STATE_LABEL: Record<number, string> = {
  0: "Low",
  1: "Medium",
  2: "High",
};

function stateColor(state: number | null) {
  if (state === null) return "#71717a";
  return STATE_COLOR[state] ?? "#71717a";
}

// ── Candle shape ────────────────────────────────────────────────────────────
function CandleBar(props: any) {
  const { x, y, width, height, payload } = props;
  if (!payload || width <= 0) return null;

  const { open, high, low, close, state } = payload;
  const color = stateColor(state);
  const range = high - low;
  if (range === 0) return null;

  const pxPerUnit  = height / range;
  const yHigh      = y;
  const yLow       = y + height;
  const yOpen      = y + (high - open)  * pxPerUnit;
  const yClose     = y + (high - close) * pxPerUnit;
  const bodyTop    = Math.min(yOpen, yClose);
  const bodyHeight = Math.max(Math.abs(yOpen - yClose), 1);
  const cx         = x + width / 2;
  const halfBody   = Math.max(width * 0.4, 2);

  return (
    <g>
      <line x1={cx} x2={cx} y1={yHigh} y2={yLow} stroke={color} strokeWidth={1} />
      <rect
        x={cx - halfBody} y={bodyTop}
        width={halfBody * 2} height={bodyHeight}
        fill={color} opacity={0.9}
      />
    </g>
  );
}

// ── Log-return bar shape (green above zero, red below) ──────────────────────
function ReturnBar(props: any) {
  const { x, y, width, height, payload } = props;
  if (!payload || width <= 0) return null;
  const lr = payload.log_return ?? 0;
  const color = lr >= 0 ? "#22c55e" : "#ef4444";
  return (
    <rect
      x={x} y={y}
      width={Math.max(width - 1, 1)} height={Math.max(Math.abs(height), 1)}
      fill={color} opacity={0.75}
    />
  );
}

// ── Shared tooltip ──────────────────────────────────────────────────────────
function CandleTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const color = stateColor(d.state);
  const lr  = d.log_return  != null ? (d.log_return  * 100).toFixed(4) : "—";
  const vol = d.volatility  != null ? (d.volatility  * 100).toFixed(4) : "—";

  return (
    <div className="rounded-xl border border-white/10 bg-black/80 px-3 py-2 text-xs text-zinc-200 space-y-0.5 min-w-[140px]">
      <div className="text-zinc-400 mb-1">{formatTime(d.timestamp)}</div>
      <div>O: {formatUSD(d.open)}</div>
      <div>H: {formatUSD(d.high)}</div>
      <div>L: {formatUSD(d.low)}</div>
      <div>C: {formatUSD(d.close)}</div>
      <div className="border-t border-white/10 mt-1 pt-1">
        <div>Log Ret: <span className={d.log_return >= 0 ? "text-green-400" : "text-red-400"}>{lr}%</span></div>
       
      </div>
      {d.state !== null && (
        <div className="font-semibold mt-1" style={{ color }}>
          {STATE_LABEL[d.state] ?? `State ${d.state}`} Volatility
        </div>
      )}
    </div>
  );
}

function Legend() {
  return (
    <div className="mb-2 flex gap-4 text-xs text-zinc-400">
      {Object.entries(STATE_LABEL).map(([k, label]) => (
        <span key={k} className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: STATE_COLOR[Number(k)] }} />
          {label} Vol
        </span>
      ))}
      <span className="flex items-center gap-1.5 ml-auto">
        <span className="inline-block h-2.5 w-2.5 rounded-sm bg-green-500 opacity-75" />
        +Return
      </span>
      <span className="flex items-center gap-1.5">
        <span className="inline-block h-2.5 w-2.5 rounded-sm bg-red-500 opacity-75" />
        −Return
      </span>
    </div>
  );
}

export function PriceChart({ data }: { data: Candle[] }) {
  if (!data?.length) return null;

  const chartData = data.map((d) => ({
    ...d,
    range: [d.low, d.high] as [number, number],
    // bar dataKey needs a positive number; we use raw value, ReturnBar handles sign
    log_return_bar: d.log_return ?? 0,
  }));

  const domainMin = Math.min(...data.map((d) => d.low));
  const domainMax = Math.max(...data.map((d) => d.high));
  const pad = (domainMax - domainMin) * 0.04;

  const maxLR  = Math.max(...data.map((d) => Math.abs(d.log_return ?? 0)));

  return (
    <div className="w-full">
      <Legend />

      {/* ── Price + candles ───────────────────────────────────────────── */}
      <div className="h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} syncId="btc">
            <CartesianGrid stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="timestamp" tickFormatter={formatTime} tick={{ fill: "#aaa", fontSize: 11 }} minTickGap={40} hide />
            <YAxis
              domain={[domainMin - pad, domainMax + pad]}
              tick={{ fill: "#aaa", fontSize: 11 }}
              tickFormatter={(v) => `$${Number(v).toFixed(0)}`}
              width={80}
            />
            <Tooltip content={<CandleTooltip />} cursor={{ stroke: "rgba(255,255,255,0.1)" }} />
            <Bar dataKey="range" shape={<CandleBar />} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ── Log return bars ───────────────────────────────────────────── */}
      <div className="h-[100px] mt-1">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }} syncId="btc">
            <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatTime}
              tick={{ fill: "#aaa", fontSize: 10 }}
              minTickGap={40}
            />
            <YAxis
              domain={[-maxLR * 1.2, maxLR * 1.2]}
              tick={{ fill: "#aaa", fontSize: 10 }}
              tickFormatter={(v) => `${(v * 100).toFixed(2)}%`}
              width={80}
            />
            <Tooltip content={<CandleTooltip />} cursor={{ stroke: "rgba(255,255,255,0.1)" }} />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
            <Bar dataKey="log_return_bar" shape={<ReturnBar />} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}