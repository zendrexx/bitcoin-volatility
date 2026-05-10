import type { Timeframe } from "@/utils/data";

type Props = {
  value: Timeframe;
  onChange: (v: Timeframe) => void;
};

const options: { label: string; value: Timeframe }[] = [
  { label: "5 Minute", value: "5m" },
  { label: "15 Minute", value: "15m" },
  { label: "30 Minute", value: "30m" },
];

export function TimeframeSelector({ value, onChange }: Props) {
  return (
    <div className="flex w-full flex-wrap items-center gap-2">
      <div className="text-xs font-medium tracking-widest text-zinc-400">
        TIMEFRAME
      </div>
      <div className="ml-auto flex items-center gap-2">
        {options.map((o) => {
          const active = o.value === value;
          return (
            <button
              key={o.value}
              onClick={() => onChange(o.value)}
              className={[
                "h-9 rounded-xl px-3 text-sm transition",
                active
                  ? "bg-white/10 text-zinc-50 ring-1 ring-white/10"
                  : "bg-white/5 text-zinc-300 hover:bg-white/10",
              ].join(" ")}
            >
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

