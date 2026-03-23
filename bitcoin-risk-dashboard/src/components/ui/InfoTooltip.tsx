type Props = {
  label?: string;
  children: React.ReactNode;
};

export function InfoTooltip({ label, children }: Props) {
  return (
    <span className="group relative inline-flex items-center">
      <span className="ml-1 inline-flex h-4 w-4 items-center justify-center rounded-full border border-white/30 bg-white/10 text-[10px] font-semibold text-zinc-100 shadow-sm">
        i
      </span>
      <span className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 w-72 -translate-x-1/2 rounded-xl border border-white/15 bg-black/90 px-3 py-3 text-xs text-zinc-100 opacity-0 shadow-xl ring-1 ring-black/60 backdrop-blur transition group-hover:opacity-100">
        {label ? (
          <span className="mb-1 block text-[10px] font-semibold uppercase tracking-widest text-zinc-400">
            {label}
          </span>
        ) : null}
        {children}
      </span>
    </span>
  );
}

