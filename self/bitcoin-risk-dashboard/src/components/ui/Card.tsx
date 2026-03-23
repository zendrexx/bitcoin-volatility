type Props = {
  title?: string;
  subtitle?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
};

export function Card({ title, subtitle, right, children, className }: Props) {
  return (
    <section
      className={[
        "rounded-2xl border border-white/10 bg-white/5 shadow-[0_0_0_1px_rgba(255,255,255,0.04),0_20px_50px_-30px_rgba(0,0,0,0.8)] backdrop-blur",
        className ?? "",
      ].join(" ")}
    >
      {title ? (
        <div className="flex items-start gap-3 border-b border-white/10 px-5 py-4">
          <div className="min-w-0">
            <div className="text-sm font-semibold text-zinc-50">{title}</div>
            {subtitle ? (
              <div className="mt-1 text-xs text-zinc-400">{subtitle}</div>
            ) : null}
          </div>
          {right ? <div className="ml-auto">{right}</div> : null}
        </div>
      ) : null}
      <div className="px-5 py-4">{children}</div>
    </section>
  );
}

