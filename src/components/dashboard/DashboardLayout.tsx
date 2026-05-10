import Head from "next/head";
import { useEffect, useMemo, useState } from "react";

import { formatUSD } from "@/utils/format";

type Props = {
  title?: string;
  currentPrice?: number | null;
  children: React.ReactNode;
};

export function DashboardLayout({ title, currentPrice, children }: Props) {

  const [now, setNow] = useState<Date | null>(null);

 

  useEffect(() => {
  setNow(new Date());

  const id = window.setInterval(() => {
    setNow(new Date());
  }, 1000);

  return () => window.clearInterval(id);
}, []);



  return (
    <>
      <Head>
        <title>{title ?? "Dashboard"} | Bitcoin Market Risk Monitor</title>
      </Head>

      <div className="min-h-screen bg-[#070A10] text-zinc-100">
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 left-1/2 h-[520px] w-[520px] -translate-x-1/2 rounded-full bg-cyan-500/12 blur-3xl" />
          <div className="absolute -bottom-52 right-[-120px] h-[560px] w-[560px] rounded-full bg-blue-500/10 blur-3xl" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_10%,rgba(255,255,255,0.07),transparent_35%),radial-gradient(circle_at_70%_30%,rgba(34,211,238,0.05),transparent_40%)]" />
        </div>

        <header className="relative border-b border-white/10 bg-black/10 backdrop-blur">
          <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-4 sm:px-6">
            <div className="flex min-w-0 items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-cyan-400/20 to-blue-500/10 ring-1 ring-white/10">
                <div className="h-3 w-3 rounded-full bg-cyan-400 shadow-[0_0_0_6px_rgba(34,211,238,0.12)]" />
              </div>
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold text-zinc-50">
                  Bitcoin Market Volatility Monitor
                </div>
                <div className="truncate text-xs text-zinc-400">
                  Volatility regime classification (HMM)
                </div>
              </div>
            </div>

            <div className="ml-auto flex items-center gap-3">
              <div className="hidden items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-3 py-2 sm:flex">
                <div className="text-xs text-zinc-400">BTC</div>
                <div className="text-sm font-semibold text-zinc-50">
                  {typeof currentPrice === "number" ? formatUSD(currentPrice) : "—"}
                </div>
              </div>

              <div className="hidden rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-zinc-200 sm:block">
                  {now
                    ? now.toLocaleTimeString(undefined, {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      })
                    : "--:--:--"}
                </div>

             
            </div>
          </div>

 
        </header>

        <main className="relative mx-auto max-w-7xl px-4 py-6 sm:px-6">
          {children}
        </main>
      </div>
    </>
  );
}

