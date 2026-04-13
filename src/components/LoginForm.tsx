import { useState } from "react";

type Props = {
  onSubmit: (email: string, password: string) => Promise<void> | void;
  isLoading?: boolean;
  error?: string | null;
};

export function LoginForm({ onSubmit, isLoading, error }: Props) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  return (
    <div className="w-full max-w-md">
      <div className="rounded-2xl border border-white/10 bg-white/5 p-6 shadow-[0_0_0_1px_rgba(255,255,255,0.04),0_25px_60px_-30px_rgba(0,0,0,0.8)] backdrop-blur">
        <div className="mb-6">
          <div className="text-xs font-medium tracking-widest text-zinc-400">
            BITCOIN VOLATILITY DASHBOARD
          </div>
          <div className="mt-2 text-2xl font-semibold text-zinc-50">
            Sign in
          </div>
          <div className="mt-1 text-sm text-zinc-400">
            Market volatility classification, not price prediction.
          </div>
        </div>

        <form
          className="space-y-4"
          onSubmit={async (e) => {
            e.preventDefault();
            await onSubmit(email.trim(), password);
          }}
        >
          <div className="space-y-2">
            <label className="text-sm font-medium text-zinc-200">Email</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="h-11 w-full rounded-xl border border-white/10 bg-black/30 px-4 text-zinc-100 outline-none ring-0 placeholder:text-zinc-500 focus:border-cyan-400/40 focus:shadow-[0_0_0_3px_rgba(34,211,238,0.08)]"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-zinc-200">
              Password
            </label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="h-11 w-full rounded-xl border border-white/10 bg-black/30 px-4 text-zinc-100 outline-none ring-0 placeholder:text-zinc-500 focus:border-cyan-400/40 focus:shadow-[0_0_0_3px_rgba(34,211,238,0.08)]"
            />
          </div>

          {error ? (
            <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              {error}
            </div>
          ) : null}

          <button
            type="submit"
            disabled={!!isLoading}
            className="group relative h-11 w-full overflow-hidden rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 text-sm font-semibold text-white shadow-[0_16px_40px_-22px_rgba(34,211,238,0.6)] transition disabled:cursor-not-allowed disabled:opacity-60"
          >
            <span className="relative z-10">
              {isLoading ? "Signing in…" : "Login"}
            </span>
            <span className="absolute inset-0 opacity-0 transition group-hover:opacity-100 bg-[radial-gradient(circle_at_30%_30%,rgba(255,255,255,0.35),transparent_60%)]" />
          </button>
        </form>
      </div>

      <div className="mt-4 text-center text-xs text-zinc-500">
        Demo login accepts any email/password.
      </div>
    </div>
  );
}

