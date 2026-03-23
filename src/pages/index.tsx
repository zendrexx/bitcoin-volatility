import { useRouter } from "next/router";
import { useEffect } from "react";
import { isAuthenticated } from "@/auth/session";

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    router.replace(isAuthenticated() ? "/dashboard" : "/login");
  }, [router]);

  return (
    <div className="min-h-screen bg-[#070A10] text-zinc-100">
      <div className="mx-auto flex min-h-screen max-w-4xl items-center justify-center px-6">
        <div className="text-sm text-zinc-400">Redirecting…</div>
      </div>
    </div>
  );
}
