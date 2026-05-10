// import Head from "next/head";
// import { useRouter } from "next/router";
// import { useEffect, useState } from "react";
// import { LoginForm } from "@/components/LoginForm";
// import { isAuthenticated, login } from "@/auth/session";

// export default function LoginPage() {
//   const router = useRouter();
//   const [isLoading, setIsLoading] = useState(false);
//   const [error, setError] = useState<string | null>(null);

//   useEffect(() => {
//     if (isAuthenticated()) router.replace("/dashboard");
//   }, [router]);

//   return (
//     <>
//       <Head>
//         <title>Login | Bitcoin Volatility Risk Dashboard</title>
//       </Head>

//       <div className="min-h-screen bg-[#070A10] text-zinc-100">
//         <div className="pointer-events-none absolute inset-0 overflow-hidden">
//           <div className="absolute -top-40 left-1/2 h-[520px] w-[520px] -translate-x-1/2 rounded-full bg-cyan-500/15 blur-3xl" />
//           <div className="absolute -bottom-52 right-[-120px] h-[560px] w-[560px] rounded-full bg-blue-500/10 blur-3xl" />
//           <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_15%,rgba(255,255,255,0.08),transparent_38%),radial-gradient(circle_at_80%_40%,rgba(34,211,238,0.06),transparent_45%)]" />
//         </div>

//         <div className="relative mx-auto flex min-h-screen max-w-6xl items-center justify-center px-6 py-14">
//           <div className="w-full">
//             <div className="mb-10 text-center">
//               <div className="mx-auto inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs text-zinc-300 backdrop-blur">
//                 <span className="h-2 w-2 rounded-full bg-cyan-400" />
//                 Thesis Dashboard
//               </div>
//               <h1 className="mt-4 text-3xl font-semibold tracking-tight text-zinc-50 sm:text-4xl">
//                 Bitcoin Volatility Risk Dashboard
//               </h1>
//               <p className="mt-2 text-sm text-zinc-400">
//                 Hidden Markov Model (HMM) regimes
//               </p>
//             </div>

//             <div className="flex justify-center">
//               <LoginForm
//                 isLoading={isLoading}
//                 error={error}
//                 onSubmit={async (email, password) => {
//                   setError(null);
//                   setIsLoading(true);
//                   try {
//                     if (!email || !password) {
//                       setError("Please enter both email and password.");
//                       return;
//                     }
//                     login(email);
//                     await router.push("/dashboard");
//                   } finally {
//                     setIsLoading(false);
//                   }
//                 }}
//               />
//             </div>
//           </div>
//         </div>
//       </div>
//     </>
//   );
// }

