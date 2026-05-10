// const AUTH_STORAGE_KEY = "bvrdauth:v1";

// export type Session = {
//   email: string;
//   createdAt: number;
// };

// export function getSession(): Session | null {
//   if (typeof window === "undefined") return null;
//   try {
//     const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
//     if (!raw) return null;
//     const parsed = JSON.parse(raw) as Session;
//     if (!parsed?.email) return null;
//     return parsed;
//   } catch {
//     return null;
//   }
// }

// export function isAuthenticated(): boolean {
//   return !!getSession();
// }

// export function login(email: string): Session {
//   const session: Session = { email, createdAt: Date.now() };
//   window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(session));
//   return session;
// }

// export function logout() {
//   if (typeof window === "undefined") return;
//   window.localStorage.removeItem(AUTH_STORAGE_KEY);
// }

