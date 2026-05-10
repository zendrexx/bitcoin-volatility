export function formatUSD(n?: number | null) {
  if (typeof n !== "number" || Number.isNaN(n)) {
    return "—";
  }

  return n.toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });
}
export function formatPct(n: number, digits = 2) {
  return `${(n * 100).toFixed(digits)}%`;
}

export function formatTime(iso: string) {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

