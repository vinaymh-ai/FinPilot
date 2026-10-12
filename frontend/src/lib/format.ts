import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const SYMBOLS: Record<string, string> = { INR: "₹", USD: "$", EUR: "€", GBP: "£" };
let currency = "INR";
export function setCurrency(c: string) {
  currency = c || "INR";
}
export function currencySymbol() {
  return SYMBOLS[currency] ?? "";
}

const locale = () => (currency === "INR" ? "en-IN" : "en-US");

export function money(x: number | null | undefined, opts: { sign?: boolean; compact?: boolean } = {}) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  const sym = currencySymbol();
  const abs = Math.abs(x);
  let body: string;
  if (opts.compact && abs >= 1000) {
    if (currency === "INR" && abs >= 100000) body = `${(abs / 100000).toFixed(abs >= 1000000 ? 1 : 2)}L`;
    else body = `${(abs / 1000).toFixed(abs >= 10000 ? 0 : 1)}k`;
  } else {
    body = abs.toLocaleString(locale(), { maximumFractionDigits: 0 });
  }
  const sign = x < 0 ? "−" : opts.sign && x > 0 ? "+" : "";
  return `${sign}${sym}${body}`;
}

export function pct(x: number | null | undefined, digits = 0, sign = false) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  const s = sign && x > 0 ? "+" : "";
  return `${s}${x.toFixed(digits)}%`;
}

export function prob(x: number | null | undefined) {
  if (x === null || x === undefined) return "–";
  return `${Math.round(x * 100)}%`;
}

export function date(d: string | null | undefined, withYear = false) {
  if (!d) return "–";
  const dt = new Date(d.length === 10 ? `${d}T00:00:00` : d);
  return dt.toLocaleDateString(locale(), {
    day: "numeric",
    month: "short",
    ...(withYear ? { year: "numeric" } : {}),
  });
}

export function dateTime(d: string | null | undefined) {
  if (!d) return "–";
  const dt = new Date(d.endsWith("Z") || d.includes("+") ? d : `${d}Z`);
  return dt.toLocaleString(locale(), { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

export function monthLabel(m: string) {
  const [y, mo] = m.split("-").map(Number);
  return new Date(y, mo - 1, 1).toLocaleDateString(locale(), { month: "short", year: "2-digit" });
}

export function titleCase(s: string) {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function isoDate(d: Date) {
  return d.toISOString().slice(0, 10);
}

export function addDays(iso: string, n: number) {
  const d = new Date(`${iso}T00:00:00`);
  d.setDate(d.getDate() + n);
  return isoDate(d);
}
