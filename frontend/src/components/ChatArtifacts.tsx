import { BarChart3, ChevronDown, Download, Table2 } from "lucide-react";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { cn, currencySymbol, money, monthLabel } from "../lib/format";
import type { ChatArtifact, ChatChartArtifact, ChatTableArtifact } from "../lib/types";

const CHANNELS: Record<string, string> = {
  upi: "UPI",
  card: "Card",
  netbanking: "Net banking",
  neft: "NEFT",
  imps: "IMPS",
  rtgs: "RTGS",
  atm: "ATM",
  auto_debit: "Auto-debit",
  cheque: "Cheque",
  cash_deposit: "Cash deposit",
  bank: "Bank",
  wallet: "Wallet",
  other: "Other",
};
const HIDDEN = new Set(["id"]);
const COLLAPSED_ROWS = 8;

/** Exact amounts (keeps paise) — artifacts are the verified evidence, so no rounding. */
function exact(x: number) {
  const sign = x < 0 ? "−" : "";
  const abs = Math.abs(x).toLocaleString("en-IN", { minimumFractionDigits: 0, maximumFractionDigits: 2 });
  return `${sign}${currencySymbol()}${abs}`;
}

function heading(col: string) {
  return col.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

function cell(col: string, value: string | number | null, moneyCols: Set<string>) {
  if (value === null || value === undefined) return "—";
  if (moneyCols.has(col) && typeof value === "number") return exact(value);
  if (col === "channel" && typeof value === "string") return CHANNELS[value] ?? value;
  if (col === "month" && typeof value === "string" && /^\d{4}-\d{2}$/.test(value)) return monthLabel(value);
  return String(value);
}

function toCsv(a: ChatTableArtifact) {
  const esc = (v: unknown) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [a.columns.map(esc).join(","), ...a.rows.map((r) => r.map(esc).join(","))].join("\n");
}

export function TableArtifact({ artifact }: { artifact: ChatTableArtifact }) {
  const [open, setOpen] = useState(artifact.rows.length <= COLLAPSED_ROWS);
  const moneyCols = new Set(artifact.money_columns ?? []);
  const cols = artifact.columns.map((c, i) => [c, i] as const).filter(([c]) => !HIDDEN.has(c));
  const rows = open ? artifact.rows : artifact.rows.slice(0, COLLAPSED_ROWS);
  const directionIdx = artifact.columns.indexOf("direction");
  const download = () => {
    const blob = new Blob([toCsv(artifact)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${artifact.title.replace(/[^\w-]+/g, "_").slice(0, 60) || "finpilot"}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };
  return (
    <div className="mt-3 overflow-hidden rounded-xl border border-line">
      <div className="flex items-center justify-between gap-2 border-b border-line bg-surface-2/60 px-3 py-2 text-[12px]">
        <span className="flex min-w-0 items-center gap-1.5 font-medium">
          <Table2 className="size-3.5 shrink-0 text-brand" />
          <span className="truncate">{artifact.title}</span>
        </span>
        <span className="flex shrink-0 items-center gap-2 text-muted">
          {artifact.row_count}
          {artifact.total_rows > artifact.row_count ? ` of ${artifact.total_rows}` : ""} rows
          <button onClick={download} className="inline-flex items-center gap-1 hover:text-ink" title="Download CSV">
            <Download className="size-3.5" /> CSV
          </button>
        </span>
      </div>
      <div className={cn("overflow-auto", open && artifact.rows.length > 14 && "max-h-[420px]")}>
        <table className="w-full text-[12px]">
          <thead className="sticky top-0 bg-surface text-left text-muted">
            <tr>
              {cols.map(([c]) => (
                <th key={c} className={cn("whitespace-nowrap px-3 py-1.5 font-medium", moneyCols.has(c) && "text-right")}>
                  {heading(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-t border-line/70">
                {cols.map(([c, idx]) => {
                  const v = r[idx];
                  const credit = directionIdx >= 0 ? r[directionIdx] === "credit" : typeof v === "number" && c === "amount" && v > 0;
                  return (
                    <td
                      key={c}
                      title={typeof v === "string" && v.length > 40 ? v : undefined}
                      className={cn(
                        "px-3 py-1.5",
                        moneyCols.has(c) ? "num whitespace-nowrap text-right" : "max-w-[260px] truncate",
                        c === "amount" && credit && "text-emerald-600 dark:text-emerald-400",
                      )}
                    >
                      {cell(c, v, moneyCols)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {artifact.rows.length > COLLAPSED_ROWS && (
        <button
          onClick={() => setOpen(!open)}
          className="flex w-full items-center justify-center gap-1 border-t border-line py-1.5 text-[12px] text-muted hover:text-ink"
        >
          {open ? "Show fewer" : `Show all ${artifact.rows.length} rows`}
          <ChevronDown className={cn("size-3.5 transition", open && "rotate-180")} />
        </button>
      )}
      {artifact.truncated && (
        <div className="border-t border-line px-3 py-1.5 text-[11.5px] text-muted">
          Only the first {artifact.row_count} rows are included — narrow the question (a period or category) for the rest.
        </div>
      )}
    </div>
  );
}

function axisLabel(key: string, v: string) {
  if (key === "month" && /^\d{4}-\d{2}$/.test(v)) return monthLabel(v);
  if (key === "weekday") return v.slice(0, 3);
  return v;
}

/** One series, one hue: the title names it, the paired table is the exact view. */
export function ChartArtifact({ artifact }: { artifact: ChatChartArtifact }) {
  const data = artifact.data.map((d) => ({ ...d, label: axisLabel(artifact.x, String(d[artifact.x])) }));
  return (
    <div className="mt-3 rounded-xl border border-line p-3">
      <div className="mb-2 flex items-center gap-1.5 text-[12px] font-medium">
        <BarChart3 className="size-3.5 text-brand" /> {artifact.title}
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="label" tick={{ fontSize: 11, fill: "var(--muted)" }} tickLine={false} axisLine={false} />
          <YAxis
            tick={{ fontSize: 11, fill: "var(--muted)" }}
            tickLine={false}
            axisLine={false}
            width={52}
            tickFormatter={(v: number) => money(v, { compact: true })}
          />
          <Tooltip
            cursor={{ fill: "var(--surface-2)" }}
            content={({ active, payload }: any) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload;
              return (
                <div className="rounded-xl border border-line bg-surface px-3 py-2 text-xs shadow-xl">
                  <div className="mb-0.5 font-medium">{row.label}</div>
                  <div className="num">{exact(Number(row[artifact.y]))}</div>
                  {row.transactions !== undefined && <div className="text-muted">{row.transactions} transactions</div>}
                </div>
              );
            }}
          />
          <Bar dataKey={artifact.y} name={heading(artifact.y)} fill="var(--brand)" radius={[4, 4, 0, 0]} maxBarSize={34} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Artifacts({ artifacts }: { artifacts: ChatArtifact[] }) {
  if (!artifacts?.length) return null;
  // charts first (overview), then the exact tables
  const sorted = [...artifacts].sort((a, b) => (a.type === b.type ? 0 : a.type === "chart" ? -1 : 1));
  return (
    <div>
      {sorted.map((a, i) => (a.type === "chart" ? <ChartArtifact key={i} artifact={a} /> : <TableArtifact key={i} artifact={a} />))}
    </div>
  );
}
