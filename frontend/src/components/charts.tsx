import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { date, money, monthLabel } from "../lib/format";
import type { ForecastSummary } from "../lib/types";

const AXIS = { fontSize: 11, fill: "var(--muted)" };

function ChartTooltip({ active, payload, label, labelFmt }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-line bg-surface px-3 py-2 text-xs shadow-xl">
      <div className="mb-1 font-medium">{labelFmt ? labelFmt(label) : label}</div>
      {payload
        .filter((p: any) => p.value !== undefined && p.dataKey !== "band" && p.dataKey !== "sband")
        .map((p: any) => (
          <div key={p.dataKey} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-muted">
              <span className="size-2 rounded-full" style={{ background: p.color || p.stroke || p.fill }} />
              {p.name}
            </span>
            <span className="num font-medium">{money(p.value)}</span>
          </div>
        ))}
    </div>
  );
}

/** Fan chart: P10–P90 band with the P50 path, safety buffer and goal deadlines. */
export function ForecastChart({
  forecast,
  scenario,
  height = 300,
  showGoals = true,
}: {
  forecast: ForecastSummary;
  scenario?: ForecastSummary;
  height?: number;
  showGoals?: boolean;
}) {
  const s = forecast.series;
  if (!s) return null;
  const sc = scenario?.series;
  const rows = s.dates.map((d, i) => ({
    date: d,
    p50: s.p50[i],
    band: [s.p10[i], s.p90[i]] as [number, number],
    ...(sc ? { s50: sc.p50[i], sband: [sc.p10[i], sc.p90[i]] as [number, number] } : {}),
  }));
  const goalLines = showGoals ? forecast.goals.filter((g) => g.deadline <= s.dates[s.dates.length - 1]) : [];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="bandFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#34d399" stopOpacity={0.28} />
            <stop offset="100%" stopColor="#34d399" stopOpacity={0.06} />
          </linearGradient>
          <linearGradient id="sbandFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#818cf8" stopOpacity={0.3} />
            <stop offset="100%" stopColor="#818cf8" stopOpacity={0.06} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="date" tick={AXIS} tickFormatter={(d) => date(d)} minTickGap={40} axisLine={false} tickLine={false} />
        <YAxis tick={AXIS} tickFormatter={(v) => money(v, { compact: true })} width={64} axisLine={false} tickLine={false} />
        <Tooltip content={<ChartTooltip labelFmt={(d: string) => date(d, true)} />} />
        <Area dataKey="band" name="P10–P90" stroke="none" fill="url(#bandFill)" isAnimationActive={false} />
        {sc && <Area dataKey="sband" name="Scenario P10–P90" stroke="none" fill="url(#sbandFill)" isAnimationActive={false} />}
        <Line dataKey="p50" name={sc ? "Baseline (P50)" : "Expected (P50)"} stroke="#10b981" strokeWidth={2.2} dot={false} isAnimationActive={false} />
        {sc && <Line dataKey="s50" name="With decision (P50)" stroke="#818cf8" strokeWidth={2.2} dot={false} isAnimationActive={false} />}
        {forecast.projection_type !== "cashflow_change" && <ReferenceLine
          y={forecast.safety_buffer}
          stroke="#f59e0b"
          strokeDasharray="5 5"
          label={{ value: "safety buffer", position: "insideBottomRight", fontSize: 10, fill: "#f59e0b" }}
        />}
        <ReferenceLine y={0} stroke="#ef4444" strokeOpacity={0.5} />
        {goalLines.map((g) => (
          <ReferenceLine
            key={g.name}
            x={s.dates.reduce((best, d) => (Math.abs(+new Date(d) - +new Date(g.deadline)) < Math.abs(+new Date(best) - +new Date(g.deadline)) ? d : best), s.dates[0])}
            stroke="var(--muted)"
            strokeDasharray="3 4"
            label={{ value: g.name.split(" (")[0], position: "insideTopLeft", fontSize: 10, fill: "var(--muted)" }}
          />
        ))}
        {sc && <Legend wrapperStyle={{ fontSize: 12 }} />}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function CashflowChart({ data, height = 260 }: { data: { month: string; income: number; spend: number; net: number }[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 10, right: 8, left: 0, bottom: 0 }} barGap={4}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="month" tick={AXIS} tickFormatter={monthLabel} axisLine={false} tickLine={false} />
        <YAxis tick={AXIS} tickFormatter={(v) => money(v, { compact: true })} width={60} axisLine={false} tickLine={false} />
        <Tooltip content={<ChartTooltip labelFmt={monthLabel} />} cursor={{ fill: "var(--surface-2)" }} />
        <Bar dataKey="income" name="Income" fill="#10b981" radius={[6, 6, 0, 0]} maxBarSize={26} />
        <Bar dataKey="spend" name="Spending" fill="#818cf8" radius={[6, 6, 0, 0]} maxBarSize={26} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DonutChart({
  data,
  height = 220,
}: {
  data: { label: string; amount: number; color: string }[];
  height?: number;
}) {
  const total = data.reduce((a, d) => a + d.amount, 0);
  return (
    <div className="relative">
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie data={data} dataKey="amount" nameKey="label" innerRadius="62%" outerRadius="90%" paddingAngle={2} stroke="none" isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.label} fill={d.color} />
            ))}
          </Pie>
          <Tooltip content={<ChartTooltip />} />
        </PieChart>
      </ResponsiveContainer>
      <div className="pointer-events-none absolute inset-0 grid place-items-center text-center">
        <div>
          <div className="text-[11px] text-muted">30-day spend</div>
          <div className="num text-lg font-semibold">{money(total, { compact: true })}</div>
        </div>
      </div>
    </div>
  );
}

export function SimpleBars({
  data,
  height = 220,
  keys,
}: {
  data: Record<string, any>[];
  height?: number;
  keys: { key: string; name: string; color: string }[];
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 10, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="x" tick={AXIS} axisLine={false} tickLine={false} interval={0} />
        <YAxis tick={AXIS} width={56} axisLine={false} tickLine={false} />
        <Tooltip cursor={{ fill: "var(--surface-2)" }} contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 12, fontSize: 12 }} />
        {keys.map((k) => (
          <Bar key={k.key} dataKey={k.key} name={k.name} fill={k.color} radius={[6, 6, 0, 0]} maxBarSize={34} />
        ))}
        {keys.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function Sparkline({ values, color = "#10b981", height = 36 }: { values: number[]; color?: string; height?: number }) {
  const rows = values.map((v, i) => ({ i, v }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} margin={{ top: 4, right: 2, left: 2, bottom: 2 }}>
        <YAxis hide domain={[0, 1]} />
        <Line dataKey="v" stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
