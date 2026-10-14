import { useMutation, useQuery } from "@tanstack/react-query";
import { FlaskConical, Plus, Sparkles, Trash2, Wand2 } from "lucide-react";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import { ForecastChart } from "../components/charts";
import { Verdict } from "../components/domain";
import { Badge, Button, Card, CardHeader, Empty, Input, Label, PageHeader, Select } from "../components/ui";
import { api } from "../lib/api";
import { addDays, cn, date, money, prob } from "../lib/format";
import { useCategories } from "../lib/hooks";
import type { Adjustment, ForecastSummary, RecurringItem, SimulationResult } from "../lib/types";

const TYPES: { value: Adjustment["type"]; label: string; help: string }[] = [
  { value: "one_time", label: "One-off purchase", help: "A trip, a laptop, a wedding gift…" },
  { value: "recurring", label: "New recurring cost", help: "A new EMI, rent increase, a subscription" },
  { value: "category_change", label: "Change a spending habit", help: "e.g. cut food delivery by 30%" },
  { value: "cancel_recurring", label: "Cancel subscriptions", help: "Drop recurring charges" },
  { value: "income_change", label: "Income change", help: "Raise, pay cut, job switch" },
];

type Draft = Adjustment & { key: number };
let nextKey = 1;

export default function Simulator() {
  const cats = useCategories();
  const { data: fc } = useQuery({ queryKey: ["forecast"], queryFn: () => api.get<ForecastSummary>("/forecast") });
  const { data: rec } = useQuery({
    queryKey: ["recurring"],
    queryFn: () => api.get<{ items: RecurringItem[] }>("/recurring"),
  });
  const asOf = fc?.as_of ?? new Date().toISOString().slice(0, 10);
  const [drafts, setDrafts] = useState<Draft[]>([]);
  const sim = useMutation({
    mutationFn: (adjs: Adjustment[]) => api.post<SimulationResult>("/simulate", { adjustments: adjs }),
    onError: (e: Error) => toast.error(e.message),
  });

  const subs = (rec?.items ?? []).filter((r) => r.status === "active" && r.kind === "subscription");
  const discretionary = Object.values(cats).filter((c) => ["discretionary", "essential"].includes(c.group));

  const presets = useMemo(
    () => [
      { label: `Trip next month (${money(45000, { compact: true })})`, adj: [{ type: "one_time", amount: 45000, date: addDays(asOf, 30), label: "Trip" }] },
      { label: `New ${money(80000, { compact: true })} laptop now`, adj: [{ type: "one_time", amount: 80000, date: addDays(asOf, 3), label: "Laptop" }] },
      {
        label: "Cut food delivery 30%",
        adj: [{ type: "category_change", category: "food_delivery", pct: -30, date: asOf }],
      },
      {
        label: "Cancel unused + overlapping subs",
        adj: [{ type: "cancel_recurring", merchants: (rec?.items ?? []).filter((r) => r.unused || (r.overlap_with?.length ?? 0) > 0 && r.monthly_cost < 500).map((r) => r.merchant) }],
      },
      { label: `Car EMI ${money(12000, { compact: true })}/month`, adj: [{ type: "recurring", amount: 12000, cadence: "monthly", date: addDays(asOf, 20), label: "Car EMI" }] },
      { label: "10% raise", adj: [{ type: "income_change", pct: 10, date: addDays(asOf, 20) }] },
    ] as { label: string; adj: Adjustment[] }[],
    [asOf, rec],
  );

  const add = (type: Adjustment["type"]) =>
    setDrafts((d) => [
      ...d,
      {
        key: nextKey++,
        type,
        amount: type === "one_time" ? 40000 : type === "recurring" ? 5000 : 0,
        kind: "expense",
        date: type === "one_time" ? addDays(asOf, 30) : asOf,
        cadence: "monthly",
        category: type === "category_change" ? "food_delivery" : null,
        pct: type === "category_change" ? -25 : type === "income_change" ? 10 : null,
        merchants: [],
        label: "",
      },
    ]);
  const patch = (key: number, p: Partial<Draft>) => setDrafts((d) => d.map((x) => (x.key === key ? { ...x, ...p } : x)));
  const runPreset = (adj: Adjustment[]) => {
    setDrafts(adj.map((a) => ({ ...a, key: nextKey++ })));
    sim.mutate(adj);
  };
  const runDrafts = () => {
    const adjs = drafts.map(({ key: _key, ...a }) => a).filter((a) => a.type !== "cancel_recurring" || a.merchants?.length);
    if (!adjs.length) return toast.error("Add at least one change");
    sim.mutate(adjs);
  };
  const r = sim.data;

  return (
    <div>
      <PageHeader
        title="What-if Lab"
        subtitle="Simulate a decision against your real trajectory before you commit. Baseline and scenario share the same 400 random futures, so the difference is the decision — not noise."
      />
      <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
        <div className="space-y-4">
          <Card className="p-5">
            <div className="mb-3 flex items-center gap-2 text-[13px] font-semibold">
              <Wand2 className="size-4 text-brand" /> Quick scenarios
            </div>
            <div className="flex flex-wrap gap-2">
              {presets.map((p) => (
                <button
                  key={p.label}
                  onClick={() => runPreset(p.adj)}
                  className="rounded-full border border-line bg-surface-2 px-3 py-1.5 text-[12.5px] transition hover:border-brand/50 hover:text-brand"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </Card>

          <Card className="p-5">
            <div className="mb-3 flex items-center justify-between">
              <div className="text-[13px] font-semibold">Build your own</div>
              <Select className="h-8 w-48 text-[12.5px]" value="" onChange={(e) => e.target.value && add(e.target.value as Adjustment["type"])}>
                <option value="">+ Add a change…</option>
                {TYPES.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </Select>
            </div>
            {!drafts.length && <p className="text-[13px] text-muted">Pick a quick scenario or add changes — they combine.</p>}
            <div className="space-y-3">
              {drafts.map((d) => (
                <div key={d.key} className="rounded-xl border border-line p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <Badge tone="violet">{TYPES.find((t) => t.value === d.type)?.label}</Badge>
                    <button onClick={() => setDrafts((x) => x.filter((y) => y.key !== d.key))} className="text-muted hover:text-red-500" aria-label="Remove">
                      <Trash2 className="size-4" />
                    </button>
                  </div>
                  {(d.type === "one_time" || d.type === "recurring") && (
                    <div className="grid grid-cols-2 gap-2">
                      <div className="col-span-2">
                        <Label>What is it?</Label>
                        <Input value={d.label} placeholder="e.g. Goa trip" onChange={(e) => patch(d.key, { label: e.target.value })} />
                      </div>
                      <div>
                        <Label>Amount</Label>
                        <Input type="number" min={0} value={d.amount} onChange={(e) => patch(d.key, { amount: Number(e.target.value) })} />
                      </div>
                      <div>
                        <Label>{d.type === "one_time" ? "When" : "Starting"}</Label>
                        <Input type="date" value={d.date ?? ""} onChange={(e) => patch(d.key, { date: e.target.value })} />
                      </div>
                      <div>
                        <Label>Direction</Label>
                        <Select value={d.kind} onChange={(e) => patch(d.key, { kind: e.target.value as "expense" | "income" })}>
                          <option value="expense">Money out</option>
                          <option value="income">Money in</option>
                        </Select>
                      </div>
                      {d.type === "recurring" && (
                        <div>
                          <Label>Every</Label>
                          <Select value={d.cadence} onChange={(e) => patch(d.key, { cadence: e.target.value })}>
                            {["weekly", "monthly", "quarterly", "yearly"].map((c) => (
                              <option key={c}>{c}</option>
                            ))}
                          </Select>
                        </div>
                      )}
                    </div>
                  )}
                  {d.type === "category_change" && (
                    <div className="grid grid-cols-2 gap-2">
                      <div className="col-span-2">
                        <Label>Category</Label>
                        <Select value={d.category ?? ""} onChange={(e) => patch(d.key, { category: e.target.value })}>
                          {discretionary.map((c) => (
                            <option key={c.id} value={c.id}>
                              {c.label}
                            </option>
                          ))}
                        </Select>
                      </div>
                      <div className="col-span-2">
                        <Label hint={`${d.pct! > 0 ? "+" : ""}${d.pct}%`}>Change</Label>
                        <input type="range" min={-80} max={80} step={5} value={d.pct ?? 0} onChange={(e) => patch(d.key, { pct: Number(e.target.value) })} className="w-full accent-emerald-500" />
                        {fc?.category_monthly?.[d.category ?? ""] !== undefined && (
                          <div className="text-[12px] text-muted">
                            Currently ~{money(fc!.category_monthly![d.category!])}/month →{" "}
                            {money(fc!.category_monthly![d.category!] * (1 + (d.pct ?? 0) / 100))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  {d.type === "cancel_recurring" && (
                    <div className="flex flex-wrap gap-1.5">
                      {subs.map((s) => {
                        const on = d.merchants?.includes(s.merchant);
                        return (
                          <button
                            key={s.id}
                            onClick={() =>
                              patch(d.key, {
                                merchants: on ? d.merchants!.filter((m) => m !== s.merchant) : [...(d.merchants ?? []), s.merchant],
                              })
                            }
                            className={cn(
                              "rounded-full border px-2.5 py-1 text-[12px]",
                              on ? "border-red-500/40 bg-red-500/10 text-red-500 line-through" : "border-line bg-surface-2",
                            )}
                          >
                            {s.merchant} · {money(s.monthly_cost)}
                          </button>
                        );
                      })}
                    </div>
                  )}
                  {d.type === "income_change" && (
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <Label>Change %</Label>
                        <Input type="number" value={d.pct ?? 0} onChange={(e) => patch(d.key, { pct: Number(e.target.value) })} />
                      </div>
                      <div>
                        <Label>From</Label>
                        <Input type="date" value={d.date ?? ""} onChange={(e) => patch(d.key, { date: e.target.value })} />
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div className="mt-4 flex gap-2">
              <Button variant="primary" className="flex-1" loading={sim.isPending} onClick={runDrafts} disabled={!drafts.length}>
                <FlaskConical className="size-4" /> Simulate
              </Button>
              {drafts.length > 0 && (
                <Button onClick={() => add("one_time")}>
                  <Plus className="size-4" />
                </Button>
              )}
            </div>
          </Card>
        </div>

        <div className="space-y-4">
          {!r ? (
            <Card className="h-full">
              <Empty icon={<FlaskConical className="size-6" />} title="Ask “can I afford it?” before you commit">
                The Forecaster replays your scenario on your real recurring bills, spending habits and goals, then tells you
                whether it fits, what it does to each goal, and how to make it fit if it doesn't.
              </Empty>
            </Card>
          ) : (
            <>
              <Card className="p-5">
                <div className="flex flex-wrap items-center gap-3">
                  <Verdict verdict={r.verdict} />
                  <div className="text-[13px] text-muted">{r.adjustments.map((a) => a.description).join(" + ")}</div>
                </div>
                <ul className="mt-3 space-y-1 text-[14px]">
                  {r.reasons.map((x, i) => (
                    <li key={i}>• {x}</li>
                  ))}
                </ul>
                {r.suggestions.length > 0 && (
                  <div className="mt-4 space-y-2">
                    {r.suggestions.map((s, i) => (
                      <div key={i} className="flex gap-2 rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-3 text-[13.5px]">
                        <Sparkles className="mt-0.5 size-4 shrink-0 text-emerald-500" />
                        {s.text}
                      </div>
                    ))}
                  </div>
                )}
              </Card>
              <Card>
                <CardHeader title="Baseline vs with this decision" subtitle={r.baseline.projection_type === "cashflow_change" ? "Net cash-flow changes; starting balance unverified" : "Median path and P10–P90 band for both"} />
                <div className="px-2 pb-4">
                  <ForecastChart forecast={r.baseline} scenario={r.scenario} height={320} />
                </div>
              </Card>
              <div className="grid gap-4 md:grid-cols-3">
                {[
                  [r.scenario.projection_type === "cashflow_change" ? "End cash-flow change (P50)" : "End balance (P50)", r.scenario.end_balance.p50, r.delta.end_balance_p50],
                  ["Lowest point (P50)", r.scenario.low_point.p50, r.delta.low_point_p50],
                  ["Lower estimate (P10)", r.scenario.low_point.p10, r.scenario.low_point.p10 - r.baseline.low_point.p10],
                ].map(([k, v, d]) => (
                  <Card key={k as string} className="p-4">
                    <div className="text-[12.5px] text-muted">{k}</div>
                    <div className="num mt-1 text-xl font-semibold">{money(v as number)}</div>
                    <div className={cn("num text-[12.5px]", (d as number) >= 0 ? "text-emerald-500" : "text-red-500")}>
                      {money(d as number, { sign: true })} vs baseline
                    </div>
                  </Card>
                ))}
              </div>
              {r.delta.goals.length > 0 && (
                <Card>
                  <CardHeader title="Impact on your goals" />
                  <div className="overflow-x-auto p-5 pt-3">
                    <table className="w-full text-[13px]">
                      <thead className="text-left text-muted">
                        <tr>
                          <th className="py-2 font-medium">Goal</th>
                          <th className="font-medium">Chance before → after</th>
                          <th className="font-medium">ETA before → after</th>
                          <th className="font-medium">Shortfall after</th>
                        </tr>
                      </thead>
                      <tbody>
                        {r.delta.goals.map((g) => (
                          <tr key={g.name} className="border-t border-line">
                            <td className="py-2 font-medium">{g.name}</td>
                            <td className={cn("num", g.prob_after < g.prob_before - 0.001 ? "text-red-500" : g.prob_after > g.prob_before + 0.001 ? "text-emerald-500" : "")}>
                              {prob(g.prob_before)} → {prob(g.prob_after)}
                            </td>
                            <td className="num">
                              {date(g.eta_before, true)} → {date(g.eta_after, true)}
                              {g.delay_days ? <span className={g.delay_days > 0 ? "text-red-500" : "text-emerald-500"}> ({g.delay_days > 0 ? "+" : ""}{g.delay_days}d)</span> : null}
                            </td>
                            <td className="num">{money(g.shortfall_after)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
