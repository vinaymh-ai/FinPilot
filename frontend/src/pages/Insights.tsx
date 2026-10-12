import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, Ban, Brain, CircleCheck, Moon, Repeat, TriangleAlert } from "lucide-react";
import { toast } from "sonner";
import { SimpleBars } from "../components/charts";
import { CategoryPill } from "../components/domain";
import { Badge, Button, Card, CardHeader, Empty, PageHeader, Select, SeverityIcon, Skeleton, severityTone } from "../components/ui";
import { api } from "../lib/api";
import { cn, date, money, pct, titleCase } from "../lib/format";
import type { AgentRun, Insight, RecurringItem } from "../lib/types";

type Resp = { run: AgentRun | null; insights: Insight[]; analysis: any };

export default function Insights() {
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({ queryKey: ["insights"], queryFn: () => api.get<Resp>("/insights") });
  const { data: rec } = useQuery({
    queryKey: ["recurring"],
    queryFn: () => api.get<{ items: RecurringItem[]; subscriptions_monthly: number; subscriptions_count: number }>("/recurring"),
  });
  const upd = useMutation({
    mutationFn: ({ id, body }: { id: number; body: Record<string, string> }) => api.patch(`/recurring/${id}`, body),
    onSuccess: () => {
      toast.success("Saved — the agents will use this on the next run");
      qc.invalidateQueries({ queryKey: ["recurring"] });
    },
  });
  if (isLoading) return <Skeleton className="h-96" />;
  const a = data?.analysis;
  if (!data?.run || !a)
    return (
      <Card>
        <Empty icon={<Brain className="size-6" />} title="No analysis yet">Run the agents first.</Empty>
      </Card>
    );
  const trends: any[] = (a.category_trends ?? []).filter((t: any) => t.last_30d > 0 || (t.baseline_30d ?? 0) > 0);
  const weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((d) => ({ x: d, spend: Math.round(a.behavior?.weekday_spend_90d?.[d] ?? 0) }));
  const findings = data.insights.filter((i) => i.kind !== "subscription_inventory");
  const items = rec?.items ?? [];
  const subs = items.filter((i) => i.kind === "subscription");
  const others = items.filter((i) => i.kind !== "subscription");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Insights"
        subtitle={`What the Analyzer found in your history — run #${data.run.id}, data up to ${date(a.as_of, true)}.`}
      />
      <div className="grid gap-6 xl:grid-cols-5">
        <Card className="xl:col-span-3">
          <CardHeader title="Patterns & anomalies" subtitle="Rolling 30-day windows vs your previous three, robust outlier tests" icon={<Activity className="size-4" />} />
          <div className="divide-y divide-line px-5 pb-3">
            {findings.map((f) => (
              <div key={f.id} className="flex gap-3 py-3.5">
                <SeverityIcon severity={f.severity} className="mt-0.5" />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[14px] font-medium">{f.title}</span>
                    <Badge tone={severityTone(f.severity)}>{titleCase(f.kind.replace("anomaly_", ""))}</Badge>
                  </div>
                  <div className="mt-0.5 text-[13px] text-muted">{f.detail}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
        <div className="space-y-6 xl:col-span-2">
          <Card>
            <CardHeader title="Behaviour profile" icon={<Brain className="size-4" />} />
            <ul className="space-y-2 p-5 pt-3 text-[13.5px]">
              {(a.profile ?? []).map((p: string, i: number) => (
                <li key={i} className="flex gap-2">
                  <span className="mt-2 size-1.5 shrink-0 rounded-full bg-brand" />
                  {p}
                </li>
              ))}
            </ul>
          </Card>
          <Card>
            <CardHeader
              title="Impulse purchases"
              subtitle="Discretionary buys made 22:00–04:00 or ≥3× your usual ticket"
              icon={<Moon className="size-4" />}
            />
            <div className="grid grid-cols-3 gap-3 p-5 pt-3">
              <div>
                <div className="text-[12px] text-muted">Last 30 days</div>
                <div className="num font-semibold">{money(a.impulse?.recent_total)}</div>
              </div>
              <div>
                <div className="text-[12px] text-muted">Purchases</div>
                <div className="num font-semibold">{a.impulse?.recent_count}</div>
              </div>
              <div>
                <div className="text-[12px] text-muted">Of discretionary</div>
                <div className="num font-semibold">{pct(a.impulse?.share_of_discretionary)}</div>
              </div>
            </div>
          </Card>
          <Card>
            <CardHeader title="Discretionary spend by weekday" subtitle={`Weekend share ${pct(a.behavior?.weekend_share_pct)} · last 90 days`} />
            <div className="px-2 pb-3">
              <SimpleBars data={weekday} keys={[{ key: "spend", name: "Spend", color: "#818cf8" }]} height={180} />
            </div>
          </Card>
        </div>
      </div>

      <Card>
        <CardHeader title="Category trends" subtitle="Last 30 days vs average of the previous 30-day windows" />
        <div className="overflow-x-auto p-5 pt-3">
          <table className="w-full min-w-[640px] text-[13px]">
            <thead className="text-left text-muted">
              <tr>
                <th className="py-2 font-medium">Category</th>
                <th className="text-right font-medium">Last 30 days</th>
                <th className="text-right font-medium">Usual</th>
                <th className="text-right font-medium">Change</th>
                <th className="pl-6 font-medium">History (oldest → newest window)</th>
              </tr>
            </thead>
            <tbody>
              {trends.map((t) => {
                const max = Math.max(t.last_30d, ...(t.history ?? []), 1);
                return (
                  <tr key={t.category} className="border-t border-line">
                    <td className="py-2.5">
                      <CategoryPill id={t.category} label={t.label} />
                    </td>
                    <td className="num text-right font-medium">{money(t.last_30d)}</td>
                    <td className="num text-right text-muted">{money(t.baseline_30d)}</td>
                    <td className={cn("num text-right font-medium", t.change_pct > 20 ? "text-red-500" : t.change_pct < -20 ? "text-emerald-500" : "text-muted")}>
                      {t.change_pct === null ? "–" : pct(t.change_pct, 0, true)}
                    </td>
                    <td className="pl-6">
                      <div className="flex h-6 items-end gap-1">
                        {[...(t.history ?? []), t.last_30d].map((v: number, i: number, arr: number[]) => (
                          <div
                            key={i}
                            className={cn("w-4 rounded-sm", i === arr.length - 1 ? "bg-brand" : "bg-line")}
                            style={{ height: `${Math.max(6, (v / max) * 100)}%` }}
                            title={money(v)}
                          />
                        ))}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card>
        <CardHeader
          title="Subscriptions & recurring payments"
          subtitle={`${rec?.subscriptions_count ?? 0} active subscriptions · ${money(rec?.subscriptions_monthly)}/month — detected from charge regularity, not names`}
          icon={<Repeat className="size-4" />}
        />
        <div className="overflow-x-auto p-5 pt-3">
          <table className="w-full min-w-[820px] text-[13px]">
            <thead className="text-left text-muted">
              <tr>
                <th className="py-2 font-medium">Merchant</th>
                <th className="font-medium">Cadence</th>
                <th className="text-right font-medium">Per month</th>
                <th className="pl-4 font-medium">Next charge</th>
                <th className="font-medium">Signals</th>
                <th className="font-medium">Do you use it?</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {[...subs, ...others].map((r) => (
                <tr key={r.id} className={cn("border-t border-line", r.status !== "active" && "opacity-55")}>
                  <td className="py-2.5">
                    <div className="font-medium">{r.merchant}</div>
                    <div className="text-[11.5px] text-muted">
                      {titleCase(r.kind)} · {r.occurrences}× since {date(r.first_date)}
                    </div>
                  </td>
                  <td className="capitalize text-muted">{r.cadence}</td>
                  <td className={cn("num text-right font-medium", r.signed_amount > 0 && "text-emerald-500")}>
                    {money(r.monthly_cost)}
                  </td>
                  <td className="pl-4 text-muted">{r.status === "active" ? date(r.next_expected) : titleCase(r.status)}</td>
                  <td>
                    <div className="flex flex-wrap gap-1">
                      {r.unused && (
                        <Badge tone="red" title={r.usage_signal ?? ""}>
                          <TriangleAlert className="size-3" /> unused
                        </Badge>
                      )}
                      {r.overlap_with && r.overlap_with.length > 0 && <Badge tone="amber">overlaps {r.overlap_with.join(", ")}</Badge>}
                      {r.price_change_pct ? <Badge tone="amber">price {pct(r.price_change_pct, 0, true)}</Badge> : null}
                      {r.usage_signal && !r.unused && <span className="text-[11.5px] text-muted">{r.usage_signal}</span>}
                    </div>
                  </td>
                  <td>
                    {r.kind === "subscription" && (
                      <Select className="h-8 w-36 text-[12.5px]" value={r.usage} onChange={(e) => upd.mutate({ id: r.id, body: { usage: e.target.value } })}>
                        <option value="unknown">Not sure</option>
                        <option value="used">I use it</option>
                        <option value="rarely_used">Rarely use it</option>
                      </Select>
                    )}
                  </td>
                  <td className="text-right">
                    {r.signed_amount < 0 &&
                      (r.status === "cancelled" ? (
                        <Button size="sm" variant="ghost" onClick={() => upd.mutate({ id: r.id, body: { status: "active" } })}>
                          <CircleCheck className="size-3.5" /> Cancelled · undo
                        </Button>
                      ) : (
                        <Button size="sm" variant="ghost" onClick={() => upd.mutate({ id: r.id, body: { status: "cancelled" } })} title="I cancelled this — drop it from forecasts">
                          <Ban className="size-3.5" /> Mark cancelled
                        </Button>
                      ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
