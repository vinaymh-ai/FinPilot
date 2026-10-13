import { useMutation, useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  Brain,
  CalendarClock,
  Lightbulb,
  LineChart,
  PiggyBank,
  Repeat,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
  Upload,
  Wallet,
} from "lucide-react";
import { toast } from "sonner";
import { useState } from "react";
import { Link } from "react-router-dom";
import { CashflowChart, DonutChart, ForecastChart } from "../components/charts";
import { DecisionTrail, GoalMini, InsightRow, RecommendationCard } from "../components/domain";
import { Badge, Button, Card, CardHeader, Empty, ModelBadge, Skeleton, Spinner, Stat } from "../components/ui";
import { api } from "../lib/api";
import { date, dateTime, money, pct } from "../lib/format";
import { useCategories, useRefreshAll, useRunAgents } from "../lib/hooks";
import type { Dashboard, Recommendation } from "../lib/types";

export default function Overview() {
  const { data, isLoading } = useQuery({
    queryKey: ["dashboard"],
    queryFn: () => api.get<Dashboard>("/dashboard"),
    // While the agents analyse new data (a fresh demo, a just-uploaded statement) keep the view live.
    refetchInterval: (q) => {
      const d = q.state.data;
      const waiting = d && (d.analysis_running || (d.overview.transactions > 0 && !d.last_run));
      return waiting && q.state.dataUpdateCount < 60 ? 3000 : false;
    },
  });
  const cats = useCategories();
  const run = useRunAgents();
  const [open, setOpen] = useState<Recommendation | null>(null);

  if (isLoading || !data)
    return (
      <div className="space-y-4">
        <Skeleton className="h-28" />
        <div className="grid gap-4 md:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
        <Skeleton className="h-96" />
      </div>
    );

  const o = data.overview;
  const fc = data.forecast;
  if (!o.transactions) return <GetStarted />;

  const top = data.recommendations[0];
  const rate = o.last_30_days.savings_rate_pct;
  const cp30 = fc?.checkpoints.find((c) => c.days === 30);
  const cp90 = fc?.checkpoints.find((c) => c.days === 90);
  const donut = data.categories_30d.slice(0, 7).map((c) => ({
    label: c.label,
    amount: c.amount,
    color: cats[c.category]?.color ?? "#94a3b8",
  }));
  const rest = data.categories_30d.slice(7).reduce((a, c) => a + c.amount, 0);
  if (rest > 0) donut.push({ label: "Other", amount: rest, color: "#64748b" });

  return (
    <div className="space-y-6">
      {o.data_quality && (o.data_quality.needs_review > 0 || !o.data_quality.balance_known || o.data_quality.history_days < 90) && (
        <Card className="border-amber-300 p-4 text-sm">
          <div className="font-semibold">Check the evidence before relying on forecasts</div>
          <p className="mt-1 text-muted">
            Your data covers {o.data_quality.start} to {o.data_quality.end} ({o.data_quality.history_days} days).
            {o.data_quality.needs_review > 0 && ` ${o.data_quality.needs_review} transactions need category review.`}
            {!o.data_quality.balance_known && " Set a verified balance in Settings."}
          </p>
          <Link to="/transactions" className="mt-2 inline-block font-medium text-brand">Review transactions</Link>
        </Card>
      )}
      {/* Proactive headline */}
      <Card className="relative overflow-hidden p-6">
        <div className="absolute -right-24 -top-24 size-72 rounded-full bg-gradient-to-br from-emerald-400/20 to-indigo-500/20 blur-3xl" />
        <div className="relative flex flex-col gap-4 lg:flex-row lg:items-center">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 text-[12px] font-semibold uppercase tracking-wider text-brand">
              <Sparkles className="size-4" /> This week's key insight
            </div>
            <h2 className="mt-2 text-xl font-semibold leading-snug tracking-tight sm:text-2xl">
              {data.headline ??
                (data.analysis_running ? (
                  <span className="inline-flex items-center gap-2">
                    <Spinner className="size-5" /> Your agents are analysing these transactions…
                  </span>
                ) : (
                  "Run the agents to get your first insight."
                ))}
            </h2>
            {fc?.narrative && (
              <p className="mt-2 max-w-3xl text-[14px] leading-relaxed text-muted">
                {fc.narrative} <ModelBadge model={fc.narrative_model} />
              </p>
            )}
          </div>
          {top && (
            <div className="w-full shrink-0 rounded-2xl border border-line bg-surface-2/70 p-4 lg:w-96">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted">Your one action</div>
              <div className="mt-1 font-semibold">{top.action}</div>
              {top.monthly_impact > 0 && (
                <div className="num mt-1 text-sm text-emerald-500">≈ {money(top.monthly_impact)}/month</div>
              )}
              <Button size="sm" variant="ghost" className="mt-2 -ml-3 text-brand" onClick={() => setOpen(top)}>
                See why <ArrowRight className="size-3.5" />
              </Button>
            </div>
          )}
        </div>
      </Card>

      {/* KPIs */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat
          label="Balance"
          value={money(o.balance)}
          icon={<Wallet className="size-4" />}
          hint={cp30 ? `→ ${money(cp30.p50, { compact: true })} in 30 days` : `as of ${date(o.as_of)}`}
          tone={cp30 && cp30.p50 >= o.balance ? "up" : "down"}
        />
        <Stat
          label="Spent · last 30 days"
          value={money(o.last_30_days.spend)}
          icon={<TrendingDown className="size-4" />}
          hint={`income ${money(o.last_30_days.income, { compact: true })}`}
        />
        <Stat
          label="Savings rate · 30 days"
          value={pct(rate)}
          icon={<PiggyBank className="size-4" />}
          hint={rate !== null && rate < 0 ? "spending exceeded income" : "of income kept"}
          tone={rate !== null && rate >= 20 ? "up" : rate !== null && rate < 5 ? "down" : "neutral"}
        />
        <Stat
          label="Subscriptions"
          value={`${money(o.subscriptions.monthly_cost)}/mo`}
          icon={<Repeat className="size-4" />}
          hint={`${o.subscriptions.count} active`}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        {/* Forecast */}
        <Card className="xl:col-span-2">
          <CardHeader
            title="Where your money is heading"
            subtitle={
              fc
                ? `400 simulated futures from your own behaviour · shaded = likely range (P10–P90)`
                : "No forecast yet"
            }
            icon={<LineChart className="size-4" />}
            action={
              <Link to="/simulator">
                <Button size="sm">Try a what-if</Button>
              </Link>
            }
          />
          {fc ? (
            <div className="px-2 pb-4 pt-2">
              <ForecastChart forecast={fc} height={300} />
              <div className="grid grid-cols-2 gap-3 px-3 pt-2 sm:grid-cols-4">
                {[
                  ["In 90 days", cp90 ? money(cp90.p50) : "–"],
                  ["Tightest point", `${money(fc.low_point.p50, { compact: true })} · ${date(fc.low_point.date)}`],
                  ["Avg monthly net", money(fc.summary.avg_monthly_net, { sign: true })],
                  ["Overdraft risk", pct(fc.prob_overdraft * 100)],
                ].map(([k, v]) => (
                  <div key={k} className="rounded-xl bg-surface-2/70 px-3 py-2">
                    <div className="text-[11.5px] text-muted">{k}</div>
                    <div className="num text-[14px] font-semibold">{v}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <Empty title="Run the agents to build your forecast" />
          )}
        </Card>

        {/* Goals */}
        <Card>
          <CardHeader
            title="Goals"
            subtitle="Probability of hitting each goal on time"
            icon={<Target className="size-4" />}
            action={
              <Link to="/goals" className="text-[13px] text-brand hover:underline">
                Manage
              </Link>
            }
          />
          <div className="space-y-3 p-5">
            {data.goals.length ? (
              data.goals.map((g) => (
                <GoalMini
                  key={g.id}
                  goal={{ ...g, projection: fc?.goals.find((p) => p.id === g.id) ?? null }}
                />
              ))
            ) : (
              <Empty title="No goals yet">Add a goal and the Forecaster will track it.</Empty>
            )}
            {fc?.upcoming?.length ? (
              <div className="rounded-xl border border-line p-3">
                <div className="mb-2 flex items-center gap-1.5 text-[12.5px] font-medium">
                  <CalendarClock className="size-4 text-muted" /> Next 30 days
                </div>
                <div className="space-y-1.5">
                  {fc.upcoming.slice(0, 6).map((e, i) => (
                    <div key={i} className="flex items-center justify-between text-[12.5px]">
                      <span className="text-muted">
                        {date(e.date)} · <span className="text-ink">{e.merchant}</span>
                      </span>
                      <span className={`num font-medium ${e.amount > 0 ? "text-emerald-500" : ""}`}>
                        {money(e.amount, { sign: true })}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-2">
          <div className="flex items-center justify-between">
            <h3 className="flex items-center gap-2 font-semibold">
              <Lightbulb className="size-4 text-brand" /> Top recommendations
            </h3>
            <Link to="/advisor" className="text-[13px] text-brand hover:underline">
              All recommendations →
            </Link>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            {data.recommendations.slice(0, 4).map((r) => (
              <RecommendationCard key={r.id} rec={r} onOpen={setOpen} />
            ))}
          </div>
        </div>
        <Card>
          <CardHeader title="Detected patterns" subtitle="From your own history" icon={<Brain className="size-4" />} />
          <div className="divide-y divide-line px-5 pb-2">
            {data.insights.map((i) => (
              <InsightRow key={i.id} insight={i} />
            ))}
          </div>
          {data.profile.length > 0 && (
            <div className="mx-5 mb-5 rounded-xl bg-surface-2/70 p-3 text-[12.5px] text-muted">
              <div className="mb-1 font-medium text-ink">Behaviour profile</div>
              <ul className="list-disc space-y-0.5 pl-4">
                {data.profile.map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader title="Monthly cash-flow" subtitle="Income vs spending (excl. transfers & investments)" icon={<TrendingUp className="size-4" />} />
          <div className="px-2 pb-4 pt-2">
            <CashflowChart data={data.cashflow} />
          </div>
        </Card>
        <Card>
          <CardHeader title="Where it went" subtitle="Last 30 days by category" icon={<Sparkles className="size-4" />} />
          <div className="px-4 pb-4">
            <DonutChart data={donut} />
            <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1">
              {donut.map((d) => (
                <div key={d.label} className="flex items-center justify-between gap-2 text-[12px]">
                  <span className="flex min-w-0 items-center gap-1.5 truncate text-muted">
                    <span className="size-2 shrink-0 rounded-full" style={{ background: d.color }} />
                    {d.label}
                  </span>
                  <span className="num">{money(d.amount, { compact: true })}</span>
                </div>
              ))}
            </div>
          </div>
        </Card>
      </div>

      {data.last_run && (
        <div className="flex flex-wrap items-center gap-2 text-[12.5px] text-muted">
          <Badge tone={data.last_run.status === "succeeded" ? "green" : "red"}>{data.last_run.status}</Badge>
          Last agent run #{data.last_run.id} · {data.last_run.trigger} · {dateTime(data.last_run.started_at)} ·{" "}
          {data.last_run.duration_ms} ms
          <Link to={`/agents?run=${data.last_run.id}`} className="text-brand hover:underline">
            view trail
          </Link>
          <Button size="sm" variant="ghost" loading={run.isPending} onClick={() => run.mutate({})}>
            re-run
          </Button>
        </div>
      )}
      <DecisionTrail rec={open} onClose={() => setOpen(null)} />
    </div>
  );
}

function GetStarted() {
  const refresh = useRefreshAll();
  const sample = useMutation({
    mutationFn: () => api.post("/demo/reset", { persona: "young_professional" }),
    onSuccess: () => {
      toast.success("Sample data loaded and analysed");
      refresh();
    },
    onError: (e: Error) => toast.error(e.message),
  });
  return (
    <Card>
      <Empty icon={<Wallet className="size-6" />} title="No transactions yet">
        <p>Upload a bank statement — CSV, Excel or PDF from any bank — and the agents will categorise, analyse and forecast it.</p>
        <div className="mt-4 flex flex-wrap justify-center gap-2">
          <Link to="/transactions">
            <Button variant="primary">
              <Upload className="size-4" /> Upload a statement
            </Button>
          </Link>
          <Button loading={sample.isPending} onClick={() => sample.mutate()}>
            <Sparkles className="size-4" /> Explore with sample data
          </Button>
        </div>
      </Empty>
    </Card>
  );
}
