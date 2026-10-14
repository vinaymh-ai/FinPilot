import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRight,
  Ban,
  BadgeCheck,
  Check,
  ChevronRight,
  CircleDot,
  ShieldCheck,
  ThumbsDown,
  ThumbsUp,
  TriangleAlert,
} from "lucide-react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { api } from "../lib/api";
import { cn, date, dateTime, money, prob, titleCase } from "../lib/format";
import { useCategories } from "../lib/hooks";
import type { AgentRun, Goal, GoalProjection, Insight, Recommendation, Txn } from "../lib/types";
import { Badge, Button, Drawer, ModelBadge, Progress, SeverityIcon, Spinner, type Tone } from "./ui";

export function CategoryPill({ id, label }: { id: string | null; label?: string }) {
  const cats = useCategories();
  const c = id ? cats[id] : undefined;
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full bg-surface-2 px-2 py-0.5 text-[12px] font-medium">
      <span className="size-2 rounded-full" style={{ background: c?.color ?? "#a1a1aa" }} />
      {c?.label ?? label ?? "Uncategorised"}
    </span>
  );
}

const SOURCE_TONE: Record<string, Tone> = {
  rule: "blue",
  keyword: "amber",
  llm: "violet",
  jev: "cyan",
  memory: "green",
  user: "green",
  history: "green",
  pattern: "violet",
  import: "neutral",
  fallback: "red",
  review: "amber",
  p2p: "amber",
  consistency: "blue",
};
export function SourceBadge({ source, confidence, reason }: { source: string | null; confidence?: number | null; reason?: string | null }) {
  if (!source) return <Badge>pending</Badge>;
  return (
    <Badge tone={SOURCE_TONE[source] ?? "neutral"} title={reason ?? undefined}>
      {source}
      {confidence != null && <span className="opacity-70">· {Math.round(confidence * 100)}%</span>}
    </Badge>
  );
}

export function InsightRow({ insight, compact }: { insight: Insight; compact?: boolean }) {
  return (
    <div className="flex gap-3 py-3">
      <SeverityIcon severity={insight.severity} className="mt-0.5" />
      <div className="min-w-0">
        <div className="text-[13.5px] font-medium">{insight.title}</div>
        {!compact && <div className="mt-0.5 text-[13px] text-muted">{insight.detail}</div>}
      </div>
    </div>
  );
}

const URGENCY_TONE: Record<string, Tone> = { high: "red", normal: "amber", low: "neutral" };
const KIND_LABEL: Record<string, string> = {
  cancel_unused_subscription: "Unused subscription",
  consolidate_subscriptions: "Overlapping subscriptions",
  subscription_price_hike: "Price hike",
  category_spike: "Spending spike",
  impulse_spending: "Impulse buys",
  dispute_duplicate: "Duplicate charge",
  avoid_fees: "Avoidable fees",
  cashflow_risk: "Cash-flow risk",
  goal_at_risk: "Goal at risk",
  goal_on_track: "Goal on track",
  idle_cash: "Idle cash",
};

export function RecommendationCard({ rec, onOpen }: { rec: Recommendation; onOpen?: (r: Recommendation) => void }) {
  const qc = useQueryClient();
  const update = useMutation({
    mutationFn: (status: Recommendation["status"]) => api.patch(`/recommendations/${rec.id}`, { status }),
    onSuccess: (_, status) => {
      toast.success(
        status === "dismissed" ? "Snoozed — the Advisor will learn from this" : `Marked as ${status}`,
      );
      qc.invalidateQueries({ queryKey: ["recommendations"] });
      qc.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });
  const done = rec.status === "done" || rec.status === "accepted";
  return (
    <div
      className={cn(
        "group rounded-2xl border border-line bg-surface p-4 transition hover:border-brand/40",
        rec.status === "dismissed" && "opacity-60",
      )}
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={URGENCY_TONE[rec.urgency]}>{rec.urgency === "high" ? "urgent" : rec.urgency}</Badge>
        <Badge>{KIND_LABEL[rec.kind] ?? titleCase(rec.kind)}</Badge>
        {done && (
          <Badge tone="green">
            <Check className="size-3" /> {rec.status}
          </Badge>
        )}
        <span className="ml-auto">
          <ModelBadge model={rec.generated_by} />
        </span>
      </div>
      <div className="mt-2.5 text-[15px] font-semibold leading-snug">{rec.title}</div>
      <p className="mt-1 text-[13.5px] leading-relaxed text-muted">{rec.message}</p>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {rec.monthly_impact > 0 && (
          <span className="num rounded-lg bg-emerald-500/10 px-2 py-1 text-[12.5px] font-semibold text-emerald-600 dark:text-emerald-400">
            {money(rec.monthly_impact)}/mo
          </span>
        )}
        <span className="text-[12px] text-muted">→ {rec.action}</span>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-line pt-3">
        <Button size="sm" variant="ghost" onClick={() => onOpen?.(rec)} className="text-brand">
          Why? <ChevronRight className="size-3.5" />
        </Button>
        <div className="ml-auto flex gap-1">
          {!done && (
            <Button size="sm" variant="secondary" onClick={() => update.mutate("accepted")} title="I'll do this">
              <ThumbsUp className="size-3.5" /> Accept
            </Button>
          )}
          {rec.status === "accepted" && (
            <Button size="sm" variant="secondary" onClick={() => update.mutate("done")}>
              <BadgeCheck className="size-3.5" /> Done
            </Button>
          )}
          {rec.status !== "dismissed" && (
            <Button size="sm" variant="ghost" onClick={() => update.mutate("dismissed")} title="Not useful">
              <ThumbsDown className="size-3.5" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

function EvidenceValue({ v }: { v: unknown }) {
  if (v === null || v === undefined) return <span className="text-muted">–</span>;
  if (typeof v === "number") return <span className="num">{Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : v}</span>;
  if (typeof v === "string") return <span>{v}</span>;
  if (Array.isArray(v)) {
    if (v.length && typeof v[0] === "object")
      return (
        <div className="space-y-0.5">
          {v.slice(0, 6).map((it, i) => (
            <div key={i} className="text-[12px]">
              {Object.entries(it as object)
                .map(([k, val]) => `${k}: ${typeof val === "number" ? val.toLocaleString() : String(val)}`)
                .join(" · ")}
            </div>
          ))}
        </div>
      );
    return <span>{v.slice(0, 12).join(", ")}{v.length > 12 ? "…" : ""}</span>;
  }
  return <code className="text-[11px]">{JSON.stringify(v)}</code>;
}

/** The decision trail: trigger → evidence → reasoning → agent run that produced it. */
export function DecisionTrail({ rec, onClose }: { rec: Recommendation | null; onClose: () => void }) {
  const { data, isLoading } = useQuery({
    queryKey: ["recommendation", rec?.id],
    queryFn: () =>
      api.get<{ recommendation: Recommendation; run: AgentRun; transactions: Txn[]; previous: any[] }>(
        `/recommendations/${rec!.id}`,
      ),
    enabled: !!rec,
  });
  const ev = Object.entries(rec?.evidence ?? {}).filter(([k]) => k !== "transaction_ids");
  const [agentName, signal] = (rec?.trigger ?? ":").split(":");
  return (
    <Drawer open={!!rec} onClose={onClose} title="Why this recommendation?" wide>
      {rec && (
        <div className="space-y-6">
          <div>
            <div className="text-lg font-semibold">{rec.title}</div>
            <p className="mt-1 text-sm text-muted">{rec.message}</p>
          </div>

          <section>
            <h4 className="mb-2 text-[12px] font-semibold uppercase tracking-wider text-muted">1 · Trigger</h4>
            <div className="flex items-center gap-2 rounded-xl border border-line bg-surface-2 p-3 text-sm">
              <Badge tone="violet">{titleCase(agentName)} agent</Badge>
              <ArrowRight className="size-4 text-muted" />
              <span className="font-medium">{titleCase(signal ?? "")}</span>
            </div>
          </section>

          <section>
            <h4 className="mb-2 text-[12px] font-semibold uppercase tracking-wider text-muted">2 · Evidence</h4>
            <div className="overflow-hidden rounded-xl border border-line">
              <table className="w-full text-[13px]">
                <tbody>
                  {ev.map(([k, v]) => (
                    <tr key={k} className="border-b border-line last:border-0">
                      <td className="w-44 bg-surface-2 px-3 py-2 align-top text-muted">{titleCase(k)}</td>
                      <td className="px-3 py-2">
                        <EvidenceValue v={v} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section>
            <h4 className="mb-2 text-[12px] font-semibold uppercase tracking-wider text-muted">3 · Reasoning</h4>
            <ol className="relative space-y-3 border-l border-line pl-5">
              {rec.reasoning.map((step, i) => (
                <li key={i} className="text-[13px]">
                  <CircleDot className="absolute -left-[7px] size-3.5 bg-surface text-brand" />
                  {step}
                </li>
              ))}
            </ol>
            <div className="mt-3 flex flex-wrap items-center gap-2 text-[12px] text-muted">
              <ShieldCheck className="size-4 text-emerald-500" />
              {rec.grounded ? "Every number traced to computed evidence" : "Grounding check failed"}
              <span>·</span> written by <ModelBadge model={rec.generated_by} />
              <span>·</span> priority {rec.priority.toFixed(2)}
            </div>
          </section>

          {isLoading && <Spinner />}
          {data?.transactions?.length ? (
            <section>
              <h4 className="mb-2 text-[12px] font-semibold uppercase tracking-wider text-muted">
                Transactions behind it
              </h4>
              <div className="divide-y divide-line rounded-xl border border-line">
                {data.transactions.slice(0, 10).map((t) => (
                  <div key={t.id} className="flex items-center justify-between gap-3 px-3 py-2 text-[13px]">
                    <span className="w-16 shrink-0 text-muted">{date(t.date)}</span>
                    <span className="min-w-0 flex-1 truncate">{t.merchant}</span>
                    <span className="num font-medium">{money(t.amount)}</span>
                  </div>
                ))}
              </div>
            </section>
          ) : null}

          {data?.run && (
            <section>
              <h4 className="mb-2 text-[12px] font-semibold uppercase tracking-wider text-muted">4 · Agent run</h4>
              <div className="rounded-xl border border-line p-3 text-[13px]">
                <div className="mb-2 flex items-center justify-between">
                  <span>
                    Run #{data.run.id} · {data.run.trigger} · {dateTime(data.run.started_at)}
                  </span>
                  <Link to={`/agents?run=${data.run.id}`} className="text-brand hover:underline">
                    Open trail →
                  </Link>
                </div>
                <div className="space-y-1">
                  {data.run.steps?.map((s) => (
                    <div key={s.id} className="flex gap-2 text-[12.5px]">
                      <span className="w-24 shrink-0 font-medium capitalize">{s.agent}</span>
                      <span className="text-muted">{s.summary}</span>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          )}
          {data?.previous?.length ? (
            <div className="text-[12px] text-muted">
              Seen in {data.previous.length} earlier run(s) — statuses:{" "}
              {data.previous.map((p) => p.status).join(", ")}
            </div>
          ) : null}
        </div>
      )}
    </Drawer>
  );
}

export function goalTone(status: GoalProjection["status"] | undefined): "brand" | "amber" | "red" {
  return status === "off_track" ? "red" : status === "at_risk" ? "amber" : "brand";
}

export function GoalStatusBadge({ status }: { status?: GoalProjection["status"] }) {
  if (!status) return <Badge>not projected</Badge>;
  const tone: Tone = status === "on_track" || status === "achieved" ? "green" : status === "at_risk" ? "amber" : "red";
  return (
    <Badge tone={tone}>
      {status === "off_track" && <TriangleAlert className="size-3" />}
      {titleCase(status)}
    </Badge>
  );
}

export function GoalMini({ goal }: { goal: Goal }) {
  const p = goal.projection;
  const projectedPct = p ? (p.projected_at_deadline / goal.target_amount) * 100 : goal.progress_pct;
  return (
    <div className="rounded-xl border border-line p-3.5">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0 truncate text-[13.5px] font-medium">{goal.name}</div>
        <GoalStatusBadge status={p?.status} />
      </div>
      <div className="mt-2 flex items-baseline justify-between text-[12.5px] text-muted">
        <span className="num">
          {money(goal.saved_amount, { compact: true })} / {money(goal.target_amount, { compact: true })}
        </span>
        <span>by {date(goal.deadline, true)}</span>
      </div>
      <Progress value={projectedPct} tone={goalTone(p?.status)} className="mt-2" />
      {p && (
        <div className="mt-2 text-[12px] text-muted">
          {prob(p.probability)} likely on time
          {p.shortfall > 0 ? ` · ${money(p.shortfall)} short at this pace` : ""}
        </div>
      )}
    </div>
  );
}

export function Verdict({ verdict }: { verdict: "affordable" | "stretch" | "not_affordable" | "needs_balance" }) {
  const map = {
    affordable: { tone: "green" as Tone, icon: <Check className="size-3.5" />, label: "Affordable" },
    stretch: { tone: "amber" as Tone, icon: <TriangleAlert className="size-3.5" />, label: "A stretch" },
    not_affordable: { tone: "red" as Tone, icon: <Ban className="size-3.5" />, label: "Not affordable" },
    needs_balance: { tone: "amber" as Tone, icon: <TriangleAlert className="size-3.5" />, label: "Verify balance to decide affordability" },
  }[verdict];
  return (
    <Badge tone={map.tone} className="px-2.5 py-1 text-[12.5px]">
      {map.icon} {map.label}
    </Badge>
  );
}
