import { useQuery } from "@tanstack/react-query";
import { Lightbulb, ShieldCheck, Sparkles } from "lucide-react";
import { useState } from "react";
import { DecisionTrail, RecommendationCard } from "../components/domain";
import { Badge, Button, Card, Empty, PageHeader, Skeleton, Tabs } from "../components/ui";
import { api } from "../lib/api";
import { dateTime, money } from "../lib/format";
import { useRunAgents } from "../lib/hooks";
import type { AgentRun, Recommendation } from "../lib/types";

type Resp = { run: AgentRun | null; headline: string | null; recommendations: Recommendation[]; history?: Recommendation[] };

export default function Advisor() {
  const [tab, setTab] = useState<"current" | "history">("current");
  const { data, isLoading } = useQuery({
    queryKey: ["recommendations", tab],
    queryFn: () => api.get<Resp>(`/recommendations?include_history=${tab === "history"}`),
  });
  const run = useRunAgents();
  const [open, setOpen] = useState<Recommendation | null>(null);
  const recs = tab === "current" ? data?.recommendations ?? [] : data?.history ?? [];
  const totalImpact = (data?.recommendations ?? [])
    .filter((r) => !["goal_at_risk", "goal_on_track", "idle_cash"].includes(r.kind) && r.status !== "dismissed")
    .reduce((a, r) => a + r.monthly_impact, 0);

  return (
    <div>
      <PageHeader
        title="Advisor"
        subtitle="Specific, quantified actions — each with the pattern or forecast that triggered it."
        actions={
          <>
            <Tabs
              value={tab}
              onChange={setTab}
              items={[
                { value: "current", label: "Current" },
                { value: "history", label: "Your feedback" },
              ]}
            />
            <Button variant="primary" loading={run.isPending} onClick={() => run.mutate({})}>
              Re-run advisor
            </Button>
          </>
        }
      />
      {tab === "current" && data?.headline && (
        <Card className="mb-6 flex flex-col gap-3 p-5 sm:flex-row sm:items-center">
          <div className="grid size-10 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-emerald-400 to-indigo-500 text-white">
            <Sparkles className="size-5" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="font-semibold">{data.headline}</div>
            <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[12.5px] text-muted">
              {data.run && <>Run #{data.run.id} · {dateTime(data.run.started_at)}</>}
              <span className="inline-flex items-center gap-1">
                <ShieldCheck className="size-3.5 text-emerald-500" /> numbers grounded in your data
              </span>
            </div>
          </div>
          {totalImpact > 0 && (
            <div className="text-right">
              <div className="text-[12px] text-muted">Potential savings</div>
              <div className="num text-xl font-semibold text-emerald-500">{money(totalImpact)}/mo</div>
            </div>
          )}
        </Card>
      )}
      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} className="h-56" />
          ))}
        </div>
      ) : recs.length ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {recs.map((r) => (
            <RecommendationCard key={r.id} rec={r} onOpen={setOpen} />
          ))}
        </div>
      ) : (
        <Card>
          <Empty icon={<Lightbulb className="size-6" />} title={tab === "current" ? "No recommendations yet" : "No feedback yet"}>
            {tab === "current"
              ? "Run the agents to analyse your spending."
              : "Accept, complete or dismiss recommendations — the Advisor snoozes dismissed advice for 30 days and down-weights kinds you keep dismissing."}
          </Empty>
        </Card>
      )}
      {tab === "history" && recs.length > 0 && (
        <p className="mt-4 text-[12.5px] text-muted">
          <Badge>adaptive</Badge> Dismissed advice is snoozed for 30 days unless the problem grows by 25%+; accepted
          advice keeps its status across runs.
        </p>
      )}
      <DecisionTrail rec={open} onClose={() => setOpen(null)} />
    </div>
  );
}
