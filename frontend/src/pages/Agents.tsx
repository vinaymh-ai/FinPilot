import { useQuery } from "@tanstack/react-query";
import { Bot, CircleCheck, CircleX, Cpu, GitBranch, Scale, Timer } from "lucide-react";
import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { Badge, Card, CardHeader, Empty, ModelBadge, PageHeader, Skeleton } from "../components/ui";
import { api } from "../lib/api";
import { cn, dateTime } from "../lib/format";
import { useMeta } from "../lib/hooks";
import type { AgentRun, RunStep } from "../lib/types";

type RunDetail = {
  run: AgentRun;
  llm_calls: {
    agent: string;
    provider: string;
    model: string;
    prompt_tokens: number;
    completion_tokens: number;
    latency_ms: number;
    cached: boolean;
    success: boolean;
    error: string | null;
    kind: "llm" | "jev";
    purpose: string | null;
    cost_usd: number | null;
  }[];
};
type Usage = { items: { agent: string; provider: string; model: string; calls: number; prompt_tokens: number; completion_tokens: number; avg_latency_ms: number; cache_hits: number; successes: number }[]; total_calls: number; total_tokens: number };

const GRAPH = [
  { id: "coordinator", label: "plan" },
  { id: "categorizer", label: "categorize" },
  { id: "analyzer", label: "analyze" },
  { id: "forecaster", label: "forecast" },
  { id: "advisor", label: "advise" },
  { id: "digest", label: "digest" },
];

function jevCount(s: RunStep): number {
  const j = s.details?.jev_calls as { count?: number } | unknown[] | undefined;
  if (!j) return 0;
  return Array.isArray(j) ? j.length : (j.count ?? 0);
}

function GraphView({ steps }: { steps: RunStep[] }) {
  const by = Object.fromEntries(steps.map((s) => [s.agent, s]));
  return (
    <div className="overflow-x-auto">
      <div className="flex min-w-[720px] items-center gap-2 py-2">
        <div className="rounded-full border border-line px-3 py-1 text-[11px] text-muted">START</div>
        {GRAPH.map((n, i) => {
          const s = by[n.id];
          const tone = !s ? "border-dashed border-line text-muted" : s.status === "succeeded" ? "border-emerald-500/40 bg-emerald-500/10" : "border-red-500/40 bg-red-500/10";
          return (
            <div key={n.id} className="flex items-center gap-2">
              <div className="h-px w-5 bg-line" />
              <div className={cn("min-w-[104px] rounded-xl border px-3 py-2 text-center", tone)}>
                <div className="text-[12.5px] font-semibold">{n.label}</div>
                <div className="text-[10.5px] text-muted">
                  {s ? `${s.duration_ms} ms${s.llm_calls ? ` · ${s.llm_calls} LLM` : ""}${jevCount(s) ? ` · ${jevCount(s)} Jev` : ""}` : i === 5 ? "scheduled only" : "skipped"}
                </div>
              </div>
            </div>
          );
        })}
        <div className="h-px w-5 bg-line" />
        <div className="rounded-full border border-line px-3 py-1 text-[11px] text-muted">END</div>
      </div>
    </div>
  );
}

export default function Agents() {
  const [params, setParams] = useSearchParams();
  const { data: meta } = useMeta();
  const { data: runs } = useQuery({ queryKey: ["runs"], queryFn: () => api.get<{ runs: AgentRun[] }>("/runs?limit=40") });
  const { data: usage } = useQuery({ queryKey: ["llm-usage"], queryFn: () => api.get<Usage>("/llm/usage") });
  const pipelineRuns = useMemo(() => runs?.runs ?? [], [runs]);
  const selected = Number(params.get("run")) || pipelineRuns.find((r) => r.graph === "pipeline")?.id;
  const { data: detail, isLoading } = useQuery({
    queryKey: ["run", selected],
    queryFn: () => api.get<RunDetail>(`/runs/${selected}`),
    enabled: !!selected,
  });

  return (
    <div className="space-y-6">
      <PageHeader
        title="Agents & decision trail"
        subtitle="A LangGraph coordinator routes work between specialist agents; every step, LLM call, Jev decision and output is recorded."
      />
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {meta?.agents.map((a) => (
          <Card key={a.id} className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 font-semibold">
                {a.id === "coordinator" ? <GitBranch className="size-4 text-brand" /> : <Bot className="size-4 text-brand" />}
                {a.name}
              </div>
              <ModelBadge model={a.model} />
            </div>
            <p className="mt-2 text-[12.5px] leading-relaxed text-muted">{a.role}</p>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-[340px_1fr]">
        <Card>
          <CardHeader title="Runs" subtitle="Pipeline, scheduled and chat runs" />
          <div className="max-h-[640px] overflow-y-auto p-2">
            {pipelineRuns.map((r) => (
              <button
                key={r.id}
                onClick={() => setParams({ run: String(r.id) })}
                className={cn("flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left hover:bg-surface-2", selected === r.id && "bg-surface-2")}
              >
                {r.status === "succeeded" ? <CircleCheck className="size-4 text-emerald-500" /> : r.status === "failed" ? <CircleX className="size-4 text-red-500" /> : <Timer className="size-4 text-amber-500" />}
                <div className="min-w-0 flex-1">
                  <div className="text-[13px] font-medium">
                    #{r.id} · {r.graph}
                  </div>
                  <div className="truncate text-[11.5px] text-muted">
                    {r.trigger} · {dateTime(r.started_at)} {r.duration_ms != null ? `· ${r.duration_ms} ms` : ""}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </Card>

        <div className="min-w-0 space-y-6">
          {isLoading || !detail ? (
            selected ? <Skeleton className="h-96" /> : <Card><Empty title="No runs yet" /></Card>
          ) : (
            <>
              <Card>
                <CardHeader
                  title={`Run #${detail.run.id} — ${detail.run.graph}`}
                  subtitle={`${detail.run.trigger} · ${dateTime(detail.run.started_at)} · ${detail.run.duration_ms ?? "–"} ms`}
                  action={<Badge tone={detail.run.status === "succeeded" ? "green" : "red"}>{detail.run.status}</Badge>}
                />
                <div className="px-5 pb-5">
                  {detail.run.graph === "pipeline" && <GraphView steps={detail.run.steps ?? []} />}
                  {detail.run.summary?.headline && (
                    <div className="mt-3 rounded-xl bg-surface-2/70 p-3 text-[13px]">
                      <span className="text-muted">Headline: </span>
                      {detail.run.summary.headline}
                    </div>
                  )}
                  {detail.run.error && <div className="mt-3 rounded-xl bg-red-500/10 p-3 text-[13px] text-red-500">{detail.run.error}</div>}
                </div>
              </Card>
              <Card>
                <CardHeader title="Step timeline" subtitle="What each agent did and why" />
                <ol className="relative mx-5 mb-5 mt-3 space-y-5 border-l border-line pl-6">
                  {detail.run.steps?.map((s) => (
                    <li key={s.id}>
                      <span className={cn("absolute -left-[7px] mt-1 size-3.5 rounded-full border-2 border-surface", s.status === "succeeded" ? "bg-emerald-500" : "bg-red-500")} />
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-semibold capitalize">{s.agent}</span>
                        <Badge>{s.duration_ms} ms</Badge>
                        {s.llm_calls > 0 && (
                          <Badge tone="violet">
                            <Cpu className="size-3" /> {s.llm_calls} LLM call(s)
                          </Badge>
                        )}
                        {jevCount(s) > 0 && (
                          <Badge tone="cyan" title="System One (Jev) decisions made in this step">
                            <Scale className="size-3" /> {jevCount(s)} Jev decision call(s)
                          </Badge>
                        )}
                      </div>
                      <div className="mt-1 text-[13px] text-muted">{s.summary}</div>
                      {s.details && Object.keys(s.details).length > 0 && (
                        <details className="mt-1.5">
                          <summary className="cursor-pointer text-[12px] text-brand">details</summary>
                          <pre className="mt-1 max-h-64 overflow-auto rounded-xl bg-surface-2/70 p-3 font-mono text-[11px] text-muted">{JSON.stringify(s.details, null, 2)}</pre>
                        </details>
                      )}
                    </li>
                  ))}
                </ol>
              </Card>
              {detail.llm_calls.length > 0 && (
                <Card>
                  <CardHeader title="Model calls in this run" subtitle="Generative (LLM) and decision (Jev) calls, with tokens and cost" />
                  <div className="overflow-x-auto p-5 pt-3">
                    <table className="w-full text-[12.5px]">
                      <thead className="text-left text-muted">
                        <tr>
                          <th className="py-1.5 font-medium">Agent</th>
                          <th className="font-medium">Kind</th>
                          <th className="font-medium">Provider · model</th>
                          <th className="text-right font-medium">Tokens in/out</th>
                          <th className="text-right font-medium">Cost</th>
                          <th className="text-right font-medium">Latency</th>
                          <th className="text-right font-medium">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.llm_calls.map((c, i) => (
                          <tr key={i} className="border-t border-line">
                            <td className="py-1.5 capitalize">{c.agent}</td>
                            <td>
                              {c.kind === "jev" ? (
                                <Badge tone="cyan" title={c.purpose ?? undefined}>
                                  jev{c.purpose ? ` · ${c.purpose.replace(/_/g, " ")}` : ""}
                                </Badge>
                              ) : (
                                <Badge tone="violet">llm</Badge>
                              )}
                            </td>
                            <td>
                              {c.provider} · {c.model}
                            </td>
                            <td className="num text-right">
                              {c.prompt_tokens}/{c.completion_tokens}
                            </td>
                            <td className="num text-right">{c.cost_usd != null ? `$${c.cost_usd.toFixed(6)}` : "–"}</td>
                            <td className="num text-right">{c.cached ? "cache" : `${c.latency_ms} ms`}</td>
                            <td className="text-right">{c.success ? <Badge tone="green">ok</Badge> : <Badge tone="red" title={c.error ?? ""}>failed</Badge>}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              )}
            </>
          )}
          <Card>
            <CardHeader title="LLM usage (all time)" subtitle={`${usage?.total_calls ?? 0} calls · ${(usage?.total_tokens ?? 0).toLocaleString()} tokens — deterministic stages cost nothing`} />
            <div className="overflow-x-auto p-5 pt-3">
              {usage?.items.length ? (
                <table className="w-full text-[12.5px]">
                  <thead className="text-left text-muted">
                    <tr>
                      <th className="py-1.5 font-medium">Agent</th>
                      <th className="font-medium">Model</th>
                      <th className="text-right font-medium">Calls</th>
                      <th className="text-right font-medium">Cache hits</th>
                      <th className="text-right font-medium">Tokens</th>
                      <th className="text-right font-medium">Avg latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {usage.items.map((u, i) => (
                      <tr key={i} className="border-t border-line">
                        <td className="py-1.5 capitalize">{u.agent}</td>
                        <td>
                          {u.provider} · {u.model}
                        </td>
                        <td className="num text-right">{u.calls}</td>
                        <td className="num text-right">{u.cache_hits}</td>
                        <td className="num text-right">{(u.prompt_tokens + u.completion_tokens).toLocaleString()}</td>
                        <td className="num text-right">{u.avg_latency_ms} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="text-[13px] text-muted">No LLM calls yet — every agent is running on its deterministic path.</p>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
