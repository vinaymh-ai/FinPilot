import { useQuery } from "@tanstack/react-query";
import { Brain, Coins, Gauge, PenLine, Scale, ShieldCheck, Split } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Badge, Card, CardHeader, Empty, ModelBadge, PageHeader, ProbabilityBars, Skeleton, Stat, Tabs, type Tone } from "../components/ui";
import { api } from "../lib/api";
import { cn, dateTime, titleCase } from "../lib/format";
import type { DecisionRow, DecisionStats, JevState } from "../lib/types";

const PURPOSES: Record<string, { label: string; what: string }> = {
  merchant_category: { label: "Merchant category", what: "Categorizer: one Choice over the taxonomy per unknown merchant" },
  service_family: { label: "Service family", what: "Categorizer: which kind of subscription — powers overlap detection" },
  chat_route: { label: "Chat routing", what: "Chat: intent, slots and complexity for every message" },
  faithfulness: { label: "Faithfulness guard", what: "Is LLM-written text backed by the facts it was given?" },
  goal_type: { label: "Goal type", what: "Forecaster: is a goal spent on its deadline or kept as savings?" },
  csv_column: { label: "CSV columns", what: "Ingestion: what an unfamiliar statement column holds" },
};

const ACTION_TONE: Record<string, Tone> = {
  accepted: "green",
  deferred: "amber",
  escalated: "violet",
  rejected: "red",
  overridden: "blue",
};

const usd = (x: number | undefined | null, digits = 4) => (x == null ? "–" : `$${x.toFixed(digits)}`);

function DecisionItem({ d }: { d: DecisionRow }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-t border-line px-5 py-3">
      <button onClick={() => setOpen(!open)} className="flex w-full flex-wrap items-center gap-x-3 gap-y-1.5 text-left">
        <span className="min-w-0 flex-1 truncate text-[13px] font-medium" title={d.subject}>
          {d.subject}
        </span>
        <span className="text-[12px] text-muted">{d.question.replace(/_/g, " ")} →</span>
        <Badge tone="cyan">{d.answer.replace(/_/g, " ")}</Badge>
        <span className="num w-12 text-right text-[12px] text-muted" title="Jev confidence (0 = coin flip, 1 = certain)">
          {d.confidence != null ? d.confidence.toFixed(2) : "–"}
        </span>
        <Badge tone={ACTION_TONE[d.action] ?? "neutral"}>{d.action}</Badge>
      </button>
      {open && (
        <div className="mt-3 grid gap-4 rounded-xl bg-surface-2/50 p-3 md:grid-cols-[1fr_220px]">
          <ProbabilityBars probs={d.probabilities} k={5} highlight={d.answer} />
          <div className="space-y-1 text-[11.5px] text-muted">
            <div>
              {d.agent} · {PURPOSES[d.purpose]?.label ?? d.purpose}
            </div>
            <div>{d.cached ? "served from cache (free)" : `${d.latency_ms} ms`}</div>
            <div>{dateTime(d.created_at)}</div>
            {d.run_id && (
              <Link className="text-brand hover:underline" to={`/agents?run=${d.run_id}`}>
                run #{d.run_id} →
              </Link>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function Decisions() {
  const [purpose, setPurpose] = useState<string>("all");
  const { data: stats, isLoading } = useQuery({ queryKey: ["decision-stats"], queryFn: () => api.get<DecisionStats>("/decisions/stats") });
  const { data: jev } = useQuery({ queryKey: ["jev-settings"], queryFn: () => api.get<JevState>("/settings/decisions") });
  const { data: log } = useQuery({
    queryKey: ["decisions", purpose],
    queryFn: () => api.get<{ decisions: DecisionRow[] }>(`/decisions?limit=150${purpose === "all" ? "" : `&purpose=${purpose}`}`),
  });

  const totals = useMemo(() => {
    const rows = stats?.savings.rows ?? [];
    return {
      decisions: Object.values(stats?.decisions ?? {}).reduce((a, d) => a + d.total, 0),
      avoided: rows.reduce((a, r) => a + (r.llm_calls_avoided ?? 0), 0),
    };
  }, [stats]);
  const tabs = [{ value: "all", label: "All" }, ...Object.keys(stats?.decisions ?? {}).map((p) => ({ value: p, label: PURPOSES[p]?.label ?? titleCase(p) }))];
  const jevUse = stats?.by_kind.jev;
  const llmUse = stats?.by_kind.llm;

  return (
    <div className="space-y-6">
      <PageHeader
        title="System One decisions"
        subtitle={
          <>
            Closed-set judgments are made by <b>Jev</b> (TypeSafe) through Mesh <code className="text-[12px]">/v1/evaluate</code> — a typed answer
            with a probability for every option, never text to parse. LLMs only write; Jev decides.
          </>
        }
        actions={stats && <ModelBadge model={stats.decision_model === "off" ? "rules" : stats.decision_model} />}
      />

      {jev && !jev.available && (
        <Card className="border-amber-500/30 bg-amber-500/5 p-4 text-[13px]">
          <b>Jev is off</b> — {jev.enabled ? `set ${jev.key_env} in your .env` : "it is disabled in Settings"}. Every agent keeps working on its
          deterministic fallback (lexicon, regex router, numeric grounding check); decisions appear here once Jev is on.{" "}
          <Link to="/settings" className="text-brand hover:underline">
            Settings →
          </Link>
        </Card>
      )}

      {isLoading || !stats ? (
        <div className="grid gap-4 md:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <Stat label="Decisions made" value={totals.decisions.toLocaleString()} hint={`${jevUse?.calls ?? 0} Jev calls · ${jevUse?.cache_hits ?? 0} from cache`} icon={<Scale className="size-4" />} />
          <Stat label="LLM calls avoided" value={totals.avoided.toLocaleString()} hint="chat answered from tools + merchants not sent to an LLM" icon={<Split className="size-4" />} tone="up" />
          <Stat
            label="Net saving (est.)"
            value={usd(stats.savings.net_savings_usd)}
            hint={`${usd(stats.savings.llm_cost_avoided_usd)} LLM avoided − ${usd(stats.savings.jev_cost_replacing_usd)} Jev`}
            icon={<Coins className="size-4" />}
            tone={stats.savings.net_savings_usd >= 0 ? "up" : "down"}
          />
          <Stat
            label="Spend by kind"
            value={usd((jevUse?.cost_usd ?? 0) + (llmUse?.cost_usd ?? 0))}
            hint={`Jev ${usd(jevUse?.cost_usd)} (${(jevUse?.input_tokens ?? 0).toLocaleString()} tok) · LLM ${usd(llmUse?.cost_usd)}`}
            icon={<Gauge className="size-4" />}
          />
        </div>
      )}

      <div className="grid gap-6 xl:grid-cols-[1fr_380px]">
        <Card>
          <CardHeader title="Where System One decides" subtitle="Only decisions that replaced an LLM call count as savings; new checks are shown as added cost." icon={<Brain className="size-4" />} />
          <div className="overflow-x-auto p-5 pt-3">
            {stats?.savings.rows.length ? (
              <table className="w-full text-[12.5px]">
                <thead className="text-left text-muted">
                  <tr>
                    <th className="py-1.5 font-medium">Decision</th>
                    <th className="font-medium">Outcomes</th>
                    <th className="text-right font-medium">Jev cost</th>
                    <th className="text-right font-medium">LLM avoided</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.savings.rows.map((r) => (
                    <tr key={r.purpose} className="border-t border-line align-top">
                      <td className="py-2 pr-3">
                        <div className="font-medium">{PURPOSES[r.purpose.split(":")[0]]?.label ?? titleCase(r.purpose)}{r.purpose.includes(":") && <span className="text-muted"> · {r.purpose.split(":")[1].replace("_", " ")}</span>}</div>
                        <div className="text-[11.5px] text-muted">{PURPOSES[r.purpose]?.what ?? (r.purpose.endsWith("order_check") ? "Re-asked with options reversed (order-bias check)" : "")}</div>
                      </td>
                      <td className="py-2">
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(r.actions).map(([a, n]) => (
                            <Badge key={a} tone={ACTION_TONE[a] ?? "neutral"}>
                              {n} {a}
                            </Badge>
                          ))}
                          {!Object.keys(r.actions).length && <span className="text-muted">{r.jev_calls} calls</span>}
                        </div>
                      </td>
                      <td className="num py-2 text-right">{usd(r.jev_cost_usd, 5)}</td>
                      <td className="num py-2 text-right">
                        {r.replaces_llm && r.llm_cost_avoided_usd != null ? (
                          <span className="text-emerald-600 dark:text-emerald-400" title={`${r.llm_calls_avoided} × (${r.llm_equivalent_tokens?.input} in + ${r.llm_equivalent_tokens?.output} out tokens)`}>
                            {usd(r.llm_cost_avoided_usd, 5)}
                          </span>
                        ) : (
                          <span className="text-muted">{r.replaces_llm ? "—" : "new check"}</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <Empty title="No decisions yet" icon={<Scale className="size-5" />}>
                Run the agents or ask the chat a question with Jev enabled.
              </Empty>
            )}
            {stats && <p className="mt-3 text-[11.5px] text-muted">{stats.savings.assumptions.note}</p>}
          </div>
        </Card>

        <Card>
          <CardHeader title="Confidence policy" subtitle="Thresholds are data — tune them with run_eval --jev" icon={<ShieldCheck className="size-4" />} />
          <div className="space-y-3 p-5 pt-3 text-[13px]">
            {jev ? (
              <>
                {[
                  ["Act on a Choice", jev.accept_confidence, "below → keep the rules answer (deferred)"],
                  ["Answer chat without an LLM", jev.route_confidence, "below → escalate to the LLM tool loop"],
                  ["Show LLM-written text", jev.guard_confidence, "P(supported) below → template text"],
                ].map(([label, v, note]) => (
                  <div key={label as string}>
                    <div className="flex items-center justify-between">
                      <span>{label}</span>
                      <span className="num font-semibold">≥ {(v as number).toFixed(2)}</span>
                    </div>
                    <div className="text-[11.5px] text-muted">{note}</div>
                  </div>
                ))}
                <div className="flex flex-wrap gap-1.5 pt-1">
                  <Badge tone={jev.order_check ? "green" : "neutral"}>order-bias check {jev.order_check ? "on" : "off"}</Badge>
                  <Badge tone={jev.escalate_to_llm ? "violet" : "neutral"}>LLM cascade {jev.escalate_to_llm ? "on" : "off"}</Badge>
                </div>
                <p className="flex gap-2 rounded-xl bg-surface-2/60 p-3 text-[12px] text-muted">
                  <PenLine className="mt-0.5 size-3.5 shrink-0" />
                  Math, dates and amounts never go to Jev — they are computed in code and passed as bands ("small: under ₹500", "in about 5 months").
                </p>
              </>
            ) : (
              <Skeleton className="h-40" />
            )}
          </div>
        </Card>
      </div>

      <Card>
        <CardHeader
          title="Decision log"
          subtitle="Every judgment with its probabilities and what the code did with it"
          action={<Tabs value={purpose} onChange={setPurpose} items={tabs} />}
          className="flex-wrap"
        />
        <div className="mt-3">
          {log?.decisions.length ? (
            log.decisions.map((d) => <DecisionItem key={d.id} d={d} />)
          ) : (
            <div className={cn("p-5")}>
              <Empty title="Nothing logged yet" icon={<Scale className="size-5" />} />
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
