import { useQuery } from "@tanstack/react-query";
import { FlaskConical, Gauge, MessageCircle, Radar, Scale, Tags, TrendingUp } from "lucide-react";
import { SimpleBars } from "../components/charts";
import { Badge, Card, CardHeader, Empty, PageHeader, Skeleton, Stat } from "../components/ui";
import { api } from "../lib/api";
import { titleCase } from "../lib/format";

const pctv = (x: number | null | undefined, d = 0) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);

function JevSection({ j }: { j: any }) {
  const b = j.baselines;
  const measured = !!j.merchants;
  const rows: { decision: string; det: string; detResult: string; jev: string }[] = [
    {
      decision: `Merchant category (${b.merchants.cases} merchants)`,
      det: "lexicon + keywords",
      detResult: `${pctv(b.merchants.rules_accuracy_all, 1)} · ${pctv(b.merchants.rules_accuracy_unsure, 1)} on the ${b.merchants.rules_unsure} it is unsure about`,
      jev: measured ? `${pctv(j.merchants.accuracy_all, 1)} · ${pctv(j.merchants.jev_accuracy_where_rules_unsure, 1)} where rules are unsure · ECE ${j.merchants.ece}` : "pending",
    },
    {
      decision: `Chat routing (${b.chat.cases} messages)`,
      det: "regex intent router",
      detResult: `${pctv(b.chat.regex_intent_accuracy, 1)} · cannot refuse ${b.chat.blocked_messages} unsafe/off-topic`,
      jev: measured
        ? `${pctv(j.chat.intent_accuracy, 1)} · unsafe recall ${pctv(j.chat.routing_policy.unsafe_recall)} · ${pctv(j.chat.routing_policy.answered_without_llm)} answered without an LLM`
        : "pending",
    },
    {
      decision: `Faithfulness (${b.faithfulness.cases} drafts)`,
      det: "numeric grounding check",
      detResult: `catches ${pctv(b.faithfulness.numeric_check_recall)} of unfaithful drafts`,
      jev: measured ? `numeric + Jev catches ${pctv(j.faithfulness.numeric_plus_jev.recall_unfaithful)}` : "pending",
    },
    {
      decision: `Goal type (${b.goals.cases} goals)`,
      det: "keyword table",
      detResult: `type ${pctv(b.goals.keyword_type_accuracy)} · spend/keep ${pctv(b.goals.keyword_spend_accuracy)}`,
      jev: measured ? `type ${pctv(j.goals.type_accuracy)} · spend/keep ${pctv(j.goals.spend_accuracy)}` : "pending",
    },
    {
      decision: `CSV columns (${b.csv_columns.cases} headers)`,
      det: "synonym list",
      detResult: `${b.csv_columns.unmapped} unrecognised`,
      jev: measured ? pctv(j.csv_columns.role_accuracy) : "pending",
    },
  ];
  return (
    <Card>
      <CardHeader
        title="System One decisions (Jev)"
        subtitle={<>Hand-labelled sets in <code>evaluation/jev_cases.py</code> · thresholds picked on a validation split, reported on a held-out test split</>}
        icon={<Scale className="size-4" />}
        action={<Badge tone={measured ? "green" : "amber"}>{j.status}</Badge>}
      />
      <div className="overflow-x-auto p-5 pt-3">
        <table className="w-full text-[12.5px]">
          <thead className="text-left text-muted">
            <tr>
              <th className="py-1.5 font-medium">Decision</th>
              <th className="font-medium">Deterministic path (measured)</th>
              <th className="font-medium">Jev</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.decision} className="border-t border-line align-top">
                <td className="py-2 pr-3 font-medium">{r.decision}</td>
                <td className="py-2 pr-3">
                  <div>{r.detResult}</div>
                  <div className="text-[11px] text-muted">{r.det}</div>
                </td>
                <td className="py-2">{r.jev === "pending" ? <Badge tone="amber">pending a live run</Badge> : r.jev}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-4 text-[12px] font-medium">Cost per 1,000 decisions (from the real request bodies)</div>
        <div className="mt-1 flex flex-wrap gap-1.5">
          {Object.entries(j.request_estimates).map(([k, v]: [string, any]) => (
            <Badge key={k} tone="cyan" title={`~${v.estimated_input_tokens} input tokens per request`}>
              {k.replace(/_/g, " ")}: Jev ${v.jev_usd_per_1k_estimated.toFixed(3)}
              {v.llm_usd_per_1k_estimated != null && ` vs LLM $${v.llm_usd_per_1k_estimated.toFixed(3)}`}
            </Badge>
          ))}
        </div>
        {!measured && <p className="mt-3 text-[12px] text-muted">{j.how_to_run}</p>}
      </div>
    </Card>
  );
}

export default function Evaluation() {
  const { data, isLoading } = useQuery({ queryKey: ["evaluation"], queryFn: () => api.get<any>("/evaluation") });
  if (isLoading) return <Skeleton className="h-96" />;
  if (!data?.available)
    return (
      <Card>
        <Empty icon={<Gauge className="size-6" />} title="No evaluation results yet">
          Run <code>python -m evaluation.run_eval</code> in the repo.
        </Empty>
      </Card>
    );
  const c = data.categorization;
  const det = data.detectors;
  const fc = data.forecast;
  const ablation = Object.entries(c.ablation).map(([k, v]: [string, any]) => ({
    x: titleCase(k.replace(/^\d_/, "").replace(" (deterministic)", "")),
    "All merchants": +(v.accuracy * 100).toFixed(1),
    "Long-tail merchants": v.accuracy_long_tail != null ? +(v.accuracy_long_tail * 100).toFixed(1) : 0,
  }));
  const best: any = Object.values(c.ablation).at(-1);
  const horizons = Object.entries(fc.horizons).map(([h, r]: [string, any]) => ({
    x: `${h} days`,
    FinPilot: r.finpilot_p50,
    "Naive (last month)": r.naive_last_30d,
    "3-month mean": r.mean_last_90d,
  }));
  const h30 = fc.horizons["30"];
  const perCat = Object.entries(best.per_category ?? {})
    .map(([k, v]: [string, any]) => ({ k, ...v }))
    .sort((a, b) => b.support - a.support);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Self-evaluation"
        subtitle={
          <>
            Measured on {c.transactions.toLocaleString()} labelled synthetic transactions ({c.datasets} statements, 3 personas) ·
            generated {data.generated_at} · mode <Badge>{data.mode}</Badge>
          </>
        }
      />
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Categorization accuracy" value={`${(best.accuracy * 100).toFixed(1)}%`} hint={`macro-F1 ${best.macro_f1.toFixed(3)} · baseline ${(c.baseline_majority_class.accuracy * 100).toFixed(0)}%`} icon={<Tags className="size-4" />} />
        <Stat label="Recurring detection F1" value={det.recurring_series.f1.toFixed(3)} hint={`P ${det.recurring_series.precision.toFixed(2)} · R ${det.recurring_series.recall.toFixed(2)}`} icon={<Radar className="size-4" />} />
        <Stat label="Anomaly detection F1" value={det.anomalies.f1.toFixed(3)} hint={`P ${det.anomalies.precision.toFixed(2)} · R ${det.anomalies.recall.toFixed(2)}`} icon={<Radar className="size-4" />} />
        <Stat label="30-day forecast error" value={`${h30.finpilot_p50.toFixed(1)}%`} hint={`of monthly income · naive ${h30.naive_last_30d.toFixed(0)}%`} icon={<TrendingUp className="size-4" />} tone="up" />
      </div>

      {data.jev && <JevSection j={data.jev} />}

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader title="Categorization — what each stage adds" subtitle={`${(c.long_tail_share * 100).toFixed(0)}% of rows come from local merchants absent from the lexicon`} icon={<Tags className="size-4" />} />
          <div className="px-2 pb-4">
            <SimpleBars
              data={ablation}
              keys={[
                { key: "All merchants", name: "All merchants (%)", color: "#10b981" },
                { key: "Long-tail merchants", name: "Long-tail merchants (%)", color: "#818cf8" },
              ]}
              height={260}
            />
          </div>
          <p className="px-5 pb-5 text-[12.5px] text-muted">{data.llm_note}</p>
        </Card>
        <Card>
          <CardHeader title="Forecast backtest — error vs baselines" subtitle={`${fc.origins} rolling origins · mean |pred − actual| net cash-flow as % of monthly income (lower is better)`} icon={<TrendingUp className="size-4" />} />
          <div className="px-2 pb-4">
            <SimpleBars
              data={horizons}
              keys={[
                { key: "FinPilot", name: "FinPilot P50", color: "#10b981" },
                { key: "Naive (last month)", name: "Naive: last month", color: "#f59e0b" },
                { key: "3-month mean", name: "3-month mean", color: "#94a3b8" },
              ]}
              height={260}
            />
          </div>
          <div className="grid grid-cols-3 gap-3 px-5 pb-5 text-[12.5px]">
            {Object.entries(fc.horizons).map(([h, r]: [string, any]) => (
              <div key={h} className="rounded-xl bg-surface-2/70 p-2.5">
                <div className="text-muted">{h}-day P10–P90 coverage</div>
                <div className="num font-semibold">{Math.round(r.p10_p90_coverage * 100)}%</div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader title="Per-category quality" subtitle="Best pipeline stage" />
          <div className="overflow-x-auto p-5 pt-3">
            <table className="w-full text-[12.5px]">
              <thead className="text-left text-muted">
                <tr>
                  <th className="py-1.5 font-medium">Category</th>
                  <th className="text-right font-medium">Support</th>
                  <th className="text-right font-medium">Precision</th>
                  <th className="text-right font-medium">Recall</th>
                  <th className="text-right font-medium">F1</th>
                </tr>
              </thead>
              <tbody>
                {perCat.map((r: any) => (
                  <tr key={r.k} className="border-t border-line">
                    <td className="py-1.5">{titleCase(r.k)}</td>
                    <td className="num text-right">{r.support}</td>
                    <td className="num text-right">{r.precision.toFixed(2)}</td>
                    <td className="num text-right">{r.recall.toFixed(2)}</td>
                    <td className="num text-right font-medium">{r.f1.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <div className="space-y-6">
          <Card className="p-5">
            <div className="flex items-center gap-2 font-semibold">
              <FlaskConical className="size-4 text-brand" /> What-if consistency
            </div>
            <div className="mt-2 text-[13px] text-muted">
              {data.simulation.scenarios} scenarios · exact-delta rate{" "}
              <b className="text-ink">{Math.round(data.simulation.exact_delta_rate * 100)}%</b> · verdict monotonic in{" "}
              <b className="text-ink">{data.simulation.verdict_monotonic_datasets}</b> datasets.
            </div>
            <p className="mt-2 text-[12px] text-muted">{data.simulation.note}</p>
          </Card>
          <Card className="p-5">
            <div className="flex items-center gap-2 font-semibold">
              <MessageCircle className="size-4 text-brand" /> Conversational routing
            </div>
            <div className="mt-2 text-[13px] text-muted">
              {data.chat.cases} labelled questions → right tool sequence in{" "}
              <b className="text-ink">{Math.round(data.chat.accuracy * 100)}%</b> ({data.chat.mode}).
            </div>
          </Card>
          <Card className="p-5">
            <div className="font-semibold">Anomaly recall by type</div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {Object.entries(det.anomalies.recall_by_type).map(([k, v]: [string, any]) => (
                <Badge key={k} tone="green">
                  {titleCase(k)} {Math.round(v * 100)}%
                </Badge>
              ))}
            </div>
            <div className="mt-2 text-[12px] text-muted">
              Unused-membership detection: {det.unused_membership_detection.exact_match}/{det.unused_membership_detection.datasets} datasets exactly right.
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
