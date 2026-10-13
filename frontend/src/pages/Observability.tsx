import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { Activity, CircleAlert, Cpu, FileText, Radar, RefreshCw, Search, Timer, Wallet } from "lucide-react";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { CallDrawer, JsonBlock, type CallRow } from "../components/CallDrawer";
import { Badge, Button, Card, CardHeader, Empty, Input, PageHeader, Select, Skeleton, Stat, Tabs, Toggle } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { cn, dateTime } from "../lib/format";

type Status = {
  langsmith: { enabled: boolean; requested: boolean; key_set: boolean; project: string; url: string | null };
  trace_payloads: boolean;
  trace_max_chars: number;
  trace_retention_days: number;
  log_level: string;
  log_file_enabled: boolean;
  log_file?: string;
  log_file_bytes?: number;
  is_admin: boolean;
};

type KindStats = { calls: number; failures: number; cache_hits: number; input_tokens: number; output_tokens: number; cost_usd: number; avg_latency_ms: number };
type Summary = {
  hours: number;
  by_kind: Record<string, KindStats>;
  by_agent: { agent: string; kind: string; calls: number; failures: number }[];
  latency_ms: { p50: number | null; p95: number | null; p99: number | null };
  recent_errors: { id: number; created_at: string | null; agent: string; kind: string; error: string }[];
};

type LogLine = { ts: string; level: string; logger: string; msg: string; request_id?: string; user_id?: number; run_id?: number; event?: string; exc?: string; [k: string]: unknown };

const WINDOWS = [
  { value: "1", label: "1 hour" },
  { value: "24", label: "24 hours" },
  { value: "168", label: "7 days" },
  { value: "720", label: "30 days" },
];

const LEVEL_TONE: Record<string, "neutral" | "blue" | "amber" | "red"> = { DEBUG: "neutral", INFO: "blue", WARNING: "amber", ERROR: "red", CRITICAL: "red" };

function kb(n: number | undefined) {
  if (!n) return "0 KB";
  return n > 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.ceil(n / 1024)} KB`;
}

export default function ObservabilityPage() {
  const { user } = useAuth();
  const isAdmin = !!user?.is_admin;
  const [params, setParams] = useSearchParams();
  const tab = (params.get("tab") as "calls" | "logs") ?? "calls";
  const [hours, setHours] = useState("24");
  const [allUsers, setAllUsers] = useState(false);
  const [openCall, setOpenCall] = useState<number | null>(params.get("call") ? Number(params.get("call")) : null);
  const [logRequestId, setLogRequestId] = useState(params.get("request_id") ?? "");

  const setTab = (t: "calls" | "logs") => setParams((p) => (p.set("tab", t), p), { replace: true });
  const status = useQuery({ queryKey: ["observability", "status"], queryFn: () => api.get<Status>("/observability/status") });
  const summary = useQuery({
    queryKey: ["observability", "summary", hours, allUsers],
    queryFn: () => api.get<Summary>(`/observability/summary?hours=${hours}&all_users=${allUsers}`),
  });

  const showLogs = (rid: string) => {
    setOpenCall(null);
    setLogRequestId(rid);
    setTab("logs");
  };

  const s = summary.data;
  const llm = s?.by_kind.llm;
  const jev = s?.by_kind.jev;
  const calls = (llm?.calls ?? 0) + (jev?.calls ?? 0);
  const failures = (llm?.failures ?? 0) + (jev?.failures ?? 0);
  const cost = (llm?.cost_usd ?? 0) + (jev?.cost_usd ?? 0);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Observability"
        subtitle="Every model call with its raw prompt and response, structured server logs, and tracing status — for debugging what the agents actually did."
        actions={
          <>
            {isAdmin && (
              <label className="flex items-center gap-2 text-[13px] text-muted">
                <Toggle checked={allUsers} onChange={setAllUsers} label="All users" /> All users
              </label>
            )}
            <Select value={hours} onChange={(e) => setHours(e.target.value)} className="h-9 w-32 text-[13px]">
              {WINDOWS.map((w) => (
                <option key={w.value} value={w.value}>
                  {w.label}
                </option>
              ))}
            </Select>
          </>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {summary.isLoading ? (
          [0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-28" />)
        ) : (
          <>
            <Stat label="Model calls" value={calls.toLocaleString()} icon={<Cpu className="size-4" />} hint={`${llm?.calls ?? 0} LLM · ${jev?.calls ?? 0} Jev · ${(llm?.cache_hits ?? 0) + (jev?.cache_hits ?? 0)} cached`} />
            <Stat
              label="Failures"
              value={failures.toLocaleString()}
              icon={<CircleAlert className="size-4" />}
              hint={calls ? `${((failures / calls) * 100).toFixed(1)}% of calls` : "no calls in this window"}
            />
            <Stat label="Latency p50 / p95" value={s?.latency_ms.p50 != null ? `${s.latency_ms.p50} / ${s.latency_ms.p95} ms` : "–"} icon={<Timer className="size-4" />} hint="live calls (cache hits excluded)" />
            <Stat
              label="Tokens · cost"
              value={`${(((llm?.input_tokens ?? 0) + (jev?.input_tokens ?? 0) + (llm?.output_tokens ?? 0)) / 1000).toFixed(1)}k`}
              icon={<Wallet className="size-4" />}
              hint={`≈ $${cost.toFixed(4)} at configured prices`}
            />
          </>
        )}
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
        <div className="min-w-0 space-y-4">
          <Tabs
            value={isAdmin ? tab : "calls"}
            onChange={setTab}
            items={[
              { value: "calls", label: "Model calls" },
              ...(isAdmin ? [{ value: "logs" as const, label: "Server logs" }] : []),
            ]}
          />
          {tab === "logs" && isAdmin ? (
            <LogsPanel requestId={logRequestId} setRequestId={setLogRequestId} />
          ) : (
            <CallsPanel allUsers={allUsers} onOpen={setOpenCall} />
          )}
        </div>
        <div className="space-y-6">
          <StatusCard status={status.data} />
          {s && s.recent_errors.length > 0 && (
            <Card>
              <CardHeader title="Recent failures" icon={<CircleAlert className="size-4" />} />
              <ul className="space-y-2 p-5 pt-3">
                {s.recent_errors.map((e) => (
                  <li key={e.id}>
                    <button onClick={() => setOpenCall(e.id)} className="w-full rounded-xl border border-line p-2.5 text-left hover:border-brand/40">
                      <div className="flex items-center justify-between text-[12px] text-muted">
                        <span>
                          {e.kind} · {e.agent}
                        </span>
                        <span>{e.created_at ? dateTime(e.created_at) : ""}</span>
                      </div>
                      <div className="mt-1 line-clamp-2 font-mono text-[11.5px] text-red-600 dark:text-red-400">{e.error}</div>
                    </button>
                  </li>
                ))}
              </ul>
            </Card>
          )}
          {s && s.by_agent.length > 0 && (
            <Card>
              <CardHeader title="Calls by agent" icon={<Activity className="size-4" />} />
              <table className="m-5 mt-3 w-[calc(100%-2.5rem)] text-[12.5px]">
                <tbody>
                  {s.by_agent.map((a) => (
                    <tr key={`${a.agent}-${a.kind}`} className="border-t border-line first:border-t-0">
                      <td className="py-1.5 capitalize">{a.agent}</td>
                      <td>
                        <Badge tone={a.kind === "jev" ? "cyan" : "violet"}>{a.kind}</Badge>
                      </td>
                      <td className="num text-right">{a.calls}</td>
                      <td className="num w-14 text-right text-red-500">{a.failures || ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>
      </div>

      <CallDrawer callId={openCall} onClose={() => setOpenCall(null)} onShowLogs={isAdmin ? showLogs : undefined} />
    </div>
  );
}

function StatusCard({ status }: { status: Status | undefined }) {
  if (!status) return <Skeleton className="h-64" />;
  const ls = status.langsmith;
  return (
    <Card>
      <CardHeader title="Tracing & logs" icon={<Radar className="size-4" />} />
      <div className="space-y-4 p-5 pt-3 text-[12.5px]">
        <div>
          <div className="flex items-center justify-between">
            <span className="font-medium">LangSmith</span>
            {ls.enabled ? <Badge tone="green">on · {ls.project}</Badge> : <Badge>{ls.requested && !ls.key_set ? "no key" : "off"}</Badge>}
          </div>
          {ls.enabled ? (
            <a className="mt-1 block text-brand hover:underline" href={ls.url ?? "https://smith.langchain.com/"} target="_blank" rel="noreferrer">
              Open traces in LangSmith →
            </a>
          ) : (
            <p className="mt-1 text-muted">
              Optional. Set <code className="font-mono">LANGSMITH_TRACING=true</code> and <code className="font-mono">LANGSMITH_API_KEY</code> to trace the agent graph,
              every model call and Jev decision as nested spans. Off by default — traces contain financial data.
            </p>
          )}
        </div>
        <div className="border-t border-line pt-3">
          <div className="flex items-center justify-between">
            <span className="font-medium">Raw payload capture</span>
            <Badge tone={status.trace_payloads ? "green" : "neutral"}>{status.trace_payloads ? "on" : "off"}</Badge>
          </div>
          <p className="mt-1 text-muted">
            Prompts and responses are stored locally (up to {status.trace_max_chars.toLocaleString()} chars per field) and purged after {status.trace_retention_days} days.
          </p>
        </div>
        <div className="border-t border-line pt-3">
          <div className="flex items-center justify-between">
            <span className="font-medium">Log file</span>
            <Badge tone={status.log_file_enabled ? "green" : "neutral"}>{status.log_file_enabled ? status.log_level : "off"}</Badge>
          </div>
          {status.log_file && (
            <p className="mt-1 break-all font-mono text-[11px] text-muted">
              {status.log_file} · {kb(status.log_file_bytes)}
            </p>
          )}
          <p className="mt-1 text-muted">
            Every response carries an <code className="font-mono">X-Request-ID</code>; it links API errors, log lines and model calls.
          </p>
        </div>
      </div>
    </Card>
  );
}

function CallsPanel({ allUsers, onOpen }: { allUsers: boolean; onOpen: (id: number) => void }) {
  const [kind, setKind] = useState("");
  const [state, setState] = useState("");
  const [q, setQ] = useState("");
  const [query, setQuery] = useState("");
  const calls = useInfiniteQuery({
    queryKey: ["observability", "calls", kind, state, query, allUsers],
    initialPageParam: null as number | null,
    queryFn: ({ pageParam }) => {
      const p = new URLSearchParams({ limit: "50", all_users: String(allUsers) });
      if (kind) p.set("kind", kind);
      if (state) p.set("success", state === "ok" ? "true" : "false");
      if (query) p.set("q", query);
      if (pageParam) p.set("before_id", String(pageParam));
      return api.get<{ calls: CallRow[]; next_before_id: number | null }>(`/observability/calls?${p}`);
    },
    getNextPageParam: (last) => last.next_before_id,
  });
  const rows = calls.data?.pages.flatMap((p) => p.calls) ?? [];
  return (
    <Card>
      <form
        className="flex flex-wrap items-center gap-2 border-b border-line p-4"
        onSubmit={(e) => {
          e.preventDefault();
          setQuery(q.trim());
        }}
      >
        <Select value={kind} onChange={(e) => setKind(e.target.value)} className="h-9 w-32 text-[13px]">
          <option value="">All kinds</option>
          <option value="llm">LLM</option>
          <option value="jev">Jev</option>
        </Select>
        <Select value={state} onChange={(e) => setState(e.target.value)} className="h-9 w-32 text-[13px]">
          <option value="">Any status</option>
          <option value="ok">Succeeded</option>
          <option value="failed">Failed</option>
        </Select>
        <div className="relative min-w-[200px] flex-1">
          <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted" />
          <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Model, purpose, error text or request id" className="h-9 pl-9 text-[13px]" />
        </div>
        <Button size="sm" type="submit">
          Search
        </Button>
        <Button size="sm" variant="ghost" type="button" onClick={() => calls.refetch()} aria-label="Refresh">
          <RefreshCw className={cn("size-4", calls.isFetching && "animate-spin")} />
        </Button>
      </form>
      {calls.isLoading ? (
        <Skeleton className="m-4 h-64" />
      ) : rows.length === 0 ? (
        <Empty icon={<Cpu className="size-5" />} title="No model calls yet">
          Run the agents or ask FinPilot a question — each LLM and Jev call shows up here with its raw prompt and response.
        </Empty>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-[12.5px]">
            <thead className="text-left text-muted">
              <tr className="border-b border-line">
                <th className="px-4 py-2 font-medium">When</th>
                <th className="font-medium">Kind</th>
                <th className="font-medium">Agent · purpose</th>
                <th className="font-medium">Model</th>
                <th className="text-right font-medium">Tokens</th>
                <th className="text-right font-medium">Latency</th>
                <th className="px-4 text-right font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((c) => (
                <tr key={c.id} onClick={() => onOpen(c.id)} className="cursor-pointer border-t border-line hover:bg-surface-2/60">
                  <td className="whitespace-nowrap px-4 py-2 text-muted">{c.created_at ? dateTime(c.created_at) : "–"}</td>
                  <td>{c.kind === "jev" ? <Badge tone="cyan">jev</Badge> : <Badge tone="violet">llm</Badge>}</td>
                  <td className="max-w-[220px] truncate">
                    <span className="capitalize">{c.agent}</span>
                    {c.purpose && <span className="text-muted"> · {c.purpose.replace(/_/g, " ")}</span>}
                  </td>
                  <td className="max-w-[220px] truncate font-mono text-[11.5px]">{c.model}</td>
                  <td className="num text-right">
                    {c.prompt_tokens ?? 0}/{c.completion_tokens ?? 0}
                  </td>
                  <td className="num text-right">{c.cached ? <Badge>cache</Badge> : `${c.latency_ms ?? 0} ms`}</td>
                  <td className="px-4 text-right">{c.success ? <Badge tone="green">ok</Badge> : <Badge tone="red">failed</Badge>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {calls.hasNextPage && (
            <div className="flex justify-center border-t border-line p-3">
              <Button size="sm" loading={calls.isFetchingNextPage} onClick={() => calls.fetchNextPage()}>
                Load older calls
              </Button>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

function LogsPanel({ requestId, setRequestId }: { requestId: string; setRequestId: (v: string) => void }) {
  const [level, setLevel] = useState("INFO");
  const [q, setQ] = useState("");
  const [query, setQuery] = useState("");
  const [rid, setRid] = useState(requestId);
  const [live, setLive] = useState(false);
  const [open, setOpen] = useState<number | null>(null);
  useEffect(() => setRid(requestId), [requestId]);
  const logs = useQuery({
    queryKey: ["observability", "logs", level, query, requestId],
    queryFn: () => {
      const p = new URLSearchParams({ level, limit: "300" });
      if (query) p.set("q", query);
      if (requestId) p.set("request_id", requestId);
      return api.get<{ enabled: boolean; file: string | null; lines: LogLine[]; truncated: boolean }>(`/observability/logs?${p}`);
    },
    refetchInterval: live ? 4000 : false,
  });
  const lines = logs.data?.lines ?? [];
  return (
    <Card>
      <form
        className="flex flex-wrap items-center gap-2 border-b border-line p-4"
        onSubmit={(e) => {
          e.preventDefault();
          setQuery(q.trim());
          setRequestId(rid.trim());
        }}
      >
        <Select value={level} onChange={(e) => setLevel(e.target.value)} className="h-9 w-32 text-[13px]">
          {["DEBUG", "INFO", "WARNING", "ERROR"].map((l) => (
            <option key={l} value={l}>
              {l === "DEBUG" ? "All levels" : `${l}+`}
            </option>
          ))}
        </Select>
        <div className="relative min-w-[180px] flex-1">
          <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted" />
          <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search message or fields" className="h-9 pl-9 text-[13px]" />
        </div>
        <Input value={rid} onChange={(e) => setRid(e.target.value)} placeholder="Request id" className="h-9 w-44 font-mono text-[12px]" />
        <Button size="sm" type="submit">
          Filter
        </Button>
        <label className="ml-auto flex items-center gap-2 text-[12.5px] text-muted">
          <Toggle checked={live} onChange={setLive} label="Live tail" /> Live
        </label>
      </form>
      {logs.isLoading ? (
        <Skeleton className="m-4 h-64" />
      ) : !logs.data?.enabled ? (
        <Empty icon={<FileText className="size-5" />} title="Log file is off">
          Set <code className="font-mono">LOG_FILE</code> (default <code className="font-mono">logs/finpilot.log</code>) to keep searchable JSON logs.
        </Empty>
      ) : lines.length === 0 ? (
        <Empty icon={<FileText className="size-5" />} title="No matching log lines" />
      ) : (
        <ul className="divide-y divide-line font-mono text-[11.5px]">
          {lines.map((l, i) => (
            <li key={i}>
              <button onClick={() => setOpen(open === i ? null : i)} className="flex w-full items-start gap-2 px-4 py-1.5 text-left hover:bg-surface-2/60">
                <span className="w-[86px] shrink-0 text-muted">{l.ts.slice(11, 23)}</span>
                <Badge tone={LEVEL_TONE[l.level] ?? "neutral"} className="w-[62px] justify-center">
                  {l.level}
                </Badge>
                <span className="hidden w-40 shrink-0 truncate text-muted md:inline">{l.logger}</span>
                <span className={cn("min-w-0 flex-1 break-words", l.level === "ERROR" && "text-red-600 dark:text-red-400")}>{l.msg}</span>
                {l.user_id != null && <span className="hidden shrink-0 text-muted sm:inline">u{l.user_id}</span>}
              </button>
              {open === i && (
                <div className="px-4 pb-3">
                  {l.request_id && (
                    <button className="mb-2 text-[11.5px] text-brand hover:underline" onClick={() => setRequestId(l.request_id!)}>
                      Show everything from request {l.request_id}
                    </button>
                  )}
                  <JsonBlock title="Log record" value={l} maxHeight="20rem" />
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
      {logs.data?.truncated && <p className="border-t border-line px-4 py-2 text-[12px] text-muted">Showing the newest {lines.length} lines — narrow the filter to see more.</p>}
    </Card>
  );
}
