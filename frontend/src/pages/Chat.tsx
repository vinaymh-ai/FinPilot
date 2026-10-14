import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot, ChevronDown, Eraser, Scale, Send, ShieldCheck, User, Wrench } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { toast } from "sonner";
import { Artifacts } from "../components/ChatArtifacts";
import { Badge, Button, Card, ModelBadge, ProbabilityBars } from "../components/ui";
import { api } from "../lib/api";
import { cn } from "../lib/format";
import type { ChatArtifact, ChatMessage, ChatRoute, ChatTrace } from "../lib/types";

const SESSION = "web";
const SUGGESTIONS: { group: string; items: string[] }[] = [
  {
    group: "Ask anything about your transactions",
    items: [
      "How many transactions have I made in total?",
      "Show all my shopping transactions",
      "Spending by category this year",
      "Food delivery this month vs last month",
      "Biggest credit this year",
      "UPI vs card spending",
    ],
  },
  {
    group: "Patterns, plans & advice",
    items: [
      "Why did my spending go up this month compared to usual?",
      "Anything unusual in my transactions?",
      "What are my subscriptions?",
      "Can I afford a ₹45,000 laptop in December?",
      "Am I on track for my emergency fund?",
      "How can I save money?",
    ],
  },
];
const AGENT_LABEL: Record<string, string> = {
  guardrails: "Guardrails",
  capabilities: "Capabilities",
  retriever: "Retriever",
  sql_analyst: "SQL Analyst",
  planner: "Planner",
  router: "Router",
  tool_loop: "Tool agent",
  spending_tools: "Spending tools",
  analyzer: "Analyzer",
  forecaster: "Forecaster",
  advisor: "Advisor",
  composer: "Composer",
  verifier: "Verifier",
};

function PipelineView({ pipeline, guard }: { pipeline: string[]; guard?: ChatMessage["guard"] }) {
  if (!pipeline?.length) return null;
  return (
    <div className="mt-2 flex flex-wrap items-center gap-1 text-[11.5px] text-muted">
      <span className="mr-0.5">Agents:</span>
      {pipeline.map((p, i) => (
        <span key={i} className="inline-flex items-center gap-1">
          {i > 0 && <span className="opacity-50">→</span>}
          <Badge tone={p === "verifier" ? (guard?.ok ? "green" : "neutral") : p === "sql_analyst" ? "violet" : "neutral"}>
            {p === "verifier" && guard?.ok && <ShieldCheck className="size-3" />}
            {AGENT_LABEL[p] ?? p.replace(/_/g, " ")}
          </Badge>
        </span>
      ))}
    </div>
  );
}

const ROUTE_ACTION: Record<string, { text: string; tone: "green" | "violet" | "red" | "neutral" }> = {
  answered_by_rules: { text: "answered from tools — no LLM", tone: "green" },
  escalated_to_llm: { text: "LLM answer · verified facts", tone: "violet" },
  refused: { text: "refused — no LLM", tone: "red" },
  fallback_rules: { text: "verified fallback", tone: "neutral" },
};

function RouteView({ route }: { route: ChatRoute }) {
  const [open, setOpen] = useState(false);
  const act = ROUTE_ACTION[route.action] ?? ROUTE_ACTION.fallback_rules;
  const slots = Object.entries(route.slots).filter(([, v]) => v);
  return (
    <div className="mt-2">
      <button onClick={() => setOpen(!open)} className="inline-flex flex-wrap items-center gap-1.5 text-[12px] text-muted hover:text-ink">
        <Scale className="size-3.5 text-cyan-600 dark:text-cyan-400" />
        Jev routed as <b className="font-medium text-ink">{route.intent.replace(/_/g, " ")}</b>
        <span className="num">({route.confidence.toFixed(2)})</span>→
        <Badge tone={act.tone}>{act.text}</Badge>
        <ChevronDown className={cn("size-3.5 transition", open && "rotate-180")} />
      </button>
      {open && (
        <div className="mt-2 space-y-2 rounded-xl border border-line bg-surface-2/60 p-3 text-[11.5px]">
          <ProbabilityBars probs={route.probabilities} highlight={route.intent} />
          <div className="flex flex-wrap gap-1.5 text-muted">
            {slots.map(([k, v]) => (
              <Badge key={k}>
                {k.replace("_", " ")}: {String(v).replace(/_/g, " ")}
              </Badge>
            ))}
            <Badge title="Score question, lowest first">complexity {route.complexity.toFixed(1)} · {route.complexity_label.toLowerCase()}</Badge>
            {route.order_checked && <Badge tone="amber">re-asked with options reversed</Badge>}
          </div>
          <div className="text-muted">
            {route.model} · {route.cached ? "cache hit (free)" : `${route.latency_ms} ms · ${route.input_tokens} input tokens · $${route.cost_usd.toFixed(6)}`}
            {route.llm_model && ` · answered by ${route.llm_model}`}
          </div>
        </div>
      )}
    </div>
  );
}

function TraceView({ trace }: { trace: ChatTrace[] }) {
  const [open, setOpen] = useState(false);
  if (!trace.length) return null;
  return (
    <div className="mt-2">
      <button onClick={() => setOpen(!open)} className="inline-flex items-center gap-1.5 text-[12px] text-muted hover:text-ink">
        <Wrench className="size-3.5" />
        Used {trace.map((t) => t.tool).join(" → ")}
        <ChevronDown className={cn("size-3.5 transition", open && "rotate-180")} />
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {trace.map((t, i) => (
            <div key={i} className="rounded-xl border border-line bg-surface-2/60 p-2.5 text-[11.5px]">
              {t.tool === "query_ledger" && typeof t.args?.sql === "string" ? (
                <div>
                  <div className="font-mono font-medium text-brand">query_ledger · SQL{t.args?.purpose ? ` (${String(t.args.purpose)})` : ""}</div>
                  <pre className="mt-1 whitespace-pre-wrap break-words rounded-lg bg-surface px-2 py-1.5 font-mono text-[11px] text-ink">{String(t.args.sql)}</pre>
                </div>
              ) : (
                <div className="font-mono font-medium text-brand">
                  {t.tool}({Object.keys(t.args ?? {}).length ? JSON.stringify(t.args) : ""})
                </div>
              )}
              <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-all font-mono text-muted">
                {JSON.stringify(t.result, null, 1)}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Chat() {
  const qc = useQueryClient();
  const { data } = useQuery({
    queryKey: ["chat", SESSION],
    queryFn: () => api.get<{ messages: ChatMessage[] }>(`/chat/${SESSION}`),
  });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [text, setText] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (data) setMessages(data.messages);
  }, [data]);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = useMutation({
    mutationFn: (message: string) =>
      api.post<{
        answer: string;
        trace: ChatTrace[];
        model: string;
        route: ChatRoute | null;
        artifacts?: ChatArtifact[];
        pipeline?: string[];
        guard?: ChatMessage["guard"];
      }>("/chat", { message, session_id: SESSION }),
    onMutate: (message) =>
      setMessages((m) => [...m, { role: "user", content: message }, { role: "assistant", content: "", pending: true }]),
    onSuccess: (r) =>
      setMessages((m) => [
        ...m.filter((x) => !x.pending),
        {
          role: "assistant",
          content: r.answer,
          trace: r.trace,
          model: r.model,
          route: r.route,
          artifacts: r.artifacts ?? [],
          pipeline: r.pipeline ?? [],
          guard: r.guard ?? null,
        },
      ]),
    onError: (e: Error) => {
      setMessages((m) => m.filter((x) => !x.pending));
      toast.error(e.message);
    },
  });
  const clear = useMutation({
    mutationFn: () => api.del(`/chat/${SESSION}`),
    onSuccess: () => {
      setMessages([]);
      qc.invalidateQueries({ queryKey: ["chat", SESSION] });
    },
  });
  const submit = (q?: string) => {
    const msg = (q ?? text).trim();
    if (!msg || send.isPending) return;
    setText("");
    send.mutate(msg);
  };

  return (
    <div className="mx-auto flex h-[calc(100vh-8rem)] max-w-4xl flex-col">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Ask FinPilot</h1>
          <p className="text-sm text-muted">Answers come from tools that read your real transactions, forecast and goals.</p>
        </div>
        {messages.length > 0 && (
          <Button variant="ghost" size="sm" onClick={() => clear.mutate()}>
            <Eraser className="size-4" /> Clear
          </Button>
        )}
      </div>
      <Card className="flex min-h-0 flex-1 flex-col">
        <div className="flex-1 space-y-5 overflow-y-auto p-5">
          {!messages.length && (
            <div className="flex h-full flex-col items-center justify-center gap-5 text-center">
              <div className="grid size-14 place-items-center rounded-2xl bg-gradient-to-br from-emerald-400 to-indigo-500 text-white shadow-lg">
                <Bot className="size-7" />
              </div>
              <div>
                <div className="font-semibold">What do you want to know about your money?</div>
                <div className="text-sm text-muted">Try one of these:</div>
              </div>
              <div className="max-w-2xl space-y-3">
                {SUGGESTIONS.map((g) => (
                  <div key={g.group}>
                    <div className="mb-1.5 text-[11.5px] font-medium uppercase tracking-wide text-muted">{g.group}</div>
                    <div className="flex flex-wrap justify-center gap-2">
                      {g.items.map((s) => (
                        <button key={s} onClick={() => submit(s)} className="rounded-full border border-line bg-surface-2 px-3 py-1.5 text-[13px] transition hover:border-brand/50 hover:text-brand">
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={cn("flex gap-3", m.role === "user" && "flex-row-reverse")}>
              <div className={cn("grid size-8 shrink-0 place-items-center rounded-full", m.role === "user" ? "bg-surface-2" : "bg-gradient-to-br from-emerald-400 to-indigo-500 text-white")}>
                {m.role === "user" ? <User className="size-4" /> : <Bot className="size-4" />}
              </div>
              <div className={cn("min-w-0 max-w-[85%] rounded-2xl px-4 py-3 text-[14px]", m.role === "user" ? "bg-surface-2" : "border border-line bg-surface", m.artifacts?.length && "w-full")}>
                {m.pending ? (
                  <div className="flex gap-1 py-1">
                    {[0, 1, 2].map((d) => (
                      <span key={d} className="size-2 animate-bounce rounded-full bg-muted" style={{ animationDelay: `${d * 120}ms` }} />
                    ))}
                  </div>
                ) : (
                  <div className="prose-fp">
                    <Markdown remarkPlugins={[remarkGfm]}>{m.content}</Markdown>
                  </div>
                )}
                {m.role === "assistant" && !m.pending && (
                  <>
                    <Artifacts artifacts={m.artifacts ?? []} />
                    <PipelineView pipeline={m.pipeline ?? []} guard={m.guard} />
                    {m.route && <RouteView route={m.route} />}
                    <TraceView trace={m.trace ?? []} />
                    {m.model && (
                      <div className="mt-2">
                        <ModelBadge model={m.model} />
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}
          <div ref={endRef} />
        </div>
        <form
          className="flex gap-2 border-t border-line p-3"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Can I afford a ₹60k trip in December?"
            className="h-11 flex-1 rounded-xl border border-line bg-surface-2/60 px-4 text-sm outline-none focus:border-brand"
          />
          <Button variant="primary" type="submit" loading={send.isPending} className="h-11 px-5">
            {!send.isPending && <Send className="size-4" />}
          </Button>
        </form>
      </Card>
      <div className="mt-2 text-center text-[11.5px] text-muted">
        A team of agents answers from your recorded data: figures are computed by code and checked before you see them. Expand “Used …” to see the exact queries.
      </div>
    </div>
  );
}
