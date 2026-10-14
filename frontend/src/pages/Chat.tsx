import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot, Brain, ChevronDown, PanelLeft, Pencil, Pin, PinOff, Scale, Search, Send, ShieldCheck, SquarePen, Trash2, User, Wrench } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { toast } from "sonner";
import { Artifacts } from "../components/ChatArtifacts";
import { Badge, Button, Card, ModelBadge, ProbabilityBars, Skeleton } from "../components/ui";
import { api } from "../lib/api";
import { cn } from "../lib/format";
import type { ChatArtifact, ChatMemoryInfo, ChatMessage, ChatRoute, ChatSession, ChatTrace } from "../lib/types";
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
  {
    group: "Search by meaning & teach me",
    items: ["Show my coffee purchases", "How many cab rides this year?", "Priya Nair is my sister", "What do you remember about me?"],
  },
];
const AGENT_LABEL: Record<string, string> = {
  memory: "Memory",
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
          <Badge tone={p === "verifier" ? (guard?.ok ? "green" : "neutral") : p === "sql_analyst" || p === "memory" ? "violet" : "neutral"}>
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

function MemoryView({ memory }: { memory?: ChatMemoryInfo | null }) {
  const notes = memory?.notes ?? [];
  const saved = memory?.saved ?? [];
  if (!notes.length && !saved.length) return null;
  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11.5px] text-muted">
      <Brain className="size-3.5 text-violet-500" />
      {notes.map((n, i) => (
        <span key={i} className="rounded-full bg-violet-500/10 px-2 py-0.5 text-violet-700 dark:text-violet-300">
          {n}
        </span>
      ))}
      {saved.length > 0 && (
        <Link to="/settings#memory" className="rounded-full bg-violet-500/10 px-2 py-0.5 font-medium text-violet-700 hover:underline dark:text-violet-300">
          Memory updated
        </Link>
      )}
    </div>
  );
}

const DAY = 86_400_000;
function groupSessions(sessions: ChatSession[]): [string, ChatSession[]][] {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const groups: Record<string, ChatSession[]> = {};
  const order = ["Pinned", "Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Older"];
  for (const s of sessions) {
    const t = s.updated_at ? new Date(s.updated_at.endsWith("Z") || s.updated_at.includes("+") ? s.updated_at : s.updated_at + "Z").getTime() : 0;
    const key = s.pinned
      ? "Pinned"
      : t >= today.getTime()
        ? "Today"
        : t >= today.getTime() - DAY
          ? "Yesterday"
          : t >= today.getTime() - 7 * DAY
            ? "Previous 7 days"
            : t >= today.getTime() - 30 * DAY
              ? "Previous 30 days"
              : "Older";
    (groups[key] ??= []).push(s);
  }
  return order.filter((k) => groups[k]?.length).map((k) => [k, groups[k]]);
}

function SessionItem({ s, active, onOpen }: { s: ChatSession; active: boolean; onOpen: () => void }) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(s.title);
  const update = useMutation({
    mutationFn: (body: { title?: string; pinned?: boolean }) => api.patch(`/chat/sessions/${s.id}`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["chat-sessions"] }),
    onError: (e: Error) => toast.error(e.message),
  });
  const remove = useMutation({
    mutationFn: () => api.del(`/chat/sessions/${s.id}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["chat-sessions"] });
      qc.removeQueries({ queryKey: ["chat", s.id] });
      if (active) navigate("/chat");
    },
  });
  if (editing) {
    return (
      <form
        className="px-1"
        onSubmit={(e) => {
          e.preventDefault();
          if (title.trim()) update.mutate({ title: title.trim() });
          setEditing(false);
        }}
      >
        <input
          autoFocus
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          onBlur={() => setEditing(false)}
          onKeyDown={(e) => e.key === "Escape" && setEditing(false)}
          className="h-8 w-full rounded-lg border border-brand bg-surface px-2 text-[13px] outline-none"
          aria-label="Conversation title"
        />
      </form>
    );
  }
  return (
    <div className={cn("group flex items-center rounded-lg pr-1 transition", active ? "bg-surface-2 text-ink" : "text-muted hover:bg-surface-2/60 hover:text-ink")}>
      <button onClick={onOpen} className="min-w-0 flex-1 truncate px-2.5 py-2 text-left text-[13px]" title={s.title}>
        {s.title}
      </button>
      <div className={cn("flex shrink-0 items-center gap-0.5", active ? "opacity-100" : "opacity-0 group-hover:opacity-100 focus-within:opacity-100")}>
        <button onClick={() => update.mutate({ pinned: !s.pinned })} className="rounded p-1 hover:bg-surface hover:text-ink" aria-label={s.pinned ? "Unpin" : "Pin"} title={s.pinned ? "Unpin" : "Pin"}>
          {s.pinned ? <PinOff className="size-3.5" /> : <Pin className="size-3.5" />}
        </button>
        <button onClick={() => (setTitle(s.title), setEditing(true))} className="rounded p-1 hover:bg-surface hover:text-ink" aria-label="Rename" title="Rename">
          <Pencil className="size-3.5" />
        </button>
        <button
          onClick={() => {
            if (window.confirm(`Delete “${s.title}”? This removes the conversation's messages.`)) remove.mutate();
          }}
          className="rounded p-1 hover:bg-surface hover:text-red-500"
          aria-label="Delete"
          title="Delete"
        >
          <Trash2 className="size-3.5" />
        </button>
      </div>
    </div>
  );
}

function SessionList({ activeId, onNavigate }: { activeId?: string; onNavigate?: () => void }) {
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [term, setTerm] = useState("");
  useEffect(() => {
    const t = setTimeout(() => setTerm(search.trim()), 250);
    return () => clearTimeout(t);
  }, [search]);
  const { data, isLoading } = useQuery({
    queryKey: ["chat-sessions", term],
    queryFn: () => api.get<{ sessions: ChatSession[] }>(`/chat/sessions${term ? `?q=${encodeURIComponent(term)}` : ""}`),
    placeholderData: (prev) => prev,
  });
  const groups = groupSessions(data?.sessions ?? []);
  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <Button
        variant="primary"
        className="w-full justify-center"
        onClick={() => {
          navigate("/chat");
          onNavigate?.();
        }}
      >
        <SquarePen className="size-4" /> New chat
      </Button>
      <div className="relative">
        <Search className="absolute left-2.5 top-2.5 size-3.5 text-muted" />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search chats…"
          className="h-9 w-full rounded-lg border border-line bg-surface-2/60 pl-8 pr-2 text-[13px] outline-none focus:border-brand"
          aria-label="Search conversations"
        />
      </div>
      <div className="-mx-1 min-h-0 flex-1 space-y-3 overflow-y-auto px-1 pb-2">
        {isLoading && <Skeleton className="h-8" />}
        {!isLoading && !groups.length && (
          <div className="px-2 py-6 text-center text-[12.5px] text-muted">{term ? "No conversations match." : "Your conversations will appear here."}</div>
        )}
        {groups.map(([label, items]) => (
          <div key={label}>
            <div className="mb-1 px-2.5 text-[11px] font-medium uppercase tracking-wide text-muted">{label}</div>
            <div className="space-y-0.5">
              {items.map((s) => (
                <SessionItem
                  key={s.id}
                  s={s}
                  active={s.id === activeId}
                  onOpen={() => {
                    navigate(`/chat/${s.id}`);
                    onNavigate?.();
                  }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

type ChatResponse = {
  answer: string;
  trace: ChatTrace[];
  model: string;
  route: ChatRoute | null;
  artifacts?: ChatArtifact[];
  pipeline?: string[];
  guard?: ChatMessage["guard"];
  memory?: ChatMemoryInfo | null;
  session_id: string;
  session: ChatSession;
};

export default function Chat() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const { sessionId } = useParams();
  const [drawer, setDrawer] = useState(false);
  const { data, isFetching } = useQuery({
    queryKey: ["chat", sessionId],
    queryFn: () => api.get<{ messages: ChatMessage[]; session: ChatSession | null }>(`/chat/${sessionId}`),
    enabled: !!sessionId,
  });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [text, setText] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    if (!sessionId) setMessages([]);
    else if (data) setMessages(data.messages);
  }, [data, sessionId]);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  useEffect(() => {
    inputRef.current?.focus();
  }, [sessionId]);

  const send = useMutation({
    mutationFn: (message: string) => api.post<ChatResponse>("/chat", { message, session_id: sessionId ?? null }),
    onMutate: (message) =>
      setMessages((m) => [...m, { role: "user", content: message }, { role: "assistant", content: "", pending: true }]),
    onSuccess: (r, message) => {
      const reply: ChatMessage = {
        role: "assistant",
        content: r.answer,
        trace: r.trace,
        model: r.model,
        route: r.route,
        artifacts: r.artifacts ?? [],
        pipeline: r.pipeline ?? [],
        guard: r.guard ?? null,
        memory: r.memory ?? null,
      };
      setMessages((m) => {
        const next = [...m.filter((x) => !x.pending), reply];
        qc.setQueryData(["chat", r.session_id], { messages: next, session: r.session });
        return next;
      });
      qc.invalidateQueries({ queryKey: ["chat-sessions"] });
      if (r.memory?.saved?.length) qc.invalidateQueries({ queryKey: ["memories"] });
      if (!sessionId) navigate(`/chat/${r.session_id}`, { replace: true });
      void message;
    },
    onError: (e: Error) => {
      setMessages((m) => m.filter((x) => !x.pending));
      toast.error(e.message);
    },
  });
  const submit = (q?: string) => {
    const msg = (q ?? text).trim();
    if (!msg || send.isPending) return;
    setText("");
    send.mutate(msg);
  };
  const title = sessionId ? (data?.session?.title ?? (isFetching ? "…" : "Conversation")) : "New chat";

  return (
    <div className="mx-auto flex h-[calc(100vh-8rem)] max-w-6xl gap-4">
      <aside className="hidden w-64 shrink-0 lg:block">
        <SessionList activeId={sessionId} />
      </aside>
      {drawer && (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <div className="absolute inset-0 bg-black/40" onClick={() => setDrawer(false)} />
          <div className="relative h-full w-72 max-w-[85vw] border-r border-line bg-surface p-3">
            <SessionList activeId={sessionId} onNavigate={() => setDrawer(false)} />
          </div>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <Button variant="ghost" size="sm" className="lg:hidden" onClick={() => setDrawer(true)} aria-label="Conversations">
              <PanelLeft className="size-4" />
            </Button>
            <div className="min-w-0">
              <h1 className="truncate text-2xl font-semibold tracking-tight">Ask FinPilot</h1>
              <p className="truncate text-sm text-muted">{title} · answers come from your real transactions, forecast and goals — and FinPilot remembers the conversation.</p>
            </div>
          </div>
          {sessionId && (
            <Button variant="ghost" size="sm" className="shrink-0 whitespace-nowrap" onClick={() => navigate("/chat")}>
              <SquarePen className="size-4" /> New chat
            </Button>
          )}
        </div>
        <Card className="flex min-h-0 flex-1 flex-col">
          <div className="flex-1 space-y-5 overflow-y-auto p-5">
            {!messages.length && !(sessionId && isFetching) && (
              <div className="flex h-full flex-col items-center justify-center gap-5 text-center">
                <div className="grid size-14 place-items-center rounded-2xl bg-gradient-to-br from-emerald-400 to-indigo-500 text-white shadow-lg">
                  <Bot className="size-7" />
                </div>
                <div>
                  <div className="font-semibold">What do you want to know about your money?</div>
                  <div className="text-sm text-muted">Ask follow-ups naturally — “and last month?”, “the same person?” — I keep track.</div>
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
              <div key={m.id ?? `local-${i}`} className={cn("flex gap-3", m.role === "user" && "flex-row-reverse")}>
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
                      <MemoryView memory={m.memory} />
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
            className="flex items-end gap-2 border-t border-line p-3"
            onSubmit={(e) => {
              e.preventDefault();
              submit();
            }}
          >
            <textarea
              ref={inputRef}
              rows={1}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
              placeholder={messages.length ? "Ask a follow-up — “and last month?”, “only UPI”, “summarise our chat”" : "Can I afford a ₹60k trip in December?"}
              className="max-h-40 min-h-11 flex-1 resize-none rounded-xl border border-line bg-surface-2/60 px-4 py-2.5 text-sm outline-none focus:border-brand"
              aria-label="Message"
            />
            <Button variant="primary" type="submit" loading={send.isPending} className="h-11 px-5" aria-label="Send">
              {!send.isPending && <Send className="size-4" />}
            </Button>
          </form>
        </Card>
        <div className="mt-2 text-center text-[11.5px] text-muted">
          Figures are computed by code and checked before you see them. FinPilot remembers this conversation and what you tell it about your payees (manage in{" "}
          <Link to="/settings#memory" className="underline">
            Settings → Memory
          </Link>
          ).
        </div>
      </div>
    </div>
  );
}
