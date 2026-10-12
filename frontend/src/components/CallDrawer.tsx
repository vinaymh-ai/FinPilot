import { useQuery } from "@tanstack/react-query";
import { Check, Copy } from "lucide-react";
import { useState, type ReactNode } from "react";
import { api } from "../lib/api";
import { dateTime } from "../lib/format";
import { Badge, Button, Drawer, Skeleton } from "./ui";

export type CallRow = {
  id: number;
  created_at: string | null;
  kind: "llm" | "jev";
  agent: string;
  purpose: string | null;
  provider: string;
  model: string;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  latency_ms: number | null;
  cached: boolean | null;
  success: boolean | null;
  error: string | null;
  cost_usd: number | null;
  run_id: number | null;
  user_id: number | null;
  request_id: string | null;
  has_payload: boolean;
};

export type CallDetail = CallRow & { request: unknown; response: unknown };

export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <Button
      size="sm"
      variant="ghost"
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => {
          setDone(true);
          setTimeout(() => setDone(false), 1200);
        });
      }}
    >
      {done ? <Check className="size-3.5" /> : <Copy className="size-3.5" />} {done ? "Copied" : label}
    </Button>
  );
}

export function JsonBlock({ value, title, maxHeight = "28rem" }: { value: unknown; title: ReactNode; maxHeight?: string }) {
  const text = value === undefined ? "" : JSON.stringify(value, null, 2);
  return (
    <div className="overflow-hidden rounded-xl border border-line">
      <div className="flex items-center justify-between border-b border-line bg-surface-2/60 px-3 py-1.5">
        <span className="text-[12px] font-semibold text-muted">{title}</span>
        {value != null && <CopyButton text={text} />}
      </div>
      {value == null ? (
        <div className="px-3 py-3 text-[12.5px] text-muted">Not captured (failed before a response, or payload capture is off / purged).</div>
      ) : (
        <pre className="overflow-auto whitespace-pre-wrap break-words p-3 font-mono text-[11.5px] leading-relaxed" style={{ maxHeight }}>
          {text}
        </pre>
      )}
    </div>
  );
}

type Message = { role?: string; content?: unknown; tool_calls?: unknown; name?: string };

function Messages({ messages }: { messages: Message[] }) {
  return (
    <div className="space-y-2">
      {messages.map((m, i) => (
        <div key={i} className="rounded-xl border border-line p-3">
          <div className="mb-1.5 flex items-center gap-2">
            <Badge tone={m.role === "system" ? "neutral" : m.role === "user" ? "blue" : m.role === "tool" ? "amber" : "violet"}>{m.role ?? "?"}</Badge>
            {m.name && <span className="text-[11.5px] text-muted">{m.name}</span>}
          </div>
          {typeof m.content === "string" ? (
            <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words font-mono text-[11.5px] leading-relaxed">{m.content || "(empty)"}</pre>
          ) : m.content != null ? (
            <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words font-mono text-[11.5px]">{JSON.stringify(m.content, null, 2)}</pre>
          ) : null}
          {m.tool_calls != null && (
            <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-surface-2 p-2 font-mono text-[11px]">
              {JSON.stringify(m.tool_calls, null, 2)}
            </pre>
          )}
        </div>
      ))}
    </div>
  );
}

function answerOf(call: CallDetail): Message | null {
  const r = call.response as Record<string, any> | null;
  if (!r) return null;
  if (Array.isArray(r.choices) && r.choices[0]?.message) return r.choices[0].message as Message;
  if ("content" in r) return { role: "assistant", content: r.content, tool_calls: r.tool_calls ?? undefined };
  return null;
}

export function CallDrawer({ callId, onClose, onShowLogs }: { callId: number | null; onClose: () => void; onShowLogs?: (requestId: string) => void }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["observability", "call", callId],
    queryFn: () => api.get<{ call: CallDetail }>(`/observability/calls/${callId}`),
    enabled: callId != null,
  });
  const call = data?.call;
  const req = call?.request as { messages?: Message[] } | null | undefined;
  const answer = call ? answerOf(call) : null;
  return (
    <Drawer
      open={callId != null}
      onClose={onClose}
      wide
      title={
        call ? (
          <span className="flex items-center gap-2">
            {call.kind === "jev" ? <Badge tone="cyan">jev</Badge> : <Badge tone="violet">llm</Badge>}
            {call.agent}
            {call.purpose ? ` · ${call.purpose.replace(/_/g, " ")}` : ""}
            <span className="font-normal text-muted">#{call.id}</span>
          </span>
        ) : (
          "Model call"
        )
      }
    >
      {isLoading && <Skeleton className="h-96" />}
      {error && <p className="text-sm text-red-500">{(error as Error).message}</p>}
      {call && (
        <div className="space-y-5">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 rounded-xl border border-line p-3 text-[12.5px] sm:grid-cols-3">
            <Meta k="When">{call.created_at ? dateTime(call.created_at) : "–"}</Meta>
            <Meta k="Model">
              {call.provider} · {call.model}
            </Meta>
            <Meta k="Status">{call.success ? <Badge tone="green">ok{call.cached ? " · cache" : ""}</Badge> : <Badge tone="red">failed</Badge>}</Meta>
            <Meta k="Tokens in / out">
              {call.prompt_tokens ?? 0} / {call.completion_tokens ?? 0}
            </Meta>
            <Meta k="Latency">{call.cached ? "cache hit" : `${call.latency_ms ?? 0} ms`}</Meta>
            <Meta k="Cost">{call.cost_usd != null ? `$${call.cost_usd.toFixed(6)}` : "–"}</Meta>
            <Meta k="Agent run">{call.run_id ? `#${call.run_id}` : "–"}</Meta>
            <Meta k="Request id">
              {call.request_id ? (
                onShowLogs ? (
                  <button className="font-mono text-brand hover:underline" onClick={() => onShowLogs(call.request_id!)}>
                    {call.request_id}
                  </button>
                ) : (
                  <span className="font-mono">{call.request_id}</span>
                )
              ) : (
                "–"
              )}
            </Meta>
            <Meta k="User">{call.user_id ?? "–"}</Meta>
          </dl>
          {call.error && (
            <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-3 font-mono text-[12px] text-red-600 dark:text-red-400">{call.error}</div>
          )}
          {Array.isArray(req?.messages) && (
            <section>
              <h4 className="mb-2 text-[13px] font-semibold">Prompt</h4>
              <Messages messages={req!.messages!} />
            </section>
          )}
          {answer && (
            <section>
              <h4 className="mb-2 text-[13px] font-semibold">Answer</h4>
              <Messages messages={[answer]} />
            </section>
          )}
          <JsonBlock title="Raw request" value={call.request} />
          <JsonBlock title="Raw response" value={call.response} />
        </div>
      )}
    </Drawer>
  );
}

function Meta({ k, children }: { k: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] text-muted">{k}</dt>
      <dd className="truncate">{children}</dd>
    </div>
  );
}
