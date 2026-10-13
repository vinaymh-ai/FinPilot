import { CircleAlert, RotateCw } from "lucide-react";
import { Component, type ErrorInfo, type ReactNode } from "react";
import { isChunkError } from "../lib/lazy";
import { Button, Card } from "./ui";

type Props = { children: ReactNode; resetKey?: string };
type State = { error: Error | null; eventId: string | null };

/** Report a client-side failure to the API log so it can be debugged server-side. */
export function reportClientError(error: unknown, context: Record<string, unknown> = {}): string {
  const eventId = Math.random().toString(36).slice(2, 10);
  const err = error instanceof Error ? error : new Error(String(error));
  try {
    void fetch("/api/client-errors", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({
        event_id: eventId,
        message: err.message.slice(0, 1000),
        stack: (err.stack ?? "").slice(0, 4000),
        url: window.location.href,
        user_agent: navigator.userAgent,
        context,
      }),
      keepalive: true,
    });
  } catch {
    /* reporting must never throw */
  }
  return eventId;
}

/** Contains a crash to the page that caused it: the rest of the app (navigation, header) stays usable. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, eventId: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    const eventId = reportClientError(error, { componentStack: (info.componentStack ?? "").slice(0, 3000) });
    this.setState({ eventId });
    console.error("[FinPilot] page crashed", eventId, error);
  }

  componentDidUpdate(prev: Props) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null, eventId: null });
  }

  render() {
    const { error, eventId } = this.state;
    if (!error) return this.props.children;
    const chunk = isChunkError(error);
    return (
      <Card className="mx-auto mt-10 max-w-xl p-6">
        <div className="flex items-start gap-3">
          <CircleAlert className="mt-0.5 size-5 shrink-0 text-red-500" />
          <div className="min-w-0 space-y-2">
            <h2 className="font-semibold">{chunk ? "FinPilot was updated" : "This page hit an error"}</h2>
            <p className="text-[13px] text-muted">
              {chunk
                ? "A newer version was deployed while this tab was open. Reload to continue."
                : "The rest of the app still works — pick another page or reload this one."}
            </p>
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-surface-2/70 p-2 font-mono text-[11.5px] text-muted">
              {error.message}
            </pre>
            {eventId && <p className="text-[11.5px] text-muted">Logged on the server as client error <code>{eventId}</code>.</p>}
            <Button variant="primary" size="sm" onClick={() => window.location.reload()}>
              <RotateCw className="size-4" /> Reload
            </Button>
          </div>
        </div>
      </Card>
    );
  }
}
