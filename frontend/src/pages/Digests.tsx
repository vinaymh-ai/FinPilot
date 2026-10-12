import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CalendarClock, Inbox, Mail, MessageSquare, Send, Smartphone } from "lucide-react";
import { useEffect, useState } from "react";
import Markdown from "react-markdown";
import { toast } from "sonner";
import { Badge, Button, Card, CardHeader, Empty, Input, Label, PageHeader, Select, type Tone } from "../components/ui";
import { api } from "../lib/api";
import { cn, dateTime } from "../lib/format";
import { useRefreshAll } from "../lib/hooks";
import type { Digest } from "../lib/types";

type Resp = {
  digests: Digest[];
  scheduler: { running: boolean; jobs: { id: string; name: string; next_run: string | null; trigger: string }[]; digest_cron: string };
};

const PRESETS = [
  { cron: "0 9 * * mon", label: "Mondays 09:00" },
  { cron: "0 8 * * *", label: "Every day 08:00" },
  { cron: "0 19 * * fri", label: "Fridays 19:00" },
  { cron: "0 9 1 * *", label: "1st of month 09:00" },
];
const STATUS_TONE: Record<string, Tone> = { sent: "green", skipped: "neutral", failed: "red" };
const CHANNEL_ICON: Record<string, typeof Mail> = { in_app: Smartphone, email: Mail, telegram: MessageSquare };

export default function Digests() {
  const qc = useQueryClient();
  const refresh = useRefreshAll();
  const { data } = useQuery({ queryKey: ["digests"], queryFn: () => api.get<Resp>("/digests") });
  const [sel, setSel] = useState<number | null>(null);
  const [cron, setCron] = useState("");
  useEffect(() => {
    if (data && !cron) setCron(data.scheduler.digest_cron);
  }, [data, cron]);
  const current = data?.digests.find((d) => d.id === sel) ?? data?.digests[0];

  const sendNow = useMutation({
    mutationFn: () => api.post<{ summary: Record<string, any> }>("/digests/send-now"),
    onSuccess: () => {
      toast.success("Digest composed and delivered");
      refresh();
      setSel(null);
    },
    onError: (e: Error) => toast.error(e.message),
  });
  const saveCron = useMutation({
    mutationFn: (c: string) => api.put("/scheduler", { digest_cron: c }),
    onSuccess: () => {
      toast.success("Schedule updated");
      qc.invalidateQueries({ queryKey: ["digests"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  return (
    <div>
      <PageHeader
        title="Proactive digests"
        subtitle="A background scheduler (APScheduler) re-runs every agent and delivers one key insight + one action — you don't have to ask."
        actions={
          <Button variant="primary" loading={sendNow.isPending} onClick={() => sendNow.mutate()}>
            <Send className="size-4" /> Send a digest now
          </Button>
        }
      />
      <div className="grid gap-6 xl:grid-cols-[360px_1fr]">
        <div className="space-y-4">
          <Card>
            <CardHeader title="Schedule" icon={<CalendarClock className="size-4" />} subtitle={data?.scheduler.running ? "Scheduler running" : "Scheduler disabled (SCHEDULER_ENABLED=false)"} />
            <div className="space-y-3 p-5 pt-3">
              <div>
                <Label>Weekly digest</Label>
                <Select value={PRESETS.some((p) => p.cron === cron) ? cron : ""} onChange={(e) => e.target.value && setCron(e.target.value)}>
                  <option value="">Custom cron…</option>
                  {PRESETS.map((p) => (
                    <option key={p.cron} value={p.cron}>
                      {p.label}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="flex gap-2">
                <Input value={cron} onChange={(e) => setCron(e.target.value)} className="font-mono text-[13px]" />
                <Button onClick={() => saveCron.mutate(cron)} loading={saveCron.isPending}>
                  Save
                </Button>
              </div>
              <div className="space-y-1.5 text-[12.5px]">
                {data?.scheduler.jobs.map((j) => (
                  <div key={j.id} className="flex justify-between gap-2 rounded-lg bg-surface-2/70 px-3 py-2">
                    <span>{j.name}</span>
                    <span className="text-muted">next {dateTime(j.next_run)}</span>
                  </div>
                ))}
              </div>
              <p className="text-[12px] text-muted">
                Email (SMTP) and Telegram deliveries are configured via environment variables; set your address / chat id in
                Settings.
              </p>
            </div>
          </Card>
          <Card>
            <CardHeader title="Inbox" icon={<Inbox className="size-4" />} />
            <div className="max-h-[480px] overflow-y-auto p-2">
              {data?.digests.length ? (
                data.digests.map((d) => (
                  <button
                    key={d.id}
                    onClick={() => setSel(d.id)}
                    className={cn("w-full rounded-xl px-3 py-2.5 text-left transition hover:bg-surface-2", current?.id === d.id && "bg-surface-2")}
                  >
                    <div className="truncate text-[13.5px] font-medium">{d.subject}</div>
                    <div className="text-[12px] text-muted">
                      {dateTime(d.created_at)} · {d.trigger}
                    </div>
                  </button>
                ))
              ) : (
                <Empty title="No digests yet">Send one now or wait for the schedule.</Empty>
              )}
            </div>
          </Card>
        </div>
        <Card className="min-h-[400px]">
          {current ? (
            <div className="p-6">
              <div className="flex flex-wrap items-center gap-2">
                {Object.entries(current.deliveries).map(([ch, d]) => {
                  const Icon = CHANNEL_ICON[ch] ?? Mail;
                  return (
                    <Badge key={ch} tone={STATUS_TONE[d.status] ?? "neutral"} title={d.detail}>
                      <Icon className="size-3" /> {ch.replace("_", "-")}: {d.status}
                    </Badge>
                  );
                })}
              </div>
              <h2 className="mt-4 text-xl font-semibold">{current.subject}</h2>
              <div className="text-[12.5px] text-muted">{dateTime(current.created_at)}</div>
              <div className="prose-fp mt-5 max-w-2xl text-[14.5px] leading-relaxed">
                <Markdown>{current.body_markdown}</Markdown>
              </div>
            </div>
          ) : (
            <Empty icon={<Mail className="size-6" />} title="Select a digest" />
          )}
        </Card>
      </div>
    </div>
  );
}
