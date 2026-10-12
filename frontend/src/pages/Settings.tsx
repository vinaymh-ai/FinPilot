import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot, Check, Database, KeyRound, Plug, User, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Badge, Button, Card, CardHeader, Input, Label, PageHeader, Select, Skeleton, Toggle } from "../components/ui";
import { api } from "../lib/api";
import { cn, money } from "../lib/format";
import { useRefreshAll } from "../lib/hooks";
import type { LLMState, SettingsView } from "../lib/types";

const AGENTS = ["categorizer", "analyzer", "forecaster", "advisor", "chat", "digest"];

export default function SettingsPage() {
  const qc = useQueryClient();
  const refresh = useRefreshAll();
  const { data, isLoading } = useQuery({ queryKey: ["settings"], queryFn: () => api.get<SettingsView>("/settings") });
  const [profile, setProfile] = useState<SettingsView["profile"] | null>(null);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [agentOv, setAgentOv] = useState<Record<string, { provider?: string; model?: string }>>({});
  const [persona, setPersona] = useState("young_professional");
  const [token, setToken] = useState("");

  useEffect(() => {
    if (!data) return;
    setProfile(data.profile);
    setProvider(data.llm.overrides.provider ?? data.llm.env_provider);
    setModel(data.llm.overrides.model ?? "");
    setAgentOv(data.llm.overrides.agents ?? {});
  }, [data]);

  const { data: models } = useQuery({
    queryKey: ["models", provider],
    queryFn: () => api.get<{ models: string[]; source: string }>(`/llm/models?provider=${provider}`),
    enabled: !!provider,
    staleTime: 300_000,
  });

  const saveProfile = useMutation({
    mutationFn: () => api.put("/settings/profile", profile),
    onSuccess: () => {
      toast.success("Profile saved");
      qc.invalidateQueries({ queryKey: ["settings"] });
      refresh();
    },
    onError: (e: Error) => toast.error(e.message),
  });
  const saveLLM = useMutation({
    mutationFn: (body: Record<string, unknown>) => api.put<LLMState>("/settings/llm", body),
    onSuccess: () => {
      toast.success("LLM routing updated");
      qc.invalidateQueries({ queryKey: ["settings"] });
      qc.invalidateQueries({ queryKey: ["meta"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });
  const test = useMutation({
    mutationFn: () => api.post<{ ok: boolean; reply?: string; model?: string; latency_ms?: number; error?: string }>("/settings/llm/test", { provider, model: model || null }),
    onSuccess: (r) => (r.ok ? toast.success(`Connected: ${r.model} replied “${r.reply}” in ${r.latency_ms} ms`) : toast.error("Connection failed", { description: r.error })),
  });
  const reset = useMutation({
    mutationFn: () => api.post("/demo/reset", { persona }),
    onSuccess: () => {
      toast.success("Demo data loaded and analysed");
      refresh();
    },
    onError: (e: Error) => toast.error(e.message),
  });

  if (isLoading || !data || !profile) return <Skeleton className="h-96" />;
  const llm = data.llm;
  const current = llm.providers.find((p) => p.name === provider);

  return (
    <div className="space-y-6">
      <PageHeader title="Settings" subtitle="Profile, LLM providers (bring your own), notifications and demo data." />

      <Card>
        <CardHeader
          title="LLM providers"
          subtitle="Mesh API is the default gateway (one rsk_ key → GPT, Claude, Gemini, Llama…). Any OpenAI-compatible provider can be swapped in — keys stay in your .env, never in the database."
          icon={<Plug className="size-4" />}
          action={
            <div className="flex items-center gap-2 text-[13px]">
              <span className="text-muted">LLM stages</span>
              <Toggle checked={llm.enabled} onChange={(v) => saveLLM.mutate({ enabled: v })} label="Enable LLM" />
            </div>
          }
        />
        <div className="grid gap-3 p-5 sm:grid-cols-2 xl:grid-cols-3">
          {llm.providers.map((p) => (
            <button
              key={p.name}
              onClick={() => setProvider(p.name)}
              className={cn(
                "rounded-2xl border p-4 text-left transition",
                provider === p.name ? "border-brand bg-brand/5 ring-2 ring-brand/20" : "border-line hover:border-brand/40",
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold">{p.label}</span>
                {p.name === "ollama" ? (
                  <Badge tone="blue">local</Badge>
                ) : p.configured ? (
                  <Badge tone="green">
                    <Check className="size-3" /> ready
                  </Badge>
                ) : (
                  <Badge>no key</Badge>
                )}
              </div>
              <div className="mt-1 flex flex-wrap gap-1">
                {p.name === "mesh" && <Badge tone="violet">default</Badge>}
                {p.free_tier && <Badge tone="blue">free tier</Badge>}
              </div>
              <p className="mt-2 text-[12px] text-muted">{p.notes || `Default model ${p.default_model}`}</p>
              {p.key_env && (
                <div className="mt-2 flex items-center gap-1 font-mono text-[11px] text-muted">
                  <KeyRound className="size-3" /> {p.key_env}
                </div>
              )}
            </button>
          ))}
        </div>
        <div className="grid gap-3 border-t border-line p-5 md:grid-cols-[1fr_1fr_auto_auto] md:items-end">
          <div>
            <Label>Provider for all agents</Label>
            <Select value={provider} onChange={(e) => (setProvider(e.target.value), setModel(""))}>
              {llm.providers.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.label}
                </option>
              ))}
            </Select>
          </div>
          <div>
            <Label hint={models ? `(${models.source} list)` : ""}>Model</Label>
            <Input list="model-list" value={model} placeholder={current?.default_model} onChange={(e) => setModel(e.target.value)} />
            <datalist id="model-list">
              {(models?.models ?? current?.suggested_models ?? []).map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>
          <Button loading={test.isPending} onClick={() => test.mutate()}>
            <Zap className="size-4" /> Test
          </Button>
          <Button variant="primary" loading={saveLLM.isPending} onClick={() => saveLLM.mutate({ provider, model })}>
            Save
          </Button>
        </div>
        <div className="border-t border-line p-5">
          <div className="mb-3 flex items-center gap-2 text-[13px] font-semibold">
            <Bot className="size-4 text-brand" /> Per-agent routing
            <span className="font-normal text-muted">— e.g. a cheap model for bulk categorization, a stronger one for advice</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-[13px]">
              <thead className="text-left text-muted">
                <tr>
                  <th className="py-1.5 font-medium">Agent</th>
                  <th className="font-medium">Currently resolves to</th>
                  <th className="font-medium">Provider override</th>
                  <th className="font-medium">Model override</th>
                </tr>
              </thead>
              <tbody>
                {AGENTS.map((a) => (
                  <tr key={a} className="border-t border-line">
                    <td className="py-2 font-medium capitalize">{a}</td>
                    <td>
                      <Badge tone={llm.agents[a]?.resolved === "rules" ? "neutral" : "violet"}>{llm.agents[a]?.resolved}</Badge>
                    </td>
                    <td className="pr-2">
                      <Select className="h-8 text-[12.5px]" value={agentOv[a]?.provider ?? ""} onChange={(e) => setAgentOv({ ...agentOv, [a]: { ...agentOv[a], provider: e.target.value || undefined } })}>
                        <option value="">inherit</option>
                        {llm.providers.map((p) => (
                          <option key={p.name} value={p.name}>
                            {p.name}
                          </option>
                        ))}
                      </Select>
                    </td>
                    <td>
                      <Input className="h-8 text-[12.5px]" placeholder="inherit" value={agentOv[a]?.model ?? ""} onChange={(e) => setAgentOv({ ...agentOv, [a]: { ...agentOv[a], model: e.target.value || undefined } })} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 flex justify-end">
            <Button onClick={() => saveLLM.mutate({ agents: Object.fromEntries(AGENTS.map((a) => [a, agentOv[a] ?? {}])) })}>Save per-agent routing</Button>
          </div>
        </div>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader title="Profile & balance" icon={<User className="size-4" />} subtitle="The forecast starts from your current balance." />
          <div className="grid gap-3 p-5 sm:grid-cols-2">
            <div>
              <Label>Name</Label>
              <Input value={profile.name} onChange={(e) => setProfile({ ...profile, name: e.target.value })} />
            </div>
            <div>
              <Label>Currency</Label>
              <Select value={profile.currency} onChange={(e) => setProfile({ ...profile, currency: e.target.value })}>
                {["INR", "USD", "EUR", "GBP"].map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </Select>
            </div>
            <div>
              <Label hint={`(as of ${profile.balance_as_of})`}>Current balance</Label>
              <Input type="number" value={profile.current_balance} onChange={(e) => setProfile({ ...profile, current_balance: Number(e.target.value) })} />
            </div>
            <div>
              <Label hint="(blank = auto: ½ month of fixed + essential costs)">Safety buffer</Label>
              <Input type="number" value={profile.safety_buffer ?? ""} placeholder="auto" onChange={(e) => setProfile({ ...profile, safety_buffer: e.target.value ? Number(e.target.value) : null })} />
            </div>
            <div>
              <Label>Digest email</Label>
              <Input type="email" value={profile.digest_email ?? ""} placeholder={data.notifications.email_configured ? "you@example.com" : "configure SMTP_* in .env"} onChange={(e) => setProfile({ ...profile, digest_email: e.target.value || null })} />
            </div>
            <div>
              <Label>Telegram chat id</Label>
              <Input value={profile.telegram_chat_id ?? ""} placeholder={data.notifications.telegram_configured ? "123456789" : "configure TELEGRAM_BOT_TOKEN"} onChange={(e) => setProfile({ ...profile, telegram_chat_id: e.target.value || null })} />
            </div>
            <div className="flex items-center justify-between rounded-xl border border-line p-3 sm:col-span-2">
              <span className="text-sm">Send me the scheduled weekly digest</span>
              <Toggle checked={profile.digest_enabled} onChange={(v) => setProfile({ ...profile, digest_enabled: v })} label="Digest" />
            </div>
            <Button variant="primary" className="sm:col-span-2" loading={saveProfile.isPending} onClick={() => saveProfile.mutate()}>
              Save profile
            </Button>
          </div>
        </Card>
        <div className="space-y-6">
          <Card>
            <CardHeader title="Demo data" icon={<Database className="size-4" />} subtitle="Replace everything with a synthetic 8-month statement and re-run all agents." />
            <div className="flex flex-col gap-3 p-5 sm:flex-row">
              <Select value={persona} onChange={(e) => setPersona(e.target.value)}>
                <option value="young_professional">Arjun — young professional, Bengaluru</option>
                <option value="family">Sharma family — EMIs, school fees</option>
                <option value="freelancer">Neha — freelancer, irregular income</option>
              </Select>
              <Button variant="danger" className="shrink-0 whitespace-nowrap" loading={reset.isPending} onClick={() => confirm("Replace all data with the demo persona?") && reset.mutate()}>
                Load demo
              </Button>
            </div>
          </Card>
          <Card>
            <CardHeader title="API token" icon={<KeyRound className="size-4" />} subtitle="Only needed if the server sets API_TOKEN." />
            <div className="flex gap-2 p-5">
              <Input type="password" value={token} placeholder="Bearer token" onChange={(e) => setToken(e.target.value)} />
              <Button onClick={() => (api.setToken(token), toast.success("Token stored in this browser"), refresh())}>Save</Button>
            </div>
          </Card>
          <Card className="p-5 text-[12.5px] text-muted">
            Balance shown in the app: <b className="num text-ink">{money(profile.current_balance)}</b>. Fallback providers
            (tried in order if the primary fails): {llm.fallback_providers.length ? llm.fallback_providers.join(", ") : "none — set LLM_FALLBACK_PROVIDERS"}.
          </Card>
        </div>
      </div>
    </div>
  );
}
