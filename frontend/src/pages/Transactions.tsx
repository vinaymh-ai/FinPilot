import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowDownLeft, ArrowUpRight, Check, CheckCircle2, ChevronLeft, ChevronRight, FileUp, History, KeyRound, Plus, Receipt, Repeat, Search, TriangleAlert, Upload, Wand2, X, Zap } from "lucide-react";
import { useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { CategoryPill, SourceBadge } from "../components/domain";
import { Badge, Button, Card, Drawer, Empty, Input, Label, PageHeader, Select, Skeleton, Toggle } from "../components/ui";
import { ApiError, api } from "../lib/api";
import { cn, date, money } from "../lib/format";
import { useMeta, useRefreshAll } from "../lib/hooks";
import type { CategoryEvent, Txn } from "../lib/types";
import { EMPTY_RULE, RuleDrawer, type RuleDraft } from "./Categories";

type Page = {
  total: number;
  page: number;
  page_size: number;
  items: Txn[];
  sources: Record<string, number>;
  totals?: { debits: number; credits: number; debit_count: number; credit_count: number };
};
const CHANNELS: Record<string, string> = {
  upi: "UPI",
  card: "Card",
  netbanking: "Net banking",
  neft: "NEFT",
  imps: "IMPS",
  rtgs: "RTGS",
  atm: "ATM",
  auto_debit: "Auto-debit",
  cheque: "Cheque",
  cash_deposit: "Cash deposit",
  bank: "Bank",
  wallet: "Wallet",
  other: "Other",
};
type ImportSummary = {
  format: string;
  rows: number;
  date_range: [string, string] | null;
  money_in: number;
  money_out: number;
  checks: { balance_checked?: number; balance_reconciled?: number; balance_reconciled_pct?: number };
};
type UploadBatch = {
  filename: string;
  format: string;
  imported: number;
  duplicates: number;
  invalid: number;
  total: number;
  errors: string[];
  warnings: string[];
  mapping: Record<string, string>;
  summary: ImportSummary;
};
type UploadResp = { batch: UploadBatch; run: { run_id: number; summary: Record<string, any> } | null };
type UploadError = { message: string; code?: string; errors?: string[]; warnings?: string[]; mapping?: Record<string, string>; format?: string; filename?: string };

const ACCEPT = ".csv,.tsv,.txt,.xls,.xlsx,.xlsm,.htm,.html,.pdf";
const ROLE_LABEL: Record<string, string> = {
  date: "Date",
  value_date: "Value date",
  description: "Description",
  debit: "Money out",
  credit: "Money in",
  amount: "Amount",
  type: "Dr/Cr",
  balance: "Balance",
  reference: "Reference",
  time: "Time",
  serial: "Serial",
  account: "Account",
  category: "Category",
};

function ImportReport({ batch, onClose }: { batch: UploadBatch; onClose: () => void }) {
  const s = batch.summary;
  const pct = s?.checks?.balance_reconciled_pct;
  return (
    <Card className="mb-5 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 font-semibold">
          <CheckCircle2 className="size-4 text-emerald-500" /> Imported {batch.imported} of {batch.total} transactions from {batch.filename}
          <Badge>{batch.format?.toUpperCase()}</Badge>
        </div>
        <button onClick={onClose} className="text-muted hover:text-ink" aria-label="Close">
          <X className="size-4" />
        </button>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[12.5px] text-muted">
        {s?.date_range && (
          <span>
            {date(s.date_range[0], true)} → {date(s.date_range[1], true)}
          </span>
        )}
        <span>money in {money(s?.money_in)}</span>
        <span>money out {money(s?.money_out)}</span>
        <span>{batch.duplicates} duplicates skipped</span>
        {batch.invalid > 0 && <span className="text-amber-600">{batch.invalid} rows unreadable</span>}
        {pct != null && (
          <span className={pct >= 99 ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600"}>
            balance check {s.checks.balance_reconciled}/{s.checks.balance_checked} rows reconcile ({pct}%)
          </span>
        )}
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {Object.entries(batch.mapping).map(([role, header]) => (
          <Badge key={role} tone="blue" title={`column "${header}"`}>
            {ROLE_LABEL[role] ?? role}: {header}
          </Badge>
        ))}
      </div>
      {[...batch.warnings, ...batch.errors].length > 0 && (
        <ul className="mt-2 list-disc space-y-0.5 pl-5 text-[12px] text-amber-700 dark:text-amber-400">
          {[...batch.warnings, ...batch.errors].slice(0, 8).map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function ImportError({ err, onClose }: { err: UploadError; onClose: () => void }) {
  return (
    <Card className="mb-5 border-red-500/30 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 font-semibold text-red-600 dark:text-red-400">
          <TriangleAlert className="size-4" /> {err.message}
          {err.filename && <span className="font-normal text-muted">— {err.filename}</span>}
        </div>
        <button onClick={onClose} className="text-muted hover:text-ink" aria-label="Close">
          <X className="size-4" />
        </button>
      </div>
      {err.mapping && Object.keys(err.mapping).length > 0 && (
        <div className="mt-2 text-[12.5px] text-muted">
          Columns recognised:{" "}
          {Object.entries(err.mapping)
            .map(([r, h]) => `${ROLE_LABEL[r] ?? r} = "${h}"`)
            .join(", ")}
        </div>
      )}
      {(err.errors?.length ?? 0) > 0 && (
        <ul className="mt-2 list-disc space-y-0.5 pl-5 text-[12px] text-muted">
          {err.errors!.slice(0, 8).map((e, i) => (
            <li key={i}>{e}</li>
          ))}
        </ul>
      )}
      <p className="mt-2 text-[12px] text-muted">
        Supported: CSV/TSV/TXT, Excel (.xlsx/.xls), HTML-as-XLS and PDF e-statements from any bank. Scanned PDFs (images) can't be read —
        download the e-statement or export CSV/Excel from net banking.
      </p>
    </Card>
  );
}

const ACTOR: Record<string, string> = {
  categorizer: "Categorizer agent",
  user: "You",
  review: "You (review)",
  rule: "Your rule",
  chat: "You, in chat",
};

function HistoryDrawer({ txn, onClose, onRule }: { txn: Txn | null; onClose: () => void; onRule: (t: Txn) => void }) {
  const { data, isLoading } = useQuery({
    queryKey: ["txn-history", txn?.id],
    queryFn: () => api.get<{ events: CategoryEvent[] }>(`/transactions/${txn!.id}/history`),
    enabled: !!txn,
  });
  if (!txn) return null;
  return (
    <Drawer open={!!txn} onClose={onClose} title={<span className="inline-flex items-center gap-2"><History className="size-4" /> Category history</span>}>
      <div className="space-y-4">
        <div>
          <div className="font-medium">{txn.merchant}</div>
          <div className="break-all text-[12px] text-muted">{txn.description}</div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-[12.5px]">
            <span className={cn("num font-medium", txn.amount > 0 && "text-emerald-500")}>{money(txn.amount, { sign: true })}</span>
            <span className="text-muted">{date(txn.date)}</span>
            <CategoryPill id={txn.category} label={txn.category_label} />
          </div>
          {txn.category_reason && <p className="mt-2 rounded-xl bg-surface-2/60 p-2.5 text-[12.5px]">{txn.category_reason}</p>}
        </div>
        {isLoading ? (
          <Skeleton className="h-24" />
        ) : !data?.events.length ? (
          <p className="text-[12.5px] text-muted">No recorded changes yet — this label predates the audit trail.</p>
        ) : (
          <ol className="relative space-y-3 border-l border-line pl-4">
            {data.events.map((e) => (
              <li key={e.id} className="text-[12.5px]">
                <span className="absolute -left-[5px] mt-1.5 size-2.5 rounded-full bg-brand" />
                <div className="flex flex-wrap items-center gap-1.5">
                  <b>{ACTOR[e.actor] ?? e.actor}</b>
                  {e.from_label ? (
                    <>
                      <span className="text-muted">changed</span> {e.from_label} <span className="text-muted">→</span>
                    </>
                  ) : (
                    <span className="text-muted">set</span>
                  )}
                  <b>{e.to_label}</b>
                  {e.source && <Badge>{e.source}</Badge>}
                  {e.confidence != null && <span className="text-muted">{Math.round(e.confidence * 100)}%</span>}
                </div>
                {e.reason && <div className="mt-0.5 text-muted">{e.reason}</div>}
                <div className="mt-0.5 text-[11px] text-muted">
                  {e.at ? new Date(e.at.endsWith("Z") || e.at.includes("+") ? e.at : e.at + "Z").toLocaleString() : ""}
                  {e.run_id ? ` · run #${e.run_id}` : ""}
                </div>
              </li>
            ))}
          </ol>
        )}
        <Button onClick={() => onRule(txn)}>
          <Wand2 className="size-4" /> Make a rule for “{txn.merchant}”
        </Button>
      </div>
    </Drawer>
  );
}

const FLAGS = ["anomalies", "recurring", "impulse", "review"] as const;

export default function Transactions() {
  const { data: meta } = useMeta();
  const qc = useQueryClient();
  const refresh = useRefreshAll();
  const [params0] = useSearchParams();
  const [q, setQ] = useState(params0.get("q") ?? "");
  const [category, setCategory] = useState(params0.get("category") ?? "");
  const [source, setSource] = useState("");
  const initialFlag = params0.get("flag");
  const [flag, setFlag] = useState<"" | "anomalies" | "recurring" | "impulse" | "review">(
    FLAGS.includes(initialFlag as (typeof FLAGS)[number]) ? (initialFlag as (typeof FLAGS)[number]) : "",
  );
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [bulkCategory, setBulkCategory] = useState("");
  const [remember, setRemember] = useState(false);
  const [historyOf, setHistoryOf] = useState<Txn | null>(null);
  const [ruleDraft, setRuleDraft] = useState<RuleDraft | null>(null);
  const [direction, setDirection] = useState<"" | "debit" | "credit">("");
  const [channel, setChannel] = useState("");
  const [page, setPage] = useState(1);
  const [adding, setAdding] = useState(false);
  const [drag, setDrag] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const [report, setReport] = useState<UploadBatch | null>(null);
  const [uploadErr, setUploadErr] = useState<UploadError | null>(null);
  const [locked, setLocked] = useState<{ file: File; code: string } | null>(null);
  const [password, setPassword] = useState("");

  const params = new URLSearchParams({ page: String(page), page_size: "40" });
  if (q) params.set("q", q);
  if (category) params.set("category", category);
  if (source) params.set("source", source);
  if (flag) params.set(flag, "true");
  if (direction) params.set("direction", direction);
  if (channel) params.set("channel", channel);
  const { data, isLoading } = useQuery({
    queryKey: ["transactions", params.toString()],
    queryFn: () => api.get<Page>(`/transactions?${params}`),
    placeholderData: (prev) => prev,
  });

  const upload = useMutation({
    mutationFn: ({ file, password }: { file: File; password?: string }) => {
      const fd = new FormData();
      fd.append("file", file);
      if (password) fd.append("password", password);
      return api.upload<UploadResp>("/transactions/upload", fd);
    },
    onMutate: () => {
      setUploadErr(null);
      setReport(null);
    },
    onSuccess: (r) => {
      setLocked(null);
      setPassword("");
      setReport(r.batch);
      toast.success(`Imported ${r.batch.imported} transactions`, {
        description: `${r.batch.duplicates} duplicates skipped · ${r.batch.invalid} invalid${r.run ? ` · agents ran (#${r.run.run_id})` : ""}`,
      });
      refresh();
    },
    onError: (e: Error, vars) => {
      const detail = (e instanceof ApiError && typeof e.detail === "object" && e.detail ? e.detail : { message: e.message }) as UploadError;
      if (detail.code === "password_required" || detail.code === "password_incorrect") {
        setLocked({ file: vars.file, code: detail.code });
        if (detail.code === "password_incorrect") toast.error("That password didn't open the PDF");
        return;
      }
      setUploadErr(detail);
      toast.error("Upload failed", { description: detail.message });
    },
  });

  const recat = useMutation({
    mutationFn: ({ t, category }: { t: Txn; category: string }) =>
      api.patch<{ updated: number }>(`/transactions/${t.id}`, { category, apply_to_merchant: !t.payment_provider && !t.needs_review }),
    onSuccess: (r, v) => {
      toast.success(`Updated: ${v.t.merchant} → ${meta?.categories.find((c) => c.id === v.category)?.label}`, {
        description: v.t.payment_provider || v.t.needs_review
          ? "Only this transaction was changed; other payments may have different purposes."
          : `${r.updated} transaction(s) updated; future imports will use this.`,
      });
      qc.invalidateQueries({ queryKey: ["transactions"] });
    },
  });

  const bulk = useMutation({
    mutationFn: (body: { ids: number[]; category?: string; approve?: boolean; remember?: boolean }) =>
      api.post<{ updated: number; changed: number; learned_merchants: number; not_learned?: string[] }>("/transactions/bulk", body),
    onSuccess: (r, v) => {
      const skipped = r.not_learned?.length
        ? ` Not remembered (your selected rows had different categories): ${r.not_learned.join(", ")}.`
        : "";
      toast.success(v.approve ? `Approved ${r.updated} transaction(s)` : `Moved ${r.updated} transaction(s)`, {
        description: (r.learned_merchants ? `Learned ${r.learned_merchants} merchant(s) for future imports.` : "Logged in each transaction's history.") + skipped,
      });
      setSelected(new Set());
      qc.invalidateQueries({ queryKey: ["transactions"] });
      qc.invalidateQueries({ queryKey: ["categories"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });
  const pageIds = data?.items.map((t) => t.id) ?? [];
  const allOnPage = pageIds.length > 0 && pageIds.every((id) => selected.has(id));
  const toggle = (id: number) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const onFiles = (files: FileList | null) => {
    const f = files?.[0];
    if (!f) return;
    const ext = "." + (f.name.toLowerCase().split(".").pop() ?? "");
    if (!ACCEPT.split(",").includes(ext)) return toast.error("Unsupported file", { description: `Upload a statement as ${ACCEPT.replaceAll(",", " ")}` });
    setLocked(null);
    upload.mutate({ file: f });
  };
  const pages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;

  return (
    <div>
      <PageHeader
        title="Transactions"
        subtitle="Every row shows how it was categorised (and why). Fix a category once — FinPilot remembers the merchant."
        actions={
          <>
            <Button onClick={() => setAdding(true)}>
              <Plus className="size-4" /> Add manually
            </Button>
            <Button variant="primary" loading={upload.isPending} onClick={() => fileRef.current?.click()}>
              <Upload className="size-4" /> Upload statement
            </Button>
            <input
              ref={fileRef}
              type="file"
              accept={ACCEPT}
              hidden
              onChange={(e) => {
                onFiles(e.target.files);
                e.target.value = "";
              }}
            />
          </>
        }
      />

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          onFiles(e.dataTransfer.files);
        }}
        className={cn(
          "mb-5 flex flex-col items-center gap-1 rounded-2xl border-2 border-dashed px-4 py-5 text-center text-[13px] text-muted transition sm:flex-row sm:justify-center sm:gap-3",
          drag ? "border-brand bg-brand/5" : "border-line",
        )}
      >
        <FileUp className="size-5" />
        <span>
          Drop a bank statement here — <b>CSV, Excel (.xlsx/.xls) or PDF</b> from any bank (HDFC, ICICI, SBI, Axis, Kotak…).
          Preambles, Debit/Credit or signed amounts, any date style and password-protected PDFs all work. Re-uploading adds nothing.
        </span>
        <a href="/sample_transactions.csv" download className="text-brand hover:underline">
          sample file
        </a>
      </div>

      {locked && (
        <Card className="mb-5 p-4">
          <form
            className="flex flex-wrap items-end gap-3"
            onSubmit={(e) => {
              e.preventDefault();
              if (password) upload.mutate({ file: locked.file, password });
            }}
          >
            <div className="min-w-[240px] flex-1">
              <Label hint="(often your date of birth or customer ID — it's used once and never stored)">
                <span className="inline-flex items-center gap-1.5">
                  <KeyRound className="size-3.5" /> {locked.file.name} is password-protected
                </span>
              </Label>
              <Input type="password" autoFocus value={password} onChange={(e) => setPassword(e.target.value)} placeholder="PDF password" />
            </div>
            <Button variant="primary" type="submit" loading={upload.isPending}>
              Unlock & import
            </Button>
            <Button type="button" variant="ghost" onClick={() => (setLocked(null), setPassword(""))}>
              Cancel
            </Button>
          </form>
        </Card>
      )}
      {report && <ImportReport batch={report} onClose={() => setReport(null)} />}
      {uploadErr && <ImportError err={uploadErr} onClose={() => setUploadErr(null)} />}

      <Card className="mb-4 p-3">
        <div className="flex flex-wrap gap-2 [&>select]:w-full [&>select]:sm:w-[160px]">
          <div className="relative min-w-[220px] flex-1">
            <Search className="absolute left-3 top-3 size-4 text-muted" />
            <Input className="pl-9" placeholder="Search merchant or description…" value={q} onChange={(e) => (setQ(e.target.value), setPage(1))} />
          </div>
          <Select value={category} onChange={(e) => (setCategory(e.target.value), setPage(1))}>
            <option value="">All categories</option>
            {meta?.categories.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </Select>
          <Select value={source} onChange={(e) => (setSource(e.target.value), setPage(1))}>
            <option value="">Any source</option>
            {Object.entries(data?.sources ?? {}).map(([s, n]) => (
              <option key={s} value={s}>
                {s} ({n})
              </option>
            ))}
          </Select>
          <Select value={channel} onChange={(e) => (setChannel(e.target.value), setPage(1))} aria-label="Payment channel">
            <option value="">Any channel</option>
            {Object.entries(CHANNELS).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </Select>
          <div className="flex h-10 shrink-0 overflow-hidden rounded-xl border border-line text-[12.5px]" role="group" aria-label="Debit or credit">
            {(
              [
                ["", "All"],
                ["debit", "Debits"],
                ["credit", "Credits"],
              ] as const
            ).map(([k, label]) => (
              <button
                key={k || "all"}
                onClick={() => (setDirection(k), setPage(1))}
                className={cn("px-3 transition", direction === k ? "bg-brand text-white" : "bg-surface-2/60 text-muted hover:text-ink")}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="flex gap-1">
            {(
              [
                ["anomalies", "Anomalies", TriangleAlert],
                ["review", "Needs review", TriangleAlert],
                ["recurring", "Recurring", Repeat],
                ["impulse", "Impulse", Zap],
              ] as const
            ).map(([k, label, Icon]) => (
              <Button key={k} size="sm" variant={flag === k ? "primary" : "secondary"} className="h-10" onClick={() => (setFlag(flag === k ? "" : k), setPage(1))}>
                <Icon className="size-3.5" />
                <span className="hidden xl:inline">{label}</span>
              </Button>
            ))}
          </div>
        </div>
      </Card>

      {data?.totals && (
        <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-1 px-1 text-[12.5px] text-muted">
          <span>
            <b className="text-ink">{data.total.toLocaleString()}</b> transactions
          </span>
          <span className="inline-flex items-center gap-1">
            <ArrowUpRight className="size-3.5 text-red-500" /> {data.totals.debit_count} debits ·{" "}
            <b className="num text-ink">{money(data.totals.debits)}</b> out
          </span>
          <span className="inline-flex items-center gap-1">
            <ArrowDownLeft className="size-3.5 text-emerald-500" /> {data.totals.credit_count} credits ·{" "}
            <b className="num text-ink">{money(data.totals.credits)}</b> in
          </span>
        </div>
      )}

      {selected.size > 0 && (
        <Card className="sticky top-2 z-20 mb-3 flex flex-wrap items-center gap-2 border-brand/40 p-2.5 text-[13px] shadow-lg">
          <b className="px-1">{selected.size} selected</b>
          <Button size="sm" variant="primary" loading={bulk.isPending && !bulkCategory} onClick={() => bulk.mutate({ ids: [...selected], approve: true, remember })}>
            <Check className="size-3.5" /> Approve categories
          </Button>
          <div className="flex items-center gap-1.5">
            <Select className="h-8 w-[190px] text-[12.5px]" value={bulkCategory} onChange={(e) => setBulkCategory(e.target.value)} aria-label="Move to category">
              <option value="">Move to category…</option>
              {meta?.categories.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </Select>
            <Button size="sm" disabled={!bulkCategory} loading={bulk.isPending && !!bulkCategory} onClick={() => bulk.mutate({ ids: [...selected], category: bulkCategory, remember })}>
              Move
            </Button>
          </div>
          <label className="flex cursor-pointer items-center gap-1.5 text-muted">
            <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} className="accent-[var(--color-brand,#10b981)]" />
            remember these merchants for future imports
          </label>
          <Button size="sm" variant="ghost" className="ml-auto" onClick={() => setSelected(new Set())}>
            <X className="size-3.5" /> Clear
          </Button>
        </Card>
      )}
      <Card className="overflow-hidden">
        {isLoading ? (
          <div className="space-y-2 p-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-10" />
            ))}
          </div>
        ) : !data?.items.length ? (
          <Empty icon={<Receipt className="size-6" />} title="No transactions match">
            Try clearing filters, or upload a statement.
          </Empty>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[900px] text-[13px]">
              <thead className="bg-surface-2/60 text-left text-[12px] text-muted">
                <tr>
                  <th className="w-10 pl-4">
                    <input
                      type="checkbox"
                      aria-label="Select all on this page"
                      checked={allOnPage}
                      onChange={() =>
                        setSelected((prev) => {
                          const next = new Set(prev);
                          for (const id of pageIds) {
                            if (allOnPage) next.delete(id);
                            else next.add(id);
                          }
                          return next;
                        })
                      }
                    />
                  </th>
                  <th className="px-2 py-2.5 font-medium">Date</th>
                  <th className="font-medium">Merchant</th>
                  <th className="font-medium">Type</th>
                  <th className="font-medium">Category</th>
                  <th className="font-medium">How</th>
                  <th className="pr-4 text-right font-medium">Amount</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((t) => (
                  <tr key={t.id} className={cn("border-t border-line hover:bg-surface-2/40", selected.has(t.id) && "bg-brand/5")}>
                    <td className="pl-4">
                      <input type="checkbox" aria-label={`Select ${t.merchant}`} checked={selected.has(t.id)} onChange={() => toggle(t.id)} />
                    </td>
                    <td className="whitespace-nowrap px-2 py-2.5 text-muted">
                      {date(t.date)}
                      {t.time && <span className="ml-1 text-[11px] opacity-70">{t.time}</span>}
                    </td>
                    <td className="max-w-[360px] py-2.5">
                      <div className="flex items-center gap-1.5 font-medium">
                        {t.merchant}
                        {t.is_recurring && <Repeat className="size-3.5 text-violet-500" aria-label="recurring" />}
                        {t.is_anomaly && (
                          <Badge tone="red" title={t.anomaly_reason ?? ""}>
                            anomaly
                          </Badge>
                        )}
                        {t.tags?.includes("impulse") && <Badge tone="amber">impulse</Badge>}
                      </div>
                      <div className="truncate text-[11.5px] text-muted" title={t.description}>
                        {t.description}
                      </div>
                    </td>
                    <td className="whitespace-nowrap py-2.5">
                      <Badge tone={t.direction === "credit" ? "green" : "red"} title={t.direction === "credit" ? "Money in" : "Money out"}>
                        {t.direction === "credit" ? <ArrowDownLeft className="size-3" /> : <ArrowUpRight className="size-3" />}
                        {t.direction === "credit" ? "Credit" : "Debit"}
                      </Badge>
                      <div className="mt-0.5 text-[11px] text-muted" title={t.channel_label}>
                        {CHANNELS[t.channel] ?? t.channel}
                      </div>
                    </td>
                    <td className="py-2.5">
                      <div className="group relative inline-flex items-center">
                        <CategoryPill id={t.category} label={t.category_label} />
                        <select
                          aria-label="Change category"
                          className="absolute inset-0 cursor-pointer opacity-0"
                          value={t.category ?? ""}
                          onChange={(e) => recat.mutate({ t, category: e.target.value })}
                        >
                          {meta?.categories.map((c) => (
                            <option key={c.id} value={c.id}>
                              {c.label}
                            </option>
                          ))}
                        </select>
                      </div>
                    </td>
                    <td className="py-2.5">
                      <button onClick={() => setHistoryOf(t)} className="mr-1 rounded p-0.5 align-middle text-muted hover:bg-surface-2 hover:text-ink" aria-label="Category history" title="Why this category? (history)">
                        <History className="size-3.5" />
                      </button>
                      <SourceBadge source={t.category_source} confidence={t.category_confidence} reason={t.category_reason} />
                      {t.needs_review && <Badge tone="amber" title={t.category_reason ?? undefined}>Needs review</Badge>}
                      <Badge>{t.transaction_kind.replace(/_/g, " ")}</Badge>
                    </td>
                    <td className={cn("num whitespace-nowrap pr-4 text-right font-medium", t.amount > 0 && "text-emerald-500")}>
                      {money(t.amount, { sign: true })}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {data && data.total > data.page_size && (
          <div className="flex items-center justify-between border-t border-line px-4 py-3 text-[13px] text-muted">
            <span>
              {data.total.toLocaleString()} transactions · page {page} of {pages}
            </span>
            <div className="flex gap-1">
              <Button size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)} aria-label="Previous">
                <ChevronLeft className="size-4" />
              </Button>
              <Button size="sm" disabled={page >= pages} onClick={() => setPage(page + 1)} aria-label="Next">
                <ChevronRight className="size-4" />
              </Button>
            </div>
          </div>
        )}
      </Card>
      <AddTxn open={adding} onClose={() => setAdding(false)} />
      <HistoryDrawer
        txn={historyOf}
        onClose={() => setHistoryOf(null)}
        onRule={(t) => {
          setHistoryOf(null);
          setRuleDraft({ ...EMPTY_RULE, terms: t.merchant.toUpperCase(), field: "merchant", category: t.category ?? "" });
        }}
      />
      <RuleDrawer draft={ruleDraft} onClose={() => setRuleDraft(null)} categories={meta?.categories ?? []} />
    </div>
  );
}

function AddTxn({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { data: meta } = useMeta();
  const refresh = useRefreshAll();
  const [f, setF] = useState({ date: new Date().toISOString().slice(0, 10), description: "", amount: 0, out: true, category: "" });
  const save = useMutation({
    mutationFn: () =>
      api.post<{ transaction: Txn }>("/transactions", {
        date: f.date,
        description: f.description,
        amount: f.out ? -Math.abs(f.amount) : Math.abs(f.amount),
        category: f.category || null,
      }),
    onSuccess: (r) => {
      toast.success("Added", { description: `Categorised as ${r.transaction.category_label} (${r.transaction.category_source})` });
      refresh();
      onClose();
    },
    onError: (e: Error) => toast.error(e.message),
  });
  return (
    <Drawer open={open} onClose={onClose} title="Log a transaction">
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (!f.description || !f.amount) return toast.error("Description and amount are required");
          save.mutate();
        }}
      >
        <div>
          <Label>Description / merchant</Label>
          <Input value={f.description} placeholder="e.g. Starbucks Indiranagar" onChange={(e) => setF({ ...f, description: e.target.value })} />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label>Amount</Label>
            <Input type="number" min={0} value={f.amount || ""} onChange={(e) => setF({ ...f, amount: Number(e.target.value) })} />
          </div>
          <div>
            <Label>Date</Label>
            <Input type="date" value={f.date} onChange={(e) => setF({ ...f, date: e.target.value })} />
          </div>
        </div>
        <div className="flex items-center justify-between rounded-xl border border-line p-3 text-sm">
          <span>{f.out ? "Money out (spending)" : "Money in (income)"}</span>
          <Toggle checked={f.out} onChange={(v) => setF({ ...f, out: v })} label="Direction" />
        </div>
        <div>
          <Label hint="(leave empty to let the Categorizer decide)">Category</Label>
          <Select value={f.category} onChange={(e) => setF({ ...f, category: e.target.value })}>
            <option value="">Auto-categorise</option>
            {meta?.categories.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </Select>
        </div>
        <Button variant="primary" type="submit" loading={save.isPending} className="w-full">
          Add & re-run agents
        </Button>
      </form>
    </Drawer>
  );
}
