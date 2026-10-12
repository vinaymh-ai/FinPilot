import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, FileUp, Plus, Receipt, Repeat, Search, TriangleAlert, Upload, Zap } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";
import { CategoryPill, SourceBadge } from "../components/domain";
import { Badge, Button, Card, Drawer, Empty, Input, Label, PageHeader, Select, Skeleton, Toggle } from "../components/ui";
import { api } from "../lib/api";
import { cn, date, money } from "../lib/format";
import { useMeta, useRefreshAll } from "../lib/hooks";
import type { Txn } from "../lib/types";

type Page = { total: number; page: number; page_size: number; items: Txn[]; sources: Record<string, number> };
type UploadResp = {
  batch: { imported: number; duplicates: number; invalid: number; total: number; errors: string[]; warnings: string[]; mapping: Record<string, string> };
  run: { run_id: number; summary: Record<string, any> } | null;
};

export default function Transactions() {
  const { data: meta } = useMeta();
  const qc = useQueryClient();
  const refresh = useRefreshAll();
  const [q, setQ] = useState("");
  const [category, setCategory] = useState("");
  const [source, setSource] = useState("");
  const [flag, setFlag] = useState<"" | "anomalies" | "recurring" | "impulse">("");
  const [page, setPage] = useState(1);
  const [adding, setAdding] = useState(false);
  const [drag, setDrag] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const params = new URLSearchParams({ page: String(page), page_size: "40" });
  if (q) params.set("q", q);
  if (category) params.set("category", category);
  if (source) params.set("source", source);
  if (flag) params.set(flag, "true");
  const { data, isLoading } = useQuery({
    queryKey: ["transactions", params.toString()],
    queryFn: () => api.get<Page>(`/transactions?${params}`),
    placeholderData: (prev) => prev,
  });

  const upload = useMutation({
    mutationFn: (file: File) => {
      const fd = new FormData();
      fd.append("file", file);
      return api.upload<UploadResp>("/transactions/upload", fd);
    },
    onSuccess: (r) => {
      toast.success(`Imported ${r.batch.imported} transactions`, {
        description: `${r.batch.duplicates} duplicates skipped · ${r.batch.invalid} invalid${r.run ? ` · agents ran (#${r.run.run_id})` : ""}`,
      });
      r.batch.warnings.forEach((w) => toast.warning(w));
      refresh();
    },
    onError: (e: Error) => toast.error("Upload failed", { description: e.message }),
  });

  const recat = useMutation({
    mutationFn: ({ t, category }: { t: Txn; category: string }) =>
      api.patch<{ updated: number }>(`/transactions/${t.id}`, { category, apply_to_merchant: true }),
    onSuccess: (r, v) => {
      toast.success(`Learned: ${v.t.merchant} → ${meta?.categories.find((c) => c.id === v.category)?.label}`, {
        description: `${r.updated} transaction(s) updated; future imports will use this.`,
      });
      qc.invalidateQueries({ queryKey: ["transactions"] });
    },
  });

  const onFiles = (files: FileList | null) => {
    const f = files?.[0];
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".csv")) return toast.error("Please upload a .csv file");
    upload.mutate(f);
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
              <Upload className="size-4" /> Upload CSV
            </Button>
            <input ref={fileRef} type="file" accept=".csv,text/csv" hidden onChange={(e) => onFiles(e.target.files)} />
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
          Drop a bank statement CSV here — HDFC/ICICI/SBI exports, Debit/Credit or signed Amount columns, dd/mm or mm/dd
          dates all work. Re-uploading the same file adds nothing.
        </span>
        <a href="/sample_transactions.csv" download className="text-brand hover:underline">
          sample file
        </a>
      </div>

      <Card className="mb-4 p-3">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-[1fr_180px_160px_auto]">
          <div className="relative">
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
          <div className="flex gap-1">
            {(
              [
                ["anomalies", "Anomalies", TriangleAlert],
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
                  <th className="px-4 py-2.5 font-medium">Date</th>
                  <th className="font-medium">Merchant</th>
                  <th className="font-medium">Category</th>
                  <th className="font-medium">How</th>
                  <th className="pr-4 text-right font-medium">Amount</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((t) => (
                  <tr key={t.id} className="border-t border-line hover:bg-surface-2/40">
                    <td className="whitespace-nowrap px-4 py-2.5 text-muted">
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
                      <SourceBadge source={t.category_source} confidence={t.category_confidence} reason={t.category_reason} />
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
