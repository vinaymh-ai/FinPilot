import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ListChecks, Pencil, Plus, Tags, Trash2, TriangleAlert, Wand2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { CategoryPill } from "../components/domain";
import { Badge, Button, Card, CardHeader, Drawer, Empty, Input, Label, PageHeader, Select, Skeleton, Toggle } from "../components/ui";
import { api } from "../lib/api";
import { cn, money } from "../lib/format";
import type { Category, CategoryRule, Txn } from "../lib/types";

type Overview = {
  categories: Category[];
  rules: CategoryRule[];
  needs_review: number;
  parents: { id: string; label: string; group: string }[];
};
type Preview = { matches: number; would_change: number; total: number; merchants: string[]; sample: Txn[]; description: string };
export type RuleDraft = {
  id?: number;
  category: string;
  terms: string;
  field: CategoryRule["field"];
  match: CategoryRule["match"];
  direction: "" | "debit" | "credit";
  min_amount: string;
  max_amount: string;
  apply_to_existing: boolean;
  include_manual: boolean;
  enabled: boolean;
};

const GROUPS: [string, string][] = [
  ["essential", "Essentials"],
  ["discretionary", "Lifestyle"],
  ["financial", "Financial commitments"],
  ["income", "Income"],
  ["other", "Transfers & other"],
];
const COLORS = ["#6366f1", "#0ea5e9", "#14b8a6", "#22c55e", "#eab308", "#f97316", "#ef4444", "#ec4899", "#a855f7", "#64748b"];
const ICONS = ["tag", "home", "heart", "baby", "dog", "car", "graduation-cap", "briefcase", "gift", "users", "wrench", "plane"];

function useDebounced<T>(value: T, ms = 350): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

function splitTerms(text: string): string[] {
  return text
    .split(/[,\n]/)
    .map((t) => t.trim())
    .filter(Boolean);
}

function LabeledToggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <div className="flex items-center gap-2 text-[13px]">
      <Toggle checked={checked} onChange={onChange} label={label} />
      <span className="cursor-pointer select-none" onClick={() => onChange(!checked)}>
        {label}
      </span>
    </div>
  );
}

function RulePreview({ body }: { body: Record<string, unknown> | null }) {
  const debounced = useDebounced(body);
  const { data, isFetching, error } = useQuery({
    queryKey: ["rule-preview", JSON.stringify(debounced)],
    queryFn: () => api.post<Preview>("/category-rules/preview", debounced!),
    enabled: !!debounced,
    retry: false,
  });
  if (!body) return <div className="text-[12.5px] text-muted">Type a word to see which transactions it matches.</div>;
  if (error) return <div className="text-[12.5px] text-red-500">{(error as Error).message}</div>;
  if (!data) return <Skeleton className="h-16" />;
  return (
    <div className={cn("space-y-2 transition", isFetching && "opacity-60")}>
      <div className="text-[13px]">
        Matches <b>{data.matches}</b> transaction{data.matches === 1 ? "" : "s"} ({money(data.total)}) — <b>{data.would_change}</b> would change category.
      </div>
      {data.merchants.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {data.merchants.map((m) => (
            <Badge key={m}>{m}</Badge>
          ))}
        </div>
      )}
      {data.sample.length > 0 && (
        <div className="max-h-56 overflow-y-auto rounded-xl border border-line">
          {data.sample.slice(0, 12).map((t) => (
            <div key={t.id} className="flex items-center justify-between gap-3 border-b border-line px-3 py-1.5 text-[12px] last:border-0">
              <div className="min-w-0">
                <div className="truncate font-medium">{t.merchant}</div>
                <div className="truncate text-muted">{t.description}</div>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <CategoryPill id={t.category} label={t.category_label} />
                <span className={cn("num", t.amount > 0 && "text-emerald-500")}>{money(t.amount, { sign: true })}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function RuleDrawer({ draft, onClose, categories }: { draft: RuleDraft | null; onClose: () => void; categories: Category[] }) {
  const qc = useQueryClient();
  const [f, setF] = useState<RuleDraft | null>(draft);
  useEffect(() => setF(draft), [draft]);
  const terms = f ? splitTerms(f.terms) : [];
  const body =
    f && terms.length && f.category
      ? {
          category: f.category,
          terms,
          field: f.field,
          match: f.match,
          direction: f.direction || null,
          min_amount: f.min_amount ? Number(f.min_amount) : null,
          max_amount: f.max_amount ? Number(f.max_amount) : null,
          apply_to_existing: f.apply_to_existing,
          include_manual: f.include_manual,
          enabled: f.enabled,
        }
      : null;
  const save = useMutation({
    mutationFn: () => (f?.id ? api.patch<{ applied: number }>(`/category-rules/${f.id}`, body) : api.post<{ applied: number }>("/category-rules", body)),
    onSuccess: (r) => {
      toast.success(f?.id ? "Rule updated" : "Rule created", { description: `${r.applied} transaction(s) recategorised. Future imports follow it too.` });
      qc.invalidateQueries();
      onClose();
    },
    onError: (e: Error) => toast.error(e.message),
  });
  if (!f) return null;
  const set = (patch: Partial<RuleDraft>) => setF({ ...f, ...patch });
  return (
    <Drawer open={!!draft} onClose={onClose} title={f.id ? "Edit rule" : "New categorization rule"} wide>
      <div className="space-y-4">
        <p className="text-[13px] text-muted">
          Rules run before every other categorizer stage — your word is final. They apply to matching transactions now and to every future import.
        </p>
        <div>
          <Label hint="comma-separated; any of them matches">When the</Label>
          <div className="grid gap-2 sm:grid-cols-[160px_160px_1fr]">
            <Select value={f.field} onChange={(e) => set({ field: e.target.value as RuleDraft["field"] })} aria-label="Field">
              <option value="any">payee or narration</option>
              <option value="merchant">payee name</option>
              <option value="description">bank narration</option>
            </Select>
            <Select value={f.match} onChange={(e) => set({ match: e.target.value as RuleDraft["match"] })} aria-label="Match">
              <option value="contains">contains</option>
              <option value="word">has the word</option>
              <option value="starts_with">starts with</option>
              <option value="equals">is exactly</option>
            </Select>
            <Input autoFocus value={f.terms} onChange={(e) => set({ terms: e.target.value })} placeholder="e.g. KAMALA, HOUSE HELP" />
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <div>
            <Label>Direction</Label>
            <Select value={f.direction} onChange={(e) => set({ direction: e.target.value as RuleDraft["direction"] })}>
              <option value="">Debits and credits</option>
              <option value="debit">Debits only</option>
              <option value="credit">Credits only</option>
            </Select>
          </div>
          <div>
            <Label>Min amount</Label>
            <Input type="number" min={0} value={f.min_amount} onChange={(e) => set({ min_amount: e.target.value })} placeholder="any" />
          </div>
          <div>
            <Label>Max amount</Label>
            <Input type="number" min={0} value={f.max_amount} onChange={(e) => set({ max_amount: e.target.value })} placeholder="any" />
          </div>
        </div>
        <div>
          <Label>Then categorise as</Label>
          <Select value={f.category} onChange={(e) => set({ category: e.target.value })}>
            <option value="">Choose a category…</option>
            {categories.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
                {c.custom ? " (yours)" : ""}
              </option>
            ))}
          </Select>
        </div>
        <div className="flex flex-wrap gap-x-6 gap-y-2">
          <LabeledToggle checked={f.apply_to_existing} onChange={(v) => set({ apply_to_existing: v })} label="Apply to existing transactions" />
          <LabeledToggle checked={f.include_manual} onChange={(v) => set({ include_manual: v })} label="Also override my one-off edits" />
          <LabeledToggle checked={f.enabled} onChange={(v) => set({ enabled: v })} label="Enabled" />
        </div>
        <Card className="p-3">
          <div className="mb-2 text-[12px] font-medium uppercase tracking-wide text-muted">Preview</div>
          <RulePreview body={body} />
        </Card>
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" disabled={!body} loading={save.isPending} onClick={() => save.mutate()}>
            {f.id ? "Save rule" : "Create rule"}
          </Button>
        </div>
      </div>
    </Drawer>
  );
}

type CatDraft = { id?: string; label: string; parent: string; color: string; icon: string; description: string; keywords: string };

function CategoryDrawer({ draft, onClose, parents }: { draft: CatDraft | null; onClose: () => void; parents: Overview["parents"] }) {
  const qc = useQueryClient();
  const [f, setF] = useState<CatDraft | null>(draft);
  useEffect(() => setF(draft), [draft]);
  const save = useMutation({
    mutationFn: () => {
      const body = { label: f!.label, parent: f!.parent, color: f!.color, icon: f!.icon, description: f!.description || null };
      return f!.id
        ? api.patch<{ category: Category }>(`/categories/${f!.id}`, body)
        : api.post<{ category: Category; applied: number }>("/categories", { ...body, keywords: splitTerms(f!.keywords) });
    },
    onSuccess: (r) => {
      const applied = (r as { applied?: number }).applied;
      toast.success(f?.id ? "Category updated" : `Created “${r.category.label}”`, {
        description: applied ? `${applied} matching transaction(s) moved into it.` : "Pick it from any transaction's category menu, or add a rule.",
      });
      qc.invalidateQueries();
      onClose();
    },
    onError: (e: Error) => toast.error(e.message),
  });
  if (!f) return null;
  const set = (patch: Partial<CatDraft>) => setF({ ...f, ...patch });
  const terms = splitTerms(f.keywords);
  return (
    <Drawer open={!!draft} onClose={onClose} title={f.id ? `Edit “${draft?.label}”` : "New category"} wide>
      <div className="space-y-4">
        <div>
          <Label>Name</Label>
          <Input autoFocus value={f.label} onChange={(e) => set({ label: e.target.value })} placeholder="e.g. Household help, Pet care, Kids' school" maxLength={60} />
        </div>
        <div>
          <Label hint="decides how analytics treat it — spending vs transfer, essential vs lifestyle">Behaves like</Label>
          <Select value={f.parent} onChange={(e) => set({ parent: e.target.value })}>
            {parents.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </Select>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <Label>Colour</Label>
            <div className="flex flex-wrap gap-1.5">
              {COLORS.map((c) => (
                <button
                  key={c}
                  onClick={() => set({ color: c })}
                  className={cn("size-7 rounded-full ring-offset-2 ring-offset-surface transition", f.color === c && "ring-2 ring-brand")}
                  style={{ background: c }}
                  aria-label={`Colour ${c}`}
                />
              ))}
            </div>
          </div>
          <div>
            <Label>Icon</Label>
            <Select value={f.icon} onChange={(e) => set({ icon: e.target.value })}>
              {ICONS.map((i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </Select>
          </div>
        </div>
        <div>
          <Label hint="the categorizer's model reads this to decide what belongs here">Description</Label>
          <Input value={f.description} onChange={(e) => set({ description: e.target.value })} placeholder="e.g. Salary for our maid and cook" maxLength={255} />
        </div>
        {!f.id && (
          <>
            <div>
              <Label hint="optional — creates a rule; comma-separated">Auto-file transactions whose payee or narration contains</Label>
              <Input value={f.keywords} onChange={(e) => set({ keywords: e.target.value })} placeholder="e.g. KAMALA, LAKSHMI" />
            </div>
            {terms.length > 0 && (
              <Card className="p-3">
                <div className="mb-2 text-[12px] font-medium uppercase tracking-wide text-muted">These would move into it</div>
                <RulePreview body={{ category: f.parent, terms, field: "any", match: "contains" }} />
              </Card>
            )}
          </>
        )}
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" disabled={!f.label.trim()} loading={save.isPending} onClick={() => save.mutate()}>
            {f.id ? "Save" : "Create category"}
          </Button>
        </div>
      </div>
    </Drawer>
  );
}

export const EMPTY_RULE: RuleDraft = {
  category: "",
  terms: "",
  field: "any",
  match: "contains",
  direction: "",
  min_amount: "",
  max_amount: "",
  apply_to_existing: true,
  include_manual: false,
  enabled: true,
};

export default function Categories() {
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({ queryKey: ["categories"], queryFn: () => api.get<Overview>("/categories") });
  const [catDraft, setCatDraft] = useState<CatDraft | null>(null);
  const [ruleDraft, setRuleDraft] = useState<RuleDraft | null>(null);
  const removeCat = useMutation({
    mutationFn: (c: Category) => api.del<{ moved_to: string; transactions: number }>(`/categories/${c.id}`),
    onSuccess: () => {
      toast.success("Category deleted", { description: "Its transactions and rules moved to the category it behaved like." });
      qc.invalidateQueries();
    },
  });
  const toggleRule = useMutation({
    mutationFn: (r: CategoryRule) =>
      api.patch(`/category-rules/${r.id}`, { ...r, enabled: !r.enabled, apply_to_existing: !r.enabled }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["categories"] }),
  });
  const removeRule = useMutation({
    mutationFn: (r: CategoryRule) => api.del(`/category-rules/${r.id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["categories"] }),
  });
  const cats = data?.categories ?? [];
  const custom = cats.filter((c) => c.custom);
  return (
    <div>
      <PageHeader
        title="Categories & rules"
        subtitle="Make the categorizer yours: add your own categories, teach it rules, and review what it wasn't sure about. Every change is logged per transaction."
        actions={
          <>
            <Button onClick={() => setRuleDraft({ ...EMPTY_RULE })}>
              <Wand2 className="size-4" /> New rule
            </Button>
            <Button
              variant="primary"
              onClick={() => setCatDraft({ label: "", parent: "other", color: COLORS[0], icon: "tag", description: "", keywords: "" })}
            >
              <Plus className="size-4" /> New category
            </Button>
          </>
        }
      />
      {!!data?.needs_review && (
        <Card className="mb-5 flex flex-wrap items-center justify-between gap-3 border-amber-500/30 p-4">
          <div className="flex items-center gap-2 text-[13.5px]">
            <TriangleAlert className="size-4 text-amber-500" />
            <span>
              <b>{data.needs_review}</b> transaction{data.needs_review === 1 ? "" : "s"} need your review — the categorizer wasn't confident, so it didn't guess silently.
            </span>
          </div>
          <Link to="/transactions?flag=review">
            <Button size="sm" variant="primary">
              <ListChecks className="size-4" /> Review now
            </Button>
          </Link>
        </Card>
      )}
      <div className="grid gap-5 xl:grid-cols-[1.2fr_1fr]">
        <Card>
          <CardHeader title="Categories" subtitle={`${cats.length - custom.length} built-in · ${custom.length} yours`} />
          {isLoading ? (
            <div className="space-y-2 p-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-9" />
              ))}
            </div>
          ) : (
            <div className="space-y-4 p-4 pt-0">
              {GROUPS.map(([g, title]) => {
                const items = cats.filter((c) => c.group === g);
                if (!items.length) return null;
                return (
                  <div key={g}>
                    <div className="mb-1.5 text-[11.5px] font-medium uppercase tracking-wide text-muted">{title}</div>
                    <div className="grid gap-1.5 sm:grid-cols-2">
                      {items.map((c) => (
                        <div key={c.id} className="flex items-center justify-between gap-2 rounded-xl border border-line px-3 py-2">
                          <div className="min-w-0">
                            <div className="flex items-center gap-2 text-[13px] font-medium">
                              <span className="size-2.5 shrink-0 rounded-full" style={{ background: c.color }} />
                              <span className="truncate">{c.label}</span>
                              {c.custom && <Badge tone="violet">yours</Badge>}
                            </div>
                            <div className="truncate text-[11.5px] text-muted">
                              {c.custom ? `behaves like ${c.parent_label}` : c.description}
                            </div>
                          </div>
                          <div className="flex shrink-0 items-center gap-1">
                            <Link to={`/transactions?category=${c.id}`} className="num text-right text-[11.5px] text-muted hover:text-brand" title="Show transactions">
                              {c.transactions ?? 0} txns
                              {c.spent ? <div>{money(c.spent)}</div> : null}
                            </Link>
                            {c.custom && (
                              <>
                                <button
                                  onClick={() =>
                                    setCatDraft({
                                      id: c.id,
                                      label: c.label,
                                      parent: c.parent ?? "other",
                                      color: c.color,
                                      icon: c.icon,
                                      description: c.description.startsWith("Your category") ? "" : c.description,
                                      keywords: "",
                                    })
                                  }
                                  className="rounded p-1 text-muted hover:bg-surface-2 hover:text-ink"
                                  aria-label={`Edit ${c.label}`}
                                >
                                  <Pencil className="size-3.5" />
                                </button>
                                <button
                                  onClick={() => window.confirm(`Delete “${c.label}”? Its transactions move to ${c.parent_label}.`) && removeCat.mutate(c)}
                                  className="rounded p-1 text-muted hover:bg-surface-2 hover:text-red-500"
                                  aria-label={`Delete ${c.label}`}
                                >
                                  <Trash2 className="size-3.5" />
                                </button>
                              </>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </Card>
        <Card>
          <CardHeader title="Your rules" subtitle="Checked first, in priority order — before the lexicon and the model" />
          {!data?.rules.length ? (
            <Empty icon={<Tags className="size-6" />} title="No rules yet">
              Create one — e.g. “narration contains KAMALA → Household help” — or ask in chat: <i>“mark Hanumanthaiah as rent”</i>.
            </Empty>
          ) : (
            <div className="divide-y divide-line">
              {data.rules.map((r) => (
                <div key={r.id} className="flex items-start justify-between gap-3 px-4 py-3">
                  <div className="min-w-0">
                    <div className={cn("text-[13px]", !r.enabled && "text-muted line-through")}>{r.description}</div>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11.5px] text-muted">
                      <CategoryPill id={r.category} label={r.category_label} />
                      <span>{r.hits} applied</span>
                      {r.source && r.source !== "user" && <Badge>{r.source}</Badge>}
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <Toggle checked={r.enabled} onChange={() => toggleRule.mutate(r)} />
                    <button
                      onClick={() =>
                        setRuleDraft({
                          id: r.id,
                          category: r.category,
                          terms: r.terms.join(", "),
                          field: r.field,
                          match: r.match,
                          direction: r.direction ?? "",
                          min_amount: r.min_amount?.toString() ?? "",
                          max_amount: r.max_amount?.toString() ?? "",
                          apply_to_existing: true,
                          include_manual: false,
                          enabled: r.enabled,
                        })
                      }
                      className="rounded p-1 text-muted hover:bg-surface-2 hover:text-ink"
                      aria-label="Edit rule"
                    >
                      <Pencil className="size-3.5" />
                    </button>
                    <button onClick={() => removeRule.mutate(r)} className="rounded p-1 text-muted hover:bg-surface-2 hover:text-red-500" aria-label="Delete rule">
                      <Trash2 className="size-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
      <CategoryDrawer draft={catDraft} onClose={() => setCatDraft(null)} parents={data?.parents ?? []} />
      <RuleDrawer draft={ruleDraft} onClose={() => setRuleDraft(null)} categories={cats} />
    </div>
  );
}
