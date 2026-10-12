import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CalendarCheck, Pencil, Plus, Target, Trash2, TrendingDown, TrendingUp } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { Sparkline } from "../components/charts";
import { GoalStatusBadge, goalTone } from "../components/domain";
import { Button, Card, Drawer, Empty, Input, Label, PageHeader, Progress, Skeleton } from "../components/ui";
import { api } from "../lib/api";
import { date, money, prob } from "../lib/format";
import { useRunAgents } from "../lib/hooks";
import type { Goal } from "../lib/types";

type Form = { id?: number; name: string; target_amount: number; saved_amount: number; deadline: string; priority: number };
const EMPTY: Form = { name: "", target_amount: 100000, saved_amount: 0, deadline: "", priority: 1 };

export default function Goals() {
  const qc = useQueryClient();
  const run = useRunAgents();
  const { data, isLoading } = useQuery({ queryKey: ["goals"], queryFn: () => api.get<{ goals: Goal[] }>("/goals") });
  const [form, setForm] = useState<Form | null>(null);
  const save = useMutation({
    mutationFn: (f: Form) => {
      const { id, ...body } = f;
      return id ? api.patch(`/goals/${id}`, body) : api.post("/goals", body);
    },
    onSuccess: () => {
      setForm(null);
      toast.success("Goal saved — re-running the forecast");
      qc.invalidateQueries({ queryKey: ["goals"] });
      run.mutate({});
    },
    onError: (e: Error) => toast.error(e.message),
  });
  const del = useMutation({
    mutationFn: (id: number) => api.del(`/goals/${id}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["goals"] });
      toast.success("Goal deleted");
    },
  });

  return (
    <div>
      <PageHeader
        title="Goals"
        subtitle="Each goal is funded in priority order from your projected balance above the safety buffer. Probabilities come from 400 simulated futures and update every run."
        actions={
          <Button variant="primary" onClick={() => setForm({ ...EMPTY })}>
            <Plus className="size-4" /> New goal
          </Button>
        }
      />
      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2">
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
        </div>
      ) : !data?.goals.length ? (
        <Card>
          <Empty icon={<Target className="size-6" />} title="No goals yet">
            Add a savings goal (emergency fund, trip, laptop) and FinPilot will tell you if you're on track — and what to
            change if you're not.
          </Empty>
        </Card>
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          {data.goals.map((g) => {
            const p = g.projection;
            const probs = g.history.map((h) => h.probability);
            const trend = probs.length >= 2 ? probs[probs.length - 1] - probs[probs.length - 2] : 0;
            return (
              <Card key={g.id} className="p-5">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-lg font-semibold">{g.name}</span>
                      <GoalStatusBadge status={p?.status} />
                    </div>
                    <div className="mt-0.5 text-[13px] text-muted">
                      Priority {g.priority} · deadline {date(g.deadline, true)}
                    </div>
                  </div>
                  <div className="flex gap-1">
                    <Button size="sm" variant="ghost" onClick={() => setForm({ ...g })} aria-label="Edit">
                      <Pencil className="size-4" />
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => confirm(`Delete "${g.name}"?`) && del.mutate(g.id)} aria-label="Delete">
                      <Trash2 className="size-4" />
                    </Button>
                  </div>
                </div>

                <div className="mt-5 grid grid-cols-3 gap-3">
                  <div>
                    <div className="text-[12px] text-muted">Saved</div>
                    <div className="num font-semibold">{money(g.saved_amount)}</div>
                  </div>
                  <div>
                    <div className="text-[12px] text-muted">Target</div>
                    <div className="num font-semibold">{money(g.target_amount)}</div>
                  </div>
                  <div>
                    <div className="text-[12px] text-muted">Chance on time</div>
                    <div className="num flex items-center gap-1 font-semibold">
                      {prob(p?.probability)}
                      {trend !== 0 &&
                        (trend > 0 ? <TrendingUp className="size-4 text-emerald-500" /> : <TrendingDown className="size-4 text-red-500" />)}
                    </div>
                  </div>
                </div>

                <div className="mt-4">
                  <div className="mb-1 flex justify-between text-[12px] text-muted">
                    <span>saved {g.progress_pct}%</span>
                    {p && <span>projected at deadline {money(p.projected_at_deadline, { compact: true })}</span>}
                  </div>
                  <div className="relative">
                    <Progress value={p ? (p.projected_at_deadline / g.target_amount) * 100 : 0} tone={goalTone(p?.status)} className="opacity-35" />
                    <div className="absolute inset-0">
                      <Progress value={g.progress_pct} tone={goalTone(p?.status)} className="bg-transparent" />
                    </div>
                  </div>
                </div>

                {p && (
                  <div className="mt-4 rounded-xl bg-surface-2/70 p-3 text-[13px]">
                    {p.status === "achieved" ? (
                      "🎉 Funded."
                    ) : p.shortfall > 0 ? (
                      <>
                        At your current pace you'll be <b className="num">{money(p.shortfall)}</b> short. Setting aside{" "}
                        <b className="num">{money(p.required_monthly)}/month</b> more closes the gap
                        {p.eta ? (
                          <>
                            ; otherwise you'd get there around <b>{date(p.eta, true)}</b>.
                          </>
                        ) : (
                          "."
                        )}
                      </>
                    ) : (
                      <span className="flex items-center gap-1.5">
                        <CalendarCheck className="size-4 text-emerald-500" />
                        Funded by {date(p.eta, true)} at this pace — {prob(p.probability)} likely by the deadline.
                      </span>
                    )}
                  </div>
                )}

                {probs.length > 1 && (
                  <div className="mt-3">
                    <div className="text-[11.5px] text-muted">Probability across the last {probs.length} runs</div>
                    <Sparkline values={probs} color={trend >= 0 ? "#10b981" : "#ef4444"} />
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}

      <Drawer open={!!form} onClose={() => setForm(null)} title={form?.id ? "Edit goal" : "New goal"}>
        {form && (
          <form
            className="space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              if (!form.name || !form.deadline) return toast.error("Name and deadline are required");
              save.mutate(form);
            }}
          >
            <div>
              <Label>Name</Label>
              <Input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="Emergency fund" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label>Target amount</Label>
                <Input type="number" min={1} value={form.target_amount} onChange={(e) => setForm({ ...form, target_amount: Number(e.target.value) })} />
              </div>
              <div>
                <Label hint="(already set aside)">Saved so far</Label>
                <Input type="number" min={0} value={form.saved_amount} onChange={(e) => setForm({ ...form, saved_amount: Number(e.target.value) })} />
              </div>
              <div>
                <Label>Deadline</Label>
                <Input type="date" value={form.deadline} onChange={(e) => setForm({ ...form, deadline: e.target.value })} />
              </div>
              <div>
                <Label hint="(1 = funded first)">Priority</Label>
                <Input type="number" min={1} max={10} value={form.priority} onChange={(e) => setForm({ ...form, priority: Number(e.target.value) })} />
              </div>
            </div>
            <Button variant="primary" type="submit" loading={save.isPending} className="w-full">
              Save & re-forecast
            </Button>
          </form>
        )}
      </Drawer>
    </div>
  );
}
