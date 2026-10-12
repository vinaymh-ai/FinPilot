import { CircleCheck, Info, LoaderCircle, TriangleAlert, X, OctagonAlert } from "lucide-react";
import { useEffect, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes } from "react";
import { cn } from "../lib/format";
import type { Severity } from "../lib/types";

export function Card({ className, children, ...rest }: { className?: string; children: ReactNode } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-line bg-surface shadow-[0_1px_0_rgba(0,0,0,0.02),0_8px_24px_-12px_rgba(15,23,42,0.12)]",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
  icon,
  action,
  className,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-start justify-between gap-3 px-5 pt-5", className)}>
      <div className="flex min-w-0 items-start gap-3">
        {icon && (
          <div className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg bg-surface-2 text-brand">{icon}</div>
        )}
        <div className="min-w-0">
          <h3 className="text-[15px] font-semibold tracking-tight">{title}</h3>
          {subtitle && <p className="mt-0.5 text-[13px] text-muted">{subtitle}</p>}
        </div>
      </div>
      {action}
    </div>
  );
}

type Variant = "primary" | "secondary" | "ghost" | "danger";
export function Button({
  variant = "secondary",
  size = "md",
  loading,
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: "sm" | "md"; loading?: boolean }) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-xl font-medium transition active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50",
        size === "sm" ? "h-8 px-3 text-[13px]" : "h-10 px-4 text-sm",
        variant === "primary" &&
          "bg-gradient-to-r from-emerald-500 to-indigo-500 text-white shadow-lg shadow-emerald-500/20 hover:brightness-110",
        variant === "secondary" && "border border-line bg-surface text-ink hover:bg-surface-2",
        variant === "ghost" && "text-muted hover:bg-surface-2 hover:text-ink",
        variant === "danger" && "border border-red-500/30 bg-red-500/10 text-red-500 hover:bg-red-500/20",
        className,
      )}
      disabled={loading || rest.disabled}
      {...rest}
    >
      {loading && <LoaderCircle className="size-4 animate-spin" />}
      {children}
    </button>
  );
}

const TONES = {
  neutral: "bg-surface-2 text-muted border-line",
  green: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
  red: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20",
  amber: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
  blue: "bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/20",
  violet: "bg-violet-500/10 text-violet-600 dark:text-violet-400 border-violet-500/20",
  cyan: "bg-cyan-500/10 text-cyan-700 dark:text-cyan-300 border-cyan-500/25",
} as const;
export type Tone = keyof typeof TONES;

export function Badge({ tone = "neutral", children, className, title }: { tone?: Tone; children: ReactNode; className?: string; title?: string }) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-medium",
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function severityTone(s: Severity | string): Tone {
  return s === "critical" ? "red" : s === "warning" ? "amber" : s === "positive" ? "green" : "blue";
}

export function SeverityIcon({ severity, className }: { severity: Severity | string; className?: string }) {
  const cls = cn("size-4 shrink-0", className);
  if (severity === "critical") return <OctagonAlert className={cn(cls, "text-red-500")} />;
  if (severity === "warning") return <TriangleAlert className={cn(cls, "text-amber-500")} />;
  if (severity === "positive") return <CircleCheck className={cn(cls, "text-emerald-500")} />;
  return <Info className={cn(cls, "text-sky-500")} />;
}

export function Stat({
  label,
  value,
  hint,
  icon,
  tone,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  icon?: ReactNode;
  tone?: "up" | "down" | "neutral";
}) {
  return (
    <Card className="p-5">
      <div className="flex items-center justify-between text-[13px] text-muted">
        <span>{label}</span>
        {icon && <span className="text-muted/80">{icon}</span>}
      </div>
      <div className="num mt-2 text-2xl font-semibold tracking-tight">{value}</div>
      {hint && (
        <div
          className={cn(
            "mt-1 text-[12.5px]",
            tone === "up" ? "text-emerald-500" : tone === "down" ? "text-red-500" : "text-muted",
          )}
        >
          {hint}
        </div>
      )}
    </Card>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton rounded-xl", className)} />;
}

export function Empty({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-12 text-center">
      {icon && <div className="grid size-12 place-items-center rounded-2xl bg-surface-2 text-muted">{icon}</div>}
      <div className="font-medium">{title}</div>
      {children && <div className="max-w-md text-sm text-muted">{children}</div>}
    </div>
  );
}

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "h-10 w-full rounded-xl border border-line bg-surface px-3 text-sm outline-none transition placeholder:text-muted/70 focus:border-brand focus:ring-2 focus:ring-brand/20",
        className,
      )}
      {...rest}
    />
  );
}

export function Select({ className, children, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        "h-10 w-full rounded-xl border border-line bg-surface px-3 text-sm outline-none transition focus:border-brand focus:ring-2 focus:ring-brand/20",
        className,
      )}
      {...rest}
    >
      {children}
    </select>
  );
}

export function Label({ children, hint }: { children: ReactNode; hint?: ReactNode }) {
  return (
    <label className="mb-1.5 block text-[12.5px] font-medium text-muted">
      {children}
      {hint && <span className="ml-1 font-normal opacity-80">{hint}</span>}
    </label>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label?: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative h-6 w-11 shrink-0 rounded-full transition",
        checked ? "bg-emerald-500" : "bg-line",
      )}
    >
      <span
        className={cn(
          "absolute top-0.5 size-5 rounded-full bg-white shadow transition-all",
          checked ? "left-[22px]" : "left-0.5",
        )}
      />
    </button>
  );
}

export function Progress({ value, tone = "brand", className }: { value: number; tone?: "brand" | "amber" | "red"; className?: string }) {
  const v = Math.max(0, Math.min(100, value));
  return (
    <div className={cn("h-2 w-full overflow-hidden rounded-full bg-surface-2", className)}>
      <div
        className={cn(
          "h-full rounded-full transition-all",
          tone === "brand" && "bg-gradient-to-r from-emerald-500 to-indigo-500",
          tone === "amber" && "bg-amber-500",
          tone === "red" && "bg-red-500",
        )}
        style={{ width: `${v}%` }}
      />
    </div>
  );
}

export function Tabs<T extends string>({
  value,
  onChange,
  items,
}: {
  value: T;
  onChange: (v: T) => void;
  items: { value: T; label: ReactNode }[];
}) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-surface-2 p-1">
      {items.map((it) => (
        <button
          key={it.value}
          onClick={() => onChange(it.value)}
          className={cn(
            "rounded-lg px-3 py-1.5 text-[13px] font-medium transition",
            value === it.value ? "bg-surface text-ink shadow-sm" : "text-muted hover:text-ink",
          )}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}

export function Drawer({
  open,
  onClose,
  title,
  children,
  wide,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
  wide?: boolean;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <div
        className={cn(
          "relative flex h-full w-full flex-col border-l border-line bg-surface shadow-2xl",
          wide ? "max-w-3xl" : "max-w-xl",
        )}
      >
        <div className="flex items-center justify-between border-b border-line px-5 py-4">
          <div className="min-w-0 font-semibold">{title}</div>
          <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close">
            <X className="size-4" />
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto p-5">{children}</div>
      </div>
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <LoaderCircle className={cn("size-4 animate-spin text-muted", className)} />;
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function ModelBadge({ model }: { model: string | null | undefined }) {
  if (!model) return null;
  const isRules = model === "rules";
  if (/jev/i.test(model)) {
    return (
      <Badge tone="cyan" title={`System One decision via ${model} — typed answer with calibrated probabilities, no text generation`}>
        {model.replace(/^mesh:/, "mesh · ").replace("typesafe/", "")}
      </Badge>
    );
  }
  return (
    <Badge tone={isRules ? "neutral" : "violet"} title={isRules ? "Deterministic (no LLM)" : `Generated via ${model}`}>
      {isRules ? "rules" : model.replace(/^mesh:/, "mesh · ")}
    </Badge>
  );
}

/** Top-k probability bars for a typed (Choice) answer. */
export function ProbabilityBars({ probs, k = 3, highlight }: { probs: Record<string, number>; k?: number; highlight?: string }) {
  const rows = Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, k);
  return (
    <div className="space-y-1">
      {rows.map(([key, p]) => (
        <div key={key} className="flex items-center gap-2 text-[11.5px]">
          <span className={cn("w-32 truncate", key === highlight ? "font-medium text-ink" : "text-muted")} title={key}>
            {key.replace(/_/g, " ")}
          </span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
            <div className={cn("h-full rounded-full", key === highlight ? "bg-cyan-500" : "bg-muted/40")} style={{ width: `${Math.max(2, p * 100)}%` }} />
          </div>
          <span className="num w-10 text-right text-muted">{Math.round(p * 100)}%</span>
        </div>
      ))}
    </div>
  );
}
