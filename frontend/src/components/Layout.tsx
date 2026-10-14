import {
  Activity,
  Bot,
  Compass,
  FlaskConical,
  Gauge,
  Inbox,
  LayoutDashboard,
  Lightbulb,
  LogOut,
  Menu,
  MessageCircle,
  Moon,
  Play,
  Receipt,
  Scale,
  Settings,
  Sparkles,
  Tags,
  Sun,
  Target,
  Workflow,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { cn } from "../lib/format";
import { useMeta, useRunAgents } from "../lib/hooks";
import { Badge, Button } from "./ui";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/advisor", label: "Advisor", icon: Lightbulb },
  { to: "/simulator", label: "What-if Lab", icon: FlaskConical },
  { to: "/goals", label: "Goals", icon: Target },
  { to: "/insights", label: "Insights", icon: Sparkles },
  { to: "/transactions", label: "Transactions", icon: Receipt },
  { to: "/categories", label: "Categories & rules", icon: Tags },
  { to: "/chat", label: "Ask FinPilot", icon: MessageCircle },
  { to: "/digests", label: "Digests", icon: Inbox },
  { to: "/agents", label: "Agents & Trail", icon: Workflow },
  { to: "/decisions", label: "Decisions (Jev)", icon: Scale },
  { to: "/evaluation", label: "Evaluation", icon: Gauge },
  { to: "/observability", label: "Observability", icon: Activity },
  { to: "/settings", label: "Settings", icon: Settings },
];

function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const toggle = () => {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("finpilot-theme", next ? "dark" : "light");
    } catch {
      /* ignore */
    }
    setDark(next);
  };
  return { dark, toggle };
}

function initials(name: string | undefined) {
  const parts = (name ?? "").trim().split(/\s+/).filter(Boolean);
  return (parts.length ? parts.slice(0, 2).map((p) => p[0]) : ["?"]).join("").toUpperCase();
}

function AccountPanel({ compact = false }: { compact?: boolean }) {
  const { user, authEnabled, signOut } = useAuth();
  const navigate = useNavigate();
  if (!authEnabled || !user) return null;
  return (
    <div className={cn("flex items-center gap-2.5", compact ? "" : "rounded-2xl border border-line bg-surface-2/60 p-2.5")}>
      <div className="grid size-8 shrink-0 place-items-center rounded-full bg-gradient-to-br from-emerald-400 to-indigo-500 text-[12px] font-semibold text-white">
        {initials(user.name)}
      </div>
      {!compact && (
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13px] font-medium" title={user.name}>
            {user.name}
          </div>
          <div className="flex min-w-0 items-center gap-1.5 text-[11.5px] text-muted">
            {user.is_demo && <Badge tone="amber" className="shrink-0">demo</Badge>}
            {user.is_admin && !user.is_demo && <Badge tone="violet" className="shrink-0">admin</Badge>}
            <span className="min-w-0 truncate" title={user.is_demo ? undefined : (user.email ?? undefined)}>
              {user.is_demo ? "Temporary sandbox" : user.email}
            </span>
          </div>
        </div>
      )}
      <Button
        variant="ghost"
        size="sm"
        aria-label="Sign out"
        title="Sign out"
        onClick={async () => {
          await signOut();
          navigate("/login", { replace: true });
        }}
      >
        <LogOut className="size-4" />
      </Button>
    </div>
  );
}

function DemoBanner() {
  const { user, signOut, config } = useAuth();
  const navigate = useNavigate();
  if (!user?.is_demo) return null;
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 border-b border-amber-500/20 bg-amber-500/10 px-4 py-2 text-center text-[12.5px] text-amber-700 dark:text-amber-300">
      <span>
        You're exploring a private demo account with sample data — it's deleted after {config?.demo_ttl_hours ?? 48} hours.
      </span>
      {config?.allow_signup && (
        <button
          className="font-semibold underline underline-offset-2"
          onClick={async () => {
            await signOut();
            navigate("/signup", { replace: true });
          }}
        >
          Create your own account
        </button>
      )}
    </div>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  const { data: meta } = useMeta();
  const run = useRunAgents();
  const { dark, toggle } = useTheme();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  const llmAgents = (meta?.agents ?? []).filter(
    (a) => a.model !== "rules" && a.model !== "langgraph" && a.model.includes(":") && !a.model.startsWith("hybrid"),
  );

  const nav = (
    <nav className="flex flex-col gap-0.5">
      {NAV.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          onClick={() => setOpen(false)}
          className={({ isActive }) =>
            cn(
              "flex items-center gap-3 rounded-xl px-3 py-2 text-[13.5px] font-medium transition",
              isActive ? "bg-surface-2 text-ink" : "text-muted hover:bg-surface-2/60 hover:text-ink",
            )
          }
        >
          {({ isActive }) => (
            <>
              <Icon className={cn("size-[18px]", isActive && "text-brand")} />
              {label}
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );

  const brand = (
    <div className="flex items-center gap-2.5 px-2">
      <div className="grid size-9 place-items-center rounded-xl bg-gradient-to-br from-emerald-400 to-indigo-500 text-white shadow-lg shadow-emerald-500/25">
        <Compass className="size-5" />
      </div>
      <div>
        <div className="text-[15px] font-semibold tracking-tight">FinPilot</div>
        <div className="text-[11px] text-muted">Proactive money co-pilot</div>
      </div>
    </div>
  );

  return (
    <div className="glow flex min-h-full">
      {/* desktop sidebar: the nav scrolls on short screens; the status and your account stay pinned */}
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col gap-4 border-r border-line bg-surface/60 p-4 backdrop-blur lg:flex">
        <div className="shrink-0">{brand}</div>
        <div className="-mx-1 min-h-0 flex-1 overflow-y-auto px-1">{nav}</div>
        <div className="shrink-0 space-y-2">
          <div className="space-y-1 rounded-2xl border border-line bg-surface-2/60 px-3 py-2 text-[11.5px] text-muted">
            <div className="flex items-center gap-1.5" title="How closed-set decisions are made">
              <Scale className="size-3.5 shrink-0 text-cyan-500" />
              {meta?.decisions && meta.decisions !== "off" ? (
                <span className="truncate">
                  Decisions: Jev via <span className="text-ink">{meta.decisions.split(":")[0]}</span>
                </span>
              ) : (
                <span className="truncate">Decisions: verified tools + LLM</span>
              )}
            </div>
            <div className="flex items-center gap-1.5" title="Agents that reason through an LLM API">
              <Bot className="size-3.5 shrink-0 text-brand" />
              {llmAgents.length ? (
                <span className="truncate">
                  {llmAgents.length} agents via <span className="text-ink">{llmAgents[0].model.split(":")[0]}</span>
                </span>
              ) : (
                <span className="truncate">Templates — add an LLM key</span>
              )}
            </div>
          </div>
          <AccountPanel />
        </div>
      </aside>

      {/* mobile drawer */}
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
          <aside className="absolute left-0 top-0 flex h-full w-72 flex-col gap-4 border-r border-line bg-surface p-4">
            <div className="shrink-0">{brand}</div>
            <div className="-mx-1 min-h-0 flex-1 overflow-y-auto px-1">{nav}</div>
            <div className="shrink-0">
              <AccountPanel />
            </div>
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <DemoBanner />
        <header className="sticky top-0 z-30 flex h-16 items-center gap-3 border-b border-line bg-canvas/80 px-4 backdrop-blur sm:px-6">
          <Button variant="ghost" size="sm" className="lg:hidden" onClick={() => setOpen(true)} aria-label="Menu">
            <Menu className="size-5" />
          </Button>
          <div className="min-w-0 flex-1 truncate text-sm text-muted">
            {meta ? (
              <>
                Hi <span className="font-medium text-ink">{meta.user.name.split(" ")[0]}</span> — your agents keep
                watching your money.
              </>
            ) : (
              "Loading…"
            )}
          </div>
          <Button variant="ghost" size="sm" onClick={toggle} aria-label="Toggle theme">
            {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </Button>
          <Button variant="primary" size="sm" loading={run.isPending} onClick={() => run.mutate({})}>
            {!run.isPending && <Play className="size-4" />}
            <span className="hidden sm:inline">Run agents</span>
          </Button>
        </header>
        <main key={loc.pathname.split("/")[1]} className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 sm:px-6 lg:py-8">
          {children}
        </main>
      </div>
    </div>
  );
}
