import {
  Bot,
  Compass,
  FlaskConical,
  Gauge,
  Inbox,
  LayoutDashboard,
  Lightbulb,
  Menu,
  MessageCircle,
  Moon,
  Play,
  Receipt,
  Scale,
  Settings,
  Sparkles,
  Sun,
  Target,
  Workflow,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { cn } from "../lib/format";
import { useMeta, useRunAgents } from "../lib/hooks";
import { Button } from "./ui";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/advisor", label: "Advisor", icon: Lightbulb },
  { to: "/simulator", label: "What-if Lab", icon: FlaskConical },
  { to: "/goals", label: "Goals", icon: Target },
  { to: "/insights", label: "Insights", icon: Sparkles },
  { to: "/transactions", label: "Transactions", icon: Receipt },
  { to: "/chat", label: "Ask FinPilot", icon: MessageCircle },
  { to: "/digests", label: "Digests", icon: Inbox },
  { to: "/agents", label: "Agents & Trail", icon: Workflow },
  { to: "/decisions", label: "Decisions (Jev)", icon: Scale },
  { to: "/evaluation", label: "Evaluation", icon: Gauge },
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

export default function Layout({ children }: { children: ReactNode }) {
  const { data: meta } = useMeta();
  const run = useRunAgents();
  const { dark, toggle } = useTheme();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  const llmAgents = meta?.agents.filter((a) => a.model !== "rules" && a.model !== "langgraph") ?? [];

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
      {/* desktop sidebar */}
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col gap-6 border-r border-line bg-surface/60 p-4 backdrop-blur lg:flex">
        {brand}
        {nav}
        <div className="mt-auto space-y-2 rounded-2xl border border-line bg-surface-2/60 p-3 text-[12px] text-muted">
          <div>
            <div className="mb-0.5 flex items-center gap-1.5 font-medium text-ink">
              <Scale className="size-3.5 text-cyan-500" /> Decisions
            </div>
            {meta?.decisions && meta.decisions !== "off" ? (
              <span>
                Jev via <span className="text-ink">{meta.decisions.split(":")[0]}</span> (System One)
              </span>
            ) : (
              <span>Rules — add a Mesh key to let Jev decide.</span>
            )}
          </div>
          <div>
            <div className="mb-0.5 flex items-center gap-1.5 font-medium text-ink">
              <Bot className="size-3.5 text-brand" /> Writing (LLM)
            </div>
            {llmAgents.length ? (
              <span>
                {llmAgents.length} agents via <span className="text-ink">{llmAgents[0].model.split(":")[0]}</span>
              </span>
            ) : (
              <span>Templates — add a Mesh key to enable LLM stages.</span>
            )}
          </div>
        </div>
      </aside>

      {/* mobile drawer */}
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
          <aside className="absolute left-0 top-0 flex h-full w-72 flex-col gap-6 border-r border-line bg-surface p-4">
            {brand}
            {nav}
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
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
        <main key={loc.pathname} className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 sm:px-6 lg:py-8">
          {children}
        </main>
      </div>
    </div>
  );
}
