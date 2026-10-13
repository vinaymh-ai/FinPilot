import { useMutation } from "@tanstack/react-query";
import { ArrowRight, Bot, Compass, Eye, EyeOff, FileSpreadsheet, MailCheck, Scale, ShieldCheck, Sparkles } from "lucide-react";
import { useState, type FormEvent, type ReactNode } from "react";
import { Link, useLocation, useNavigate, useSearchParams, type LinkProps } from "react-router-dom";
import { Button, Input, Label } from "../components/ui";
import { api } from "../lib/api";
import { useAuth, type AuthUser } from "../lib/auth";

type Mode = "login" | "signup" | "forgot" | "reset";

function modeFor(path: string): Mode {
  if (path.startsWith("/signup")) return "signup";
  if (path.startsWith("/forgot-password")) return "forgot";
  if (path.startsWith("/reset-password")) return "reset";
  return "login";
}

/** Moving between sign-in, sign-up and reset keeps the page the visitor was originally going to. */
function AuthLink(props: LinkProps) {
  const loc = useLocation();
  return <Link state={loc.state} {...props} />;
}

const POINTS = [
  { icon: Bot, title: "Six agents, one plan", text: "Categorise, analyse, forecast, advise and brief you every week." },
  { icon: Scale, title: "Calibrated decisions", text: "Closed-set calls go to Jev with probabilities you can audit." },
  { icon: FileSpreadsheet, title: "Any bank statement", text: "CSV, Excel or PDF from any Indian bank — even password-protected." },
  { icon: ShieldCheck, title: "Private by design", text: "Your data stays in your account. Keys never leave the server." },
];

export default function AuthPage() {
  const loc = useLocation();
  const mode = modeFor(loc.pathname);
  return (
    <div className="glow grid min-h-screen lg:grid-cols-[1.05fr_1fr]">
      <aside className="relative hidden flex-col justify-between overflow-hidden border-r border-line bg-surface/50 p-10 lg:flex">
        <Brand />
        <div className="max-w-md space-y-7">
          <h1 className="text-[34px] font-semibold leading-tight tracking-tight">
            Your money, <span className="bg-gradient-to-r from-emerald-500 to-indigo-500 bg-clip-text text-transparent">watched over</span> by a team of agents.
          </h1>
          <ul className="space-y-4">
            {POINTS.map(({ icon: Icon, title, text }) => (
              <li key={title} className="flex gap-3">
                <div className="grid size-9 shrink-0 place-items-center rounded-xl bg-surface-2 text-brand">
                  <Icon className="size-[18px]" />
                </div>
                <div>
                  <div className="text-[14px] font-medium">{title}</div>
                  <div className="text-[13px] text-muted">{text}</div>
                </div>
              </li>
            ))}
          </ul>
        </div>
        <p className="text-[12px] text-muted">FinPilot gives information, not regulated financial advice.</p>
      </aside>
      <main className="flex items-center justify-center px-4 py-10 sm:px-8">
        <div className="w-full max-w-[400px]">
          <div className="mb-8 lg:hidden">
            <Brand />
          </div>
          {mode === "login" && <LoginForm />}
          {mode === "signup" && <SignupForm />}
          {mode === "forgot" && <ForgotForm />}
          {mode === "reset" && <ResetForm />}
        </div>
      </main>
    </div>
  );
}

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="grid size-10 place-items-center rounded-xl bg-gradient-to-br from-emerald-400 to-indigo-500 text-white shadow-lg shadow-emerald-500/25">
        <Compass className="size-5" />
      </div>
      <div>
        <div className="text-[16px] font-semibold tracking-tight">FinPilot</div>
        <div className="text-[11.5px] text-muted">Proactive money co-pilot</div>
      </div>
    </div>
  );
}

function Heading({ title, subtitle }: { title: string; subtitle?: ReactNode }) {
  return (
    <div className="mb-6">
      <h2 className="text-[22px] font-semibold tracking-tight">{title}</h2>
      {subtitle && <p className="mt-1 text-[13.5px] text-muted">{subtitle}</p>}
    </div>
  );
}

function FormError({ error }: { error: Error | null }) {
  if (!error) return null;
  return (
    <div role="alert" className="rounded-xl border border-red-500/25 bg-red-500/10 px-3 py-2 text-[13px] text-red-600 dark:text-red-400">
      {error.message}
    </div>
  );
}

function PasswordInput({
  value,
  onChange,
  autoComplete,
  placeholder,
  id,
}: {
  value: string;
  onChange: (v: string) => void;
  autoComplete: string;
  placeholder?: string;
  id: string;
}) {
  const [show, setShow] = useState(false);
  return (
    <div className="relative">
      <Input
        id={id}
        type={show ? "text" : "password"}
        value={value}
        required
        autoComplete={autoComplete}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="pr-10"
      />
      <button
        type="button"
        onClick={() => setShow(!show)}
        className="absolute right-2 top-1/2 grid size-7 -translate-y-1/2 place-items-center rounded-lg text-muted hover:text-ink"
        aria-label={show ? "Hide password" : "Show password"}
      >
        {show ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
      </button>
    </div>
  );
}

/** After sign-in the app shell takes over and returns to the page that asked for it. */
function useAfterSignIn() {
  const { signedIn } = useAuth();
  return (user: AuthUser) => signedIn(user);
}

function DemoButton() {
  const { config } = useAuth();
  const done = useAfterSignIn();
  const demo = useMutation({
    mutationFn: () => api.post<{ user: AuthUser }>("/auth/demo"),
    onSuccess: (r) => done(r.user),
  });
  if (!config?.demo_accounts) return null;
  return (
    <>
      <div className="my-6 flex items-center gap-3 text-[12px] text-muted">
        <div className="h-px flex-1 bg-line" /> or <div className="h-px flex-1 bg-line" />
      </div>
      <Button type="button" className="w-full" loading={demo.isPending} onClick={() => demo.mutate()}>
        <Sparkles className="size-4 text-brand" /> Try the demo — no sign-up
      </Button>
      <p className="mt-2 text-center text-[12px] text-muted">
        A private sandbox with 8 months of sample transactions, deleted after {config.demo_ttl_hours} hours.
      </p>
      {demo.error && (
        <div className="mt-3">
          <FormError error={demo.error} />
        </div>
      )}
    </>
  );
}

function LoginForm() {
  const done = useAfterSignIn();
  const { config } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const login = useMutation({
    mutationFn: () => api.post<{ user: AuthUser }>("/auth/login", { email, password, remember }),
    onSuccess: (r) => done(r.user),
  });
  const submit = (e: FormEvent) => {
    e.preventDefault();
    login.mutate();
  };
  return (
    <>
      <Heading
        title="Welcome back"
        subtitle={
          config?.allow_signup ? (
            <>
              New to FinPilot?{" "}
              <AuthLink to="/signup" className="font-medium text-brand hover:underline">
                Create an account
              </AuthLink>
            </>
          ) : undefined
        }
      />
      <form onSubmit={submit} className="space-y-4" noValidate>
        <div>
          <Label>E-mail</Label>
          <Input id="email" type="email" required autoComplete="email" autoFocus value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
        </div>
        <div>
          <div className="flex items-center justify-between">
            <Label>Password</Label>
            <AuthLink to="/forgot-password" className="mb-1.5 text-[12.5px] font-medium text-brand hover:underline">
              Forgot password?
            </AuthLink>
          </div>
          <PasswordInput id="password" value={password} onChange={setPassword} autoComplete="current-password" />
        </div>
        <label className="flex items-center gap-2 text-[13px] text-muted">
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} className="size-4 accent-emerald-600" />
          Keep me signed in on this device
        </label>
        <FormError error={login.error} />
        <Button variant="primary" type="submit" className="w-full" loading={login.isPending} disabled={!email || !password}>
          Sign in <ArrowRight className="size-4" />
        </Button>
      </form>
      <DemoButton />
    </>
  );
}

function SignupForm() {
  const done = useAfterSignIn();
  const { config } = useAuth();
  const min = config?.password_min_length ?? 8;
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const signup = useMutation({
    mutationFn: () => api.post<{ user: AuthUser }>("/auth/signup", { name: name || null, email, password }),
    onSuccess: (r) => done(r.user),
  });
  if (config && !config.allow_signup) {
    return (
      <>
        <Heading title="Sign-ups are closed" subtitle="Ask the administrator of this FinPilot server for an account." />
        <AuthLink to="/login" className="text-[13.5px] font-medium text-brand hover:underline">
          Back to sign in
        </AuthLink>
      </>
    );
  }
  return (
    <>
      <Heading
        title="Create your account"
        subtitle={
          <>
            Already have one?{" "}
            <AuthLink to="/login" className="font-medium text-brand hover:underline">
              Sign in
            </AuthLink>
          </>
        }
      />
      <form
        onSubmit={(e) => {
          e.preventDefault();
          signup.mutate();
        }}
        className="space-y-4"
        noValidate
      >
        <div>
          <Label hint="(optional)">Name</Label>
          <Input id="name" autoComplete="name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Asha" />
        </div>
        <div>
          <Label>E-mail</Label>
          <Input id="email" type="email" required autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
        </div>
        <div>
          <Label hint={`(at least ${min} characters)`}>Password</Label>
          <PasswordInput id="password" value={password} onChange={setPassword} autoComplete="new-password" />
          <StrengthMeter password={password} min={min} />
        </div>
        <FormError error={signup.error} />
        <Button variant="primary" type="submit" className="w-full" loading={signup.isPending} disabled={!email || password.length < min}>
          Create account <ArrowRight className="size-4" />
        </Button>
        <p className="text-center text-[12px] text-muted">Your statements and analysis are visible only to your account.</p>
      </form>
      <DemoButton />
    </>
  );
}

function StrengthMeter({ password, min }: { password: string; min: number }) {
  if (!password) return null;
  const classes = [/[a-z]/, /[A-Z]/, /\d/, /[^A-Za-z0-9]/].filter((r) => r.test(password)).length;
  const score = password.length < min ? 0 : Math.min(3, (password.length >= 12 ? 1 : 0) + (classes >= 3 ? 2 : classes >= 2 ? 1 : 0));
  const labels = ["Too short", "Weak", "Good", "Strong"];
  const colors = ["bg-red-500", "bg-amber-500", "bg-emerald-500", "bg-emerald-600"];
  return (
    <div className="mt-2 flex items-center gap-2">
      <div className="flex flex-1 gap-1">
        {[0, 1, 2].map((i) => (
          <div key={i} className={`h-1 flex-1 rounded-full ${i < Math.max(1, score) ? colors[score] : "bg-line"}`} />
        ))}
      </div>
      <span className="w-16 text-right text-[11.5px] text-muted">{labels[score]}</span>
    </div>
  );
}

function ForgotForm() {
  const { config } = useAuth();
  const [email, setEmail] = useState("");
  const forgot = useMutation({
    mutationFn: () => api.post<{ message: string }>("/auth/password/forgot", { email }),
  });
  if (forgot.isSuccess) {
    return (
      <>
        <div className="mb-4 grid size-12 place-items-center rounded-2xl bg-emerald-500/10 text-emerald-600">
          <MailCheck className="size-6" />
        </div>
        <Heading title="Check your inbox" subtitle={forgot.data.message} />
        {!config?.password_reset_by_email && (
          <p className="mb-4 rounded-xl border border-line bg-surface-2/60 p-3 text-[12.5px] text-muted">
            This server has no e-mail (SMTP) configured, so the reset link was written to the server log instead. Ask the
            administrator for it.
          </p>
        )}
        <AuthLink to="/login" className="text-[13.5px] font-medium text-brand hover:underline">
          Back to sign in
        </AuthLink>
      </>
    );
  }
  return (
    <>
      <Heading title="Reset your password" subtitle="We'll e-mail you a link to choose a new one. It's valid for an hour." />
      <form
        onSubmit={(e) => {
          e.preventDefault();
          forgot.mutate();
        }}
        className="space-y-4"
        noValidate
      >
        <div>
          <Label>E-mail</Label>
          <Input id="email" type="email" required autoComplete="email" autoFocus value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
        </div>
        <FormError error={forgot.error} />
        <Button variant="primary" type="submit" className="w-full" loading={forgot.isPending} disabled={!email}>
          Send reset link
        </Button>
        <AuthLink to="/login" className="block text-center text-[13px] font-medium text-muted hover:text-ink">
          Back to sign in
        </AuthLink>
      </form>
    </>
  );
}

function ResetForm() {
  const [params] = useSearchParams();
  const token = params.get("token") ?? "";
  const { config, signedIn } = useAuth();
  const navigate = useNavigate();
  const done = (user: AuthUser) => {
    signedIn(user);
    navigate("/", { replace: true });
  };
  const min = config?.password_min_length ?? 8;
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const mismatch = confirm.length > 0 && confirm !== password;
  const reset = useMutation({
    mutationFn: () => api.post<{ user: AuthUser }>("/auth/password/reset", { token, password }),
    onSuccess: (r) => done(r.user),
  });
  if (!token) {
    return (
      <>
        <Heading title="Reset link missing" subtitle="Open the link from your e-mail again, or request a new one." />
        <AuthLink to="/forgot-password" className="text-[13.5px] font-medium text-brand hover:underline">
          Request a new link
        </AuthLink>
      </>
    );
  }
  return (
    <>
      <Heading title="Choose a new password" subtitle="You'll be signed in, and signed out everywhere else." />
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!mismatch) reset.mutate();
        }}
        className="space-y-4"
        noValidate
      >
        <div>
          <Label hint={`(at least ${min} characters)`}>New password</Label>
          <PasswordInput id="password" value={password} onChange={setPassword} autoComplete="new-password" />
          <StrengthMeter password={password} min={min} />
        </div>
        <div>
          <Label>Confirm password</Label>
          <PasswordInput id="confirm" value={confirm} onChange={setConfirm} autoComplete="new-password" />
          {mismatch && <p className="mt-1 text-[12px] text-red-500">Passwords don't match.</p>}
        </div>
        <FormError error={reset.error} />
        <Button variant="primary" type="submit" className="w-full" loading={reset.isPending} disabled={password.length < min || mismatch || !confirm}>
          Set password and sign in
        </Button>
      </form>
    </>
  );
}
