import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode, Suspense, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Toaster } from "sonner";
import { ErrorBoundary, reportClientError } from "./components/ErrorBoundary";
import Layout from "./components/Layout";
import { Button, Skeleton, Spinner } from "./components/ui";
import "./index.css";
import { AuthProvider, useAuth } from "./lib/auth";
import { lazyWithRetry } from "./lib/lazy";

const Overview = lazyWithRetry(() => import("./pages/Overview"));
const Advisor = lazyWithRetry(() => import("./pages/Advisor"));
const Simulator = lazyWithRetry(() => import("./pages/Simulator"));
const Goals = lazyWithRetry(() => import("./pages/Goals"));
const Insights = lazyWithRetry(() => import("./pages/Insights"));
const Transactions = lazyWithRetry(() => import("./pages/Transactions"));
const Chat = lazyWithRetry(() => import("./pages/Chat"));
const Digests = lazyWithRetry(() => import("./pages/Digests"));
const Agents = lazyWithRetry(() => import("./pages/Agents"));
const Decisions = lazyWithRetry(() => import("./pages/Decisions"));
const Evaluation = lazyWithRetry(() => import("./pages/Evaluation"));
const SettingsPage = lazyWithRetry(() => import("./pages/Settings"));
const Observability = lazyWithRetry(() => import("./pages/Observability"));
const AuthPage = lazyWithRetry(() => import("./pages/Auth"));

const AUTH_PATHS = /^\/(login|signup|forgot-password|reset-password)(\/|$)/;

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 15_000 } },
});

function Fallback() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-8 w-64" />
      <div className="grid gap-4 md:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-28" />
        ))}
      </div>
      <Skeleton className="h-80" />
    </div>
  );
}

function Splash({ children }: { children?: ReactNode }) {
  return <div className="grid min-h-screen place-items-center p-6 text-center">{children ?? <Spinner className="size-6" />}</div>;
}

/** Signed out → the auth screens (remembering where the user was going); signed in → the app. */
function Gate() {
  const auth = useAuth();
  const location = useLocation();
  if (auth.status === "loading") return <Splash />;
  if (auth.status === "error") {
    return (
      <Splash>
        <div className="space-y-3">
          <div className="text-lg font-semibold">Can't reach the FinPilot server</div>
          <p className="text-sm text-muted">Check that the API is running, then try again.</p>
          <Button onClick={auth.retry}>Try again</Button>
        </div>
      </Splash>
    );
  }
  const onAuthPage = AUTH_PATHS.test(location.pathname);
  if (auth.status === "signed-out") {
    if (!onAuthPage) {
      return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />;
    }
    return (
      <ErrorBoundary resetKey={location.pathname}>
        <Suspense fallback={<Splash />}>
          <AuthPage />
        </Suspense>
      </ErrorBoundary>
    );
  }
  if (onAuthPage && !location.pathname.startsWith("/reset-password")) {
    // just signed in: continue to the page that asked for it
    const from = (location.state as { from?: string } | null)?.from;
    return <Navigate to={from && !AUTH_PATHS.test(from) ? from : "/"} replace />;
  }
  if (location.pathname.startsWith("/reset-password")) {
    return (
      <Suspense fallback={<Splash />}>
        <AuthPage />
      </Suspense>
    );
  }
  return <AppRoutes />;
}

function AppRoutes() {
  const location = useLocation();
  return (
    <Layout>
      <ErrorBoundary resetKey={location.pathname}>
        <Suspense fallback={<Fallback />}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/advisor" element={<Advisor />} />
            <Route path="/simulator" element={<Simulator />} />
            <Route path="/goals" element={<Goals />} />
            <Route path="/insights" element={<Insights />} />
            <Route path="/transactions" element={<Transactions />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/digests" element={<Digests />} />
            <Route path="/agents" element={<Agents />} />
            <Route path="/decisions" element={<Decisions />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/observability" element={<Observability />} />
            <Route path="*" element={<Overview />} />
          </Routes>
        </Suspense>
      </ErrorBoundary>
    </Layout>
  );
}

// Errors outside React's render (event handlers, timers, rejected promises) still reach the server log.
window.addEventListener("error", (e) => reportClientError(e.error ?? e.message, { source: "window.onerror" }));
window.addEventListener("unhandledrejection", (e) => reportClientError(e.reason, { source: "unhandledrejection" }));

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>
          <Gate />
        </AuthProvider>
      </BrowserRouter>
      <Toaster richColors position="top-right" closeButton />
    </QueryClientProvider>
  </StrictMode>,
);
